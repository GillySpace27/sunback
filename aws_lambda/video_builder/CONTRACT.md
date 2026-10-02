# Public contract of the `the-sun-now` bucket

Everything below is read by something outside this repository: the landing page
at gilly.space/sun.html, Heliogram (0.6 and older read the fixed keys, 0.7 and
later read the index and the versioned keys), the R2 mirror at
imagery.myheliograph.com, the `sunback-run` wallpaper client and the hourly gate
in `.github/workflows/GitCloudRunHourly.yml`. Base URL:
`https://the-sun-now.s3.us-east-2.amazonaws.com/`.

**Rule: additive only.** A new key or field is added beside the old ones, with
its row here, in the same commit that writes it (for a fragment or index field,
also in `FRAGMENT_OPTIONAL` or `INDEX_OPTIONAL` in `manifest.py`;
`sunback/__tests__/test_manifest_contract.py` fails otherwise). Removing or
renaming a key or field needs Gilly's yes and a Heliogram release that no longer
reads it, first. Deprecated keys keep their last copy in the bucket.

Check the live bucket with `python devtools/scripts/smoke_public.py` (read-only
GET and HEAD requests).

## Product ids

`manifest.PRODUCTS` is the one product list. `serve_keys.SERVED_CHANNELS`, the
wave lists in `NRTFitsFetcher.py` and the `PRODUCTS` array in `web/sun.html` are
copies bound to it by `sunback/__tests__/test_product_catalog.py`. Ids are
strings without leading zeros.

| Id | Label | Kind |
|---|---|---|
| `rainbow` | Rainbow (RHEF composite) | composite of 171, 193, 211 |
| `171` | AIA 171 Å | AIA single channel |
| `193` | AIA 193 Å | AIA single channel |
| `211` | AIA 211 Å | AIA single channel |
| `304` | AIA 304 Å | AIA single channel |
| `335` | AIA 335 Å | AIA single channel |
| `94` | AIA 94 Å | AIA single channel |
| `131` | AIA 131 Å | AIA single channel |
| `1600` | AIA 1600 Å (UV) | AIA single channel |
| `1700` | AIA 1700 Å (UV) | AIA single channel |
| `composite_uv` | Composite (1700/1600/304) | composite |
| `dem` | Temperature map (DEM) | isothermal temperature map |

`manifest.AIA_IDS` lists the nine single-channel ids. Every image is RHEF
processed: a visualization, not a calibrated radiance.

## Fixed keys (rewritten in place)

| Key | Written by | Content-Type | Cache-Control | Notes | Read by |
|---|---|---|---|---|---|
| `1k/rhef_<id>_1k.png` | reducer (`AwsPutter`) | `image/png` | none | user metadata `obstime` (ISO UTC); its upload fires the Lambda | Heliogram 0.6 and older, `sunback-run` (lists `1k/`), the Lambda |
| `thumb/rhef_<id>_thumb.png` | reducer | `image/png` | none | 512 x 512 | landing page |
| `video/rhef_<id>_1k.mp4` | Lambda | `video/mp4` | none | 48 h timelapse; user metadata `through` (compact stamp of the newest frame) | landing page, Heliogram 0.6 and older |
| `video/rhef_tscan.mp4` | reducer | `video/mp4` | none | DEM temperature scan, not a timelapse | landing page (`dem` card) |
| `manifest/<id>.json` | Lambda | `application/json` | `no-cache` | one fragment, schema below | landing page, Heliogram fallback |
| `manifest/index.json` | Lambda | `application/json` | `public, max-age=300` | every fragment, schema below | Heliogram 0.7+, R2 mirror |
| `image_times.txt` | reducer | `text/plain` | none | `T_REC` of the integrated frame, UTC without a zone suffix | landing page, hourly gate |
| `image_times_readable.txt` | reducer | `text/plain` | none | **deprecated**: no known reader; still present (see below) | none known |

## Versioned keys (written once, never rewritten)

| Key | Written by | Cache-Control | Read by |
|---|---|---|---|
| `v/<id>/<stamp>.mp4` | Lambda | `public, max-age=31536000, immutable` | Heliogram 0.7+, R2 mirror |
| `v/<id>/<stamp>.png` | Lambda | `public, max-age=31536000, immutable` | Heliogram 0.7+, R2 mirror |

`<stamp>` is `YYYYMMDDTHHMMSS` (UTC) of the newest frame the object contains.
Per Heliogram's `infra/IMAGERY.md`, a lifecycle rule removes `v/` objects after
7 days; the index always names a current one. The rule lives in AWS, not in this
repository (SB-7 records it).

## Not part of the contract

- `frames/<id>/<YYYYMMDDTHHMMSS>_1k.png`: the Lambda's 48 h queue, pruned after
  about 49 h. No reader may depend on it.
- `staging/...`: a manual staging run (SB-5) writes the same key names under
  this prefix. Nothing reads it.

## Manifest fragment (`manifest/<id>.json`)

Built by `build_manifest_fragment`, checked by `validate_fragment`
(`manifest.py`). Unknown keys are errors.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | product id from the table above |
| `label` | string | yes | display label |
| `thumb` | string | yes | key of the 512 x 512 thumbnail |
| `img1k` | string | yes | key of the 1k still |
| `video` | string | yes | key of the fixed 48 h video |
| `updated` | string | yes | ISO UTC (`YYYY-MM-DDTHH:MM:SSZ`) observation time of the newest still |
| `frame_count` | integer | yes | distinct frames in the 48 h video |
| `integration` | object | yes | `frames` (integer) and `method` (string), from the Lambda environment |
| `video_v` | string | no | key of the immutable copy of the current video |
| `still_v` | string | no | key of the immutable copy of the newest still |
| `through` | string | no | ISO UTC time of the newest frame in the video; trails `updated` by up to the encode throttle |

## Index (`manifest/index.json`)

| Field | Type | Meaning |
|---|---|---|
| `generated` | string | ISO UTC time the index was rebuilt |
| `products` | array | the fragments, in `PRODUCTS` order; a product not built yet is absent |

`INDEX_OPTIONAL` in `manifest.py` lists added top-level fields; it is empty
today.

## Deprecated: `image_times_readable.txt`

A hand-built list of the capture time in several time zones. No consumer was
found: the landing page reads `image_times.txt` only, and a grep of the Website,
Heliogram and My Heliograph repositories found no reader (2026-10-01). The
reducer writes it while `SUNBACK_WRITE_READABLE_TIMES` is on (`NrtSettings`);
the last copy stays in the bucket when writing stops.

## status.json (SB-8)

| Key | Writer | Cache-Control | Body |
|---|---|---|---|
| `status.json` | `sun-video-builder`, `handler._write_index`, on every trigger, beside `manifest/index.json` | `public, max-age=60` | `{"generated": "YYYY-MM-DDTHH:MM:SSZ", "products": [{"id": str, "updated": str, "age_s": int}], "worst_age_s": int or null}`; products in `PRODUCTS` order; a product without a fragment or with an unparseable `updated` is absent; `worst_age_s` is null only when no product could be read |

- `manifest/index.json` top-level fields other than `generated` and `products` come only from `INDEX_EXTRA_KEYS` in `manifest.py` (none yet; SB-15 and SB-18 add theirs).
- A manual invoke whose trigger key starts with `staging/` reads and writes only under `staging/`; frames there are never pruned by code.
- Additive only: `status.json` and its fields are never renamed or removed.

## Provenance (SB-9)

| Where | Name | Type | Writer | Meaning |
|---|---|---|---|---|
| fragment `manifest/<id>.json` | `obs_start` | ISO str, optional | Lambda, from the still's S3 metadata | oldest frame of the newest still's integration window (composites: oldest of any input) |
| fragment `manifest/<id>.json` | `obs_end` | ISO str, optional | Lambda | newest frame of that window. The frame key's time and `updated` are the upload time (`obstime`), so `obs_end` is older than the key by the JSOC and integration latency |
| S3 metadata on `1k/rhef_<id>_1k.png` | `obstime` (existing, unchanged), plus new keys `obs_start`, `obs_end`, `tint_n`, `tint_m`; `rhef_stamp` once the reducer sends the RH-3 stamp | str | reducer `AwsPutter.do_upload` | `obstime` stays the upload time, as before SB-9. The observation time is carried in `obs_end` (header `T_REC` of the newest frame), `obs_start` the oldest frame of the window. The four new keys are omitted when no header time could be read |
| `meta/rhef_<id>.json` | schema.org `ImageObject` | JSON | reducer, every run, `Cache-Control: no-cache` | provenance sidecar; field names of the Solar Archive provenance JSON |
| PNG tEXt in `1k/rhef_<id>_1k.png` | `obs_start`, `obs_end`, `n_frames`, `method`, `sunkit_image_version`, `sunback_version` | text | reducer | the same window inside the file |
| MP4 tags in `video/rhef_<id>_1k.mp4`, `v/<id>/<through>.mp4` | `comment` (the RH-3 stamp string, only when the still's metadata carries `rhef_stamp`), `sunback_provenance` (compact JSON: `product`, `first`, `through`, `slots`, `frames`, `fps`, `cadence_s`, `integration`, `source`), `creation_time` (= `through`) | text | Lambda | the JSON never goes in `comment` (decision A4, 2026-10-02) |

**Open question (Gilly):** whether `obstime` should switch from upload time to the header
observation time (`obs_end`). Until he decides, `obstime` stays the upload time. It keys the
Lambda's frame (`frames/<id>/<stamp>_1k.png`, `v/<id>/<stamp>.png`) and the fragment's
`updated`, and the freshness thresholds (40 min / 2 h for the alarm, 60 / 180 min for the public
badge) are measured on that upload time. A switch would make public ages include JSOC latency, and
repeated observation times could overwrite an immutable `v/<id>/<stamp>.png` key, so it needs a
Lambda change first.

Additive only: none of these is ever renamed or removed. The MP4 custom tag needs
`-movflags +faststart+use_metadata_tags` in `_build_video`.

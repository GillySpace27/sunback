# "The Sun Right Now" — Revamp Design

**Date:** 2026-06-24
**Author:** Chris Gilly (with Claude)
**Status:** Design — awaiting review

## Goal

Replace the current single-image "The Sun Right Now" feature
(https://gilly.space/sun.html) with a thumbnail-grid landing page. Each of 7
products gets a card showing a live thumbnail and links to a **1k still** and a
**1k timelapse video** of the last 48 hours. Cadence rises from hourly to every 20
minutes (3 frames/hour → 144 frames per 48h window). The 1k still is upgraded with
**multi-frame time integration** to improve image quality.

## Background / current state

- **Scheduler:** [.github/workflows/GitCloudRunHourly.yml](../../../.github/workflows/GitCloudRunHourly.yml)
  — GitHub Actions, `cron: '0 * * * *'`, also runs on push to `master` + manual
  `workflow_dispatch`. Container `ghcr.io/gillyspace27/sunback-ci:latest`, AWS via OIDC.
- **Pipeline config:** [sunback/run/run_server_github.py](../../../sunback/run/run_server_github.py)
  — `do_one("rainbow", stop=True)` → runs ONE pass and exits (not a daemon on Actions).
- **Fetcher:** [sunback/fetcher/WebFitsFetcher.py:28](../../../sunback/fetcher/WebFitsFetcher.py)
  pulls from `synoptic/mostrecent/` — the **1024² synoptic NRT** product (single latest frame).
- **Uploader:** [sunback/putter/AwsPutter.py](../../../sunback/putter/AwsPutter.py)
  — wipes the `the-sun-now` bucket every run, then re-uploads.
- **Naming:** [sunback/processor/ImageProcessorCV.py:206](../../../sunback/processor/ImageProcessorCV.py)
  produces `DrGilly_*.png`; thumbnails are 512² via `make_thumbs` in
  [sunback/utils/array_util.py](../../../sunback/utils/array_util.py).
- **Repo:** `GillySpace27/sunback` — make **public** (one-time, manual) to zero Actions cost.

## Data-availability findings (measured 2026-06-24)

This drove the resolution decision. Most-recent record actually available per source:

| Source | Resolution | Lag | Notes |
|---|---|---|---|
| synoptic NRT (`H####` dirs) | 1024² FITS | ~minutes | 3-min cadence, **many recent frames** → enables integration |
| `aia.lev1` | 4096² FITS | ~4 days | too stale for "right now" |
| `aia.lev1_euv_12s` (FidoFetcher) | 4096² FITS | ~7 days | too stale |
| `aia.lev1_nrt` (full-res NRT) | 4096² FITS | n/a | not publicly accessible (401/invalid) |
| SDO browse JPEG | 4096² JPEG | ~minutes | standard color, not FITS, not RHEF-able |

**Conclusion:** science-grade 4k "right now" is impossible via public data. The live
product is **1024² synoptic NRT**. No 4k tier.

## Key decisions (locked)

| Decision | Choice |
|---|---|
| Run platform | Public repo → free Actions for the **reducer**; **in-AWS Lambda** for video |
| AWS post-processor | **Lambda** + static-ffmpeg layer (in `us-east-2`) |
| Products (12) | rainbow composite + AIA **171, 193, 211, 304, 335, 94, 131, 1600, 1700** + **UV composite** (1700/1600/304) + **DEM temperature map** (with a temperature-scan video link). *(Expanded from 8 post-design at user request — all were already computed each run.)* |
| Resolution tiers | **1k = 1024²** still, **thumb = 256²**, **1k video**. No 4k. |
| Source | 1024² synoptic NRT only |
| Time integration | N most-recent frames per wavelength, method ∈ {median, mean, sum}; defaults below |
| Video | H.264 MP4, 1k, **18 fps** (single tunable constant), 144-frame sliding 48h window |
| Cadence | every 20 min (`cron: '*/20 * * * *'`) |
| Landing page | Static page, JS reads S3 + per-product manifest fragments |
| Filenames | `DrGilly_*` → `rhef_*`, with resolution tag (`_1k`, `_thumb`) |

### Time-integration defaults
- `INTEGRATION_FRAMES` (N): **3** (covers ~9 min at 3-min cadence; tunable).
- `INTEGRATION_METHOD`: **median** (robust to cosmic rays / transient flares).
- Both are top-of-file constants on the new fetcher/processor; mean and sum selectable.

## Architecture

Two actors, split by cost and data co-location:

```
GitHub Actions (public repo, every 20 min)          AWS us-east-2 (event-driven)
┌─────────────────────────────────────┐             ┌──────────────────────────────┐
│ reducer (run_server_github.py)       │             │ Lambda video builder          │
│  NRTFitsFetcher: grab N recent       │  s3:Put     │  on s3:ObjectCreated of        │
│   synoptic frames per wave           │ ─────────▶  │   1k/rhef_<prod>_1k.png:       │
│  integrate (median/mean/sum)         │             │   1. append frame to 48h queue │
│  RHEF/Upsilon → 1k PNG per product   │             │   2. prune to newest 144       │
│  make 256² thumb                     │             │   3. ffmpeg queue → 1k MP4     │
│  upload 1k still + thumb             │             │   4. upload video              │
└─────────────────────────────────────┘             │   5. write manifest/<prod>.json│
                                                     └──────────────────────────────┘
   bucket: the-sun-now (us-east-2)
   ┌──────────────────────────────┐
   │ 1k/    rhef_<prod>_1k.png     │ ◀── reducer uploads
   │ thumb/ rhef_<prod>_thumb.png  │ ◀── reducer uploads
   │ frames/<prod>/<ts>_1k.png ×144│ ◀── Lambda maintains
   │ video/ rhef_<prod>_1k.mp4     │ ◀── Lambda uploads
   │ manifest/<prod>.json          │ ◀── Lambda writes (one per product → no race)
   └──────────────────────────────┘
```

**Why the Lambda still exists with no 4k/block-reduce:** ffmpeg needs all ~144 frames
present each cycle; with 7 products that's ~1,000 small files. Pulling them to an
Actions runner (outside AWS) would cost ~1 TB/mo egress (~$100/mo). Keeping the queue
+ ffmpeg **inside AWS** (S3→Lambda transfer is free, in-region) makes it ~$0 and fast.

## Components

### 1. `NRTFitsFetcher` (new fetcher)
- Lists the current (and, near the hour boundary, previous) `H####` directory under
  `https://jsoc1.stanford.edu/data/aia/synoptic/nrt/YYYY/MM/DD/`, parses the
  `AIA<date>_<HHMMSS>_<wave>.fits` filenames, and selects the **N most-recent frames
  per wavelength** for the 6 EUV channels.
- Downloads those N×6 frames. Reuses `WebFitsFetcher`'s download/retry plumbing
  (subclass or shared helper — `NRTFitsFetcher` extends `WebFitsFetcher`).
- Skips 1600/1700/4500.

### 2. Time-integration step — build a new `TimeIntegrationProcessor`
- **Existing component (`FidoTimeIntProcessor`) is NOT reusable here:** it is a
  `FidoFetcher` subclass that re-queries JSOC/Fido for subframes (the 7-day-lagged
  path we avoid), and its core (`sum_subframes`) is a running `+=` accumulator
  normalized by exposure time — **sum/mean only, no median** (median needs all frames
  stacked at once). Confirmed by reading the file.
- **Decision: build a small standalone `TimeIntegrationProcessor`** that takes the N
  frames already downloaded by `NRTFitsFetcher`, **stacks them and reduces** via the
  selected method (median / mean / sum). Reuse the DN/sec normalization idea from
  `FidoTimeIntProcessor.sum_subframes` but restructured for stack-then-reduce.
- Clean fetch/reduce separation; testable on synthetic stacks; decoupled from JSOC.
  Output: one integrated frame per wavelength feeding the RHEF/Upsilon → rainbow chain
  unchanged. `N=1` must reduce to current single-frame behavior (regression safety).

### 3. Reducer changes (`run_server_github.py` + processors)
- Swap `WebFitsFetcher` → `NRTFitsFetcher`; set `INTEGRATION_FRAMES` / `INTEGRATION_METHOD`.
- Produce a 1k RHEF still per wavelength **and** the rainbow composite (rainbow is
  already made by the same script from the per-wavelength RHEF data).
- Make a 256² thumbnail per product (drop 512² default in `make_thumbs`, or pass size).
- **Filename change:** `ImageProcessorCV.py:206` `"DrGilly_"` → `"rhef_"`; emit
  `1k/rhef_<prod>_1k.png` and `thumb/rhef_<prod>_thumb.png` keys.
- **Stop the bucket wipe:** `AwsPutter.empty_the_bucket()` must NOT run (it would delete
  the frame queue + videos). Replace with targeted overwrite of the `1k/` and `thumb/`
  keys only. The reducer no longer uploads videos.

### 4. Workflow change (`GitCloudRunHourly.yml`)
- `cron: '0 * * * *'` → `cron: '*/20 * * * *'`. Rename workflow (cosmetic).
- Repo set to **public** (manual, one-time).

### 5. Lambda video builder (new)
- **Trigger:** S3 `ObjectCreated`, prefix `1k/`, suffix `.png`.
- **Layer:** static ffmpeg binary. (numpy/Pillow not needed — no block-reduce.)
- **Steps per event (one 1k still = one product):**
  1. Copy the new 1k still into `frames/<prod>/<ISO-ts>_1k.png` (ts from filename/header).
  2. List `frames/<prod>/`, delete all but newest 144 (sliding 48h window).
  3. ffmpeg the 144 frames (sorted by ts) → `video/rhef_<prod>_1k.mp4` at `FPS=18`.
  4. Upload video.
  5. Write `manifest/<prod>.json` (keys below). One file per product → no write race.
- **Idempotency:** frame key derived from source timestamp → re-delivered events overwrite.
- **Errors:** per-product isolation; CloudWatch logs + DLQ for retries. Partial queue
  (<144) still encodes (video lengthens until the window fills).

### 6. Manifest (per-product fragments)
`manifest/<prod>.json` — avoids any shared-file race across the 7 concurrent Lambdas:
```json
{
  "id": "171", "label": "AIA 171 Å",
  "thumb": "thumb/rhef_171_thumb.png",
  "img1k": "1k/rhef_171_1k.png",
  "video": "video/rhef_171_1k.mp4",
  "updated": "2026-06-24T20:20:00Z",
  "frame_count": 144,
  "integration": {"frames": 3, "method": "median"}
}
```
Page fetches a known list of 7 fragment URLs (product ids are fixed) and merges client-side.

### 7. Landing page (`sun.html` revamp)
- Static page; JS fetches the 7 `manifest/<prod>.json` fragments, renders a responsive
  grid of 7 cards. Each card: 256² thumbnail (links to the 1k still), plus a link row
  `[1k] [▶ video]`. Video opens inline (`<video>`) or new tab.
- Keep the `image_times_readable.txt` time strip (confirmed in scope).
- Cache-busting: append `?v=<manifest.updated>` to asset URLs so the 20-min refresh
  is visible despite caching.

## Data flow (one cycle)

1. Actions fires (`*/20`). `NRTFitsFetcher` grabs N recent synoptic frames × 6 waves;
   integrate; RHEF/Upsilon; render 7 × 1k stills + 256² thumbs; upload `1k/` + `thumb/`.
2. Each `1k/` upload fires an S3 event → 7 Lambda invocations (parallel, per product).
3. Each Lambda rolls its frame queue, re-encodes the 1k video, uploads it, writes its
   manifest fragment.
4. Browser loads `sun.html`, fetches the 7 manifest fragments, renders the grid.

## Testing

- **NRTFitsFetcher:** unit-test directory parsing / N-most-recent selection per wave,
  including the hour-boundary case (reach into previous `H####`).
- **Integration:** assert median/mean/sum produce expected arrays on synthetic frames;
  assert N=1 reduces to today's behavior (regression safety).
- **Reducer:** assert filename mapping (`rhef_<prod>_1k.png`, `_thumb.png`), 256² thumb,
  7 expected keys, and that no bucket wipe occurs.
- **Lambda:** sample 1k PNG event → frame queue prunes to 144, ffmpeg emits a playable
  18-fps MP4, manifest fragment is valid JSON.
- **End-to-end:** `workflow_dispatch` one run against a staging prefix; verify the page
  renders 7 cards with working 1k/video links.
- **Cost guardrail:** confirm no Actions-side frame downloads (the egress trap).

## Cost summary

- **Actions:** $0 (public repo, unlimited standard-runner minutes).
- **Lambda:** ~2,190 runs × 7 ≈ 15k invocations/mo, seconds each → a few $/mo at most,
  likely within free tier.
- **S3:** storage ~ a few GB (7 × 144 frames + stills + videos); egress = public page
  traffic only (no internal frame shuffling). Pennies to low single digits.
- **Net:** ~$0–5/mo.

## Out of scope / deferred

- True 4k science imagery (not available NRT; revisit if a low-latency full-res series
  becomes accessible).
- Optional "View 4K (SDO)" browse-JPEG click-through — easy to add later if wanted.
- Migrating the reducer off Actions — unnecessary once the repo is public.

## Resolved during design (were open items)

- **Per-channel stills already emitted (#2):** `ImageProcessorCV` produces a PNG per
  wavelength; `RainbowRGBImageProcessor` only globs/composites them. No new emission
  path — just rename `DrGilly_*`→`rhef_*` + `_1k` tag, and filter `all_wavelengths`
  (`["0171","0193","0211","0304","0131","0335","0094","1600","1700"]`) to the 6 EUV
  channels we serve (171/193/211/304/335/94) + the composite.
- **Time integration (#1):** build a new standalone `TimeIntegrationProcessor`
  (see component 2); the existing `FidoTimeIntProcessor` is Fido-coupled and median-incapable.

## Open items to resolve in planning

1. Lambda packaging: container image vs. zip + ffmpeg layer.
2. Timestamp source for frame keys: parse from FITS filename vs. header `T_REC`.
3. Whether to also keep 131 as a 7th EUV single (currently in `all_wavelengths`) or
   hold to the agreed 6 EUV + composite = 7 cards.

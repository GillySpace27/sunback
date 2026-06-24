# "The Sun Right Now" — Revamp Design

**Date:** 2026-06-24
**Author:** Chris Gilly (with Claude)
**Status:** Design — awaiting review

## Goal

Replace the current single-image "The Sun Right Now" feature (https://gilly.space/sun.html)
with a thumbnail-grid landing page. Each of 7 products gets a card showing a live
thumbnail and links to a **4k** still, a **1k** still (block-reduced from the 4k,
*not* recomputed), and a **1k timelapse video** of the last 48 hours. Cadence rises
from hourly to every 20 minutes (3 frames/hour → 144 frames per 48h window).

## Background / current state

- **Scheduler:** [.github/workflows/GitCloudRunHourly.yml](../../../.github/workflows/GitCloudRunHourly.yml)
  — GitHub Actions, `cron: '0 * * * *'`, also runs on push to `master` and manual
  `workflow_dispatch`. Container `ghcr.io/gillyspace27/sunback-ci:latest`, AWS via OIDC.
- **Pipeline config:** [sunback/run/run_server_github.py](../../../sunback/run/run_server_github.py)
  — `do_one("rainbow", stop=True)` → runs ONE pass and exits (not a daemon on Actions).
- **Uploader:** [sunback/putter/AwsPutter.py](../../../sunback/putter/AwsPutter.py)
  — wipes the `the-sun-now` bucket (`empty_the_bucket()`) and re-uploads every run.
- **Naming:** [sunback/processor/ImageProcessorCV.py:206](../../../sunback/processor/ImageProcessorCV.py)
  produces `DrGilly_*.png`; thumbnails are 512² via `make_thumbs` in
  [sunback/utils/array_util.py](../../../sunback/utils/array_util.py).
- **Repo:** `GillySpace27/sunback` is currently **private** (Actions minutes billed).

## Key decisions (locked)

| Decision | Choice |
|---|---|
| Run platform | Public repo → free Actions for the **heavy reducer**; **in-AWS Lambda** for post-processing |
| AWS post-processor | **Lambda** + static-ffmpeg layer (+ Pillow), in `us-east-2` |
| Products (7) | rainbow composite + AIA **171, 193, 211, 304, 335, 94** |
| Resolutions | 4k = 4096², 1k = 1024² (exact ÷4 block-reduce of 4k), thumb = 256² |
| Video | H.264 MP4, 1k, **18 fps** (parameterized), 144-frame sliding 48h window |
| Cadence | every 20 min (`cron: '*/20 * * * *'`) |
| Landing page | Static page, JS reads S3 + a manifest |
| Filenames | `DrGilly_*` → `rhef_*`, with a resolution tag (`_4k` / `_1k`) |

## Architecture

Two actors, split by cost and co-location of data:

```
GitHub Actions (public repo, every 20 min)          AWS us-east-2 (event-driven)
┌─────────────────────────────────────┐             ┌──────────────────────────────┐
│ reducer (run_server_github.py)       │             │ Lambda post-processor         │
│  fetch JSOC → RHEF/Upsilon/DEM       │  s3:Put     │  on s3:ObjectCreated of        │
│  render 4k PNG per product           │ ─────────▶  │   4k/rhef_<prod>_4k.png:       │
│  upload ONLY 4k stills to S3         │             │   1. block-reduce 4k→1k        │
└─────────────────────────────────────┘             │   2. thumbnail 1k→256²         │
                                                     │   3. append 1k frame to 48h    │
   bucket: the-sun-now (us-east-2)                   │      queue; prune oldest       │
   ┌──────────────────────────────┐                 │   4. ffmpeg queue→1k MP4       │
   │ 4k/    rhef_<prod>_4k.png     │ ◀── uploads ────│   5. upload 1k, thumb, video   │
   │ 1k/    rhef_<prod>_1k.png     │                 │   6. rewrite manifest.json     │
   │ thumb/ rhef_<prod>_thumb.png  │                 └──────────────────────────────┘
   │ frames/<prod>/<ts>_1k.png ×144│
   │ video/ rhef_<prod>_1k.mp4     │
   │ manifest.json                 │
   └──────────────────────────────┘
```

**Why this shape:** ffmpeg needs all ~144 frames present each run; with 7 products
that's ~1,000 small files. Pulling them to an Actions runner (outside AWS) would cost
~1 TB/mo egress (~$100/mo). Keeping the video work **inside AWS** (S3→Lambda transfer
is free, in-region) makes it ~$0 and fast. The reducer stays on Actions because it
already works and is free on a public repo.

## Components

### 1. Reducer changes (`run_server_github.py` + processors)
- Produce a **4k RHEF still per wavelength** (171/193/211/304/335/94) in addition to
  the rainbow composite. (Confirm how the existing pipeline emits per-channel RHEF;
  may require iterating `rhe_targets` / `png_frame_name` over the 6 channels.)
- Stop wiping/re-uploading 1k/video/thumb from `AwsPutter`. The reducer now uploads
  **only** `4k/rhef_<prod>_4k.png`. The Lambda owns everything downstream.
- **Filename change:** `ImageProcessorCV.py:206` `"DrGilly_"` → `"rhef_"`, and append
  the resolution tag so 4k stills land as `rhef_<prod>_4k.png`.
- `AwsPutter.empty_the_bucket()` must NOT run (it would delete the frame queue and
  videos). Replace whole-bucket wipe with targeted overwrite of the 4k keys only.

### 2. Workflow change (`GitCloudRunHourly.yml`)
- `cron: '0 * * * *'` → `cron: '*/20 * * * *'`.
- Rename workflow to reflect 20-min cadence (cosmetic).
- Repo set to **public** (manual, one-time, outside code).

### 3. Lambda post-processor (new)
- **Trigger:** S3 `ObjectCreated` filtered to `4k/` prefix, `.png` suffix.
- **Layer:** static ffmpeg binary; runtime deps Pillow + numpy (block-reduce).
- **Steps per event** (one 4k object = one product):
  1. Download the new 4k PNG (in-region, free).
  2. `block_reduce` ÷4 → `1k/rhef_<prod>_1k.png`; downscale → `thumb/rhef_<prod>_thumb.png` (256²).
  3. Write the 1k frame to `frames/<prod>/<ISO-ts>_1k.png`; list `frames/<prod>/`,
     delete all but the newest 144 (sliding 48h window).
  4. ffmpeg the 144 frames (sorted by ts) → `video/rhef_<prod>_1k.mp4` at `FPS=18`
     (single top-of-file constant).
  5. Upload 1k, thumb, video.
  6. Update `manifest.json` (see below). Use a per-product write so concurrent
     product events don't clobber each other (read-modify-write with a product key,
     or one manifest fragment per product merged client-side — decide in planning).
- **Idempotency:** keying frames by source timestamp means a re-delivered event
  overwrites rather than duplicates.

### 4. Manifest (`manifest.json` in bucket)
Drives the static page without client-side bucket listing. Shape:
```json
{
  "updated": "2026-06-24T18:31:00Z",
  "products": [
    {"id": "rainbow", "label": "Rainbow (RHEF composite)",
     "thumb": "thumb/rhef_rainbow_thumb.png",
     "img4k": "4k/rhef_rainbow_4k.png",
     "img1k": "1k/rhef_rainbow_1k.png",
     "video": "video/rhef_rainbow_1k.mp4",
     "frame_count": 144, "latest_frame": "2026-06-24T18:20:00Z"},
    {"id": "171", "label": "AIA 171 Å", ...},
    ...
  ]
}
```

### 5. Landing page (`sun.html` revamp)
- Static page; JS fetches `manifest.json`, renders a responsive grid of 7 cards.
- Each card: 256² thumbnail (links to 1k by default), plus a small link row
  `[4k] [1k] [▶ video]`. Video opens inline (`<video>` modal) or new tab.
- Keeps the existing `image_times_readable.txt` time strip if desired.
- Cache-busting: append `?v=<manifest.updated>` to asset URLs so the 20-min refresh
  is visible despite S3/browser caching.

## Data flow (one cycle)

1. Actions fires (`*/20`). Reducer fetches latest FITS, computes RHEF per product,
   uploads 7 × `4k/rhef_<prod>_4k.png`.
2. Each upload fires an S3 event → 7 Lambda invocations (parallel, one per product).
3. Each Lambda makes 1k + thumb, rolls the frame queue, re-encodes the 1k video,
   uploads, and updates its slice of `manifest.json`.
4. Browser loads `sun.html`, fetches `manifest.json`, renders the grid.

## Error handling

- **Reducer fails / JSOC gap:** no 4k upload → no Lambda → page shows last good state
  (no wipe means nothing is destroyed). Acceptable.
- **Lambda fails on one product:** other products unaffected (per-product events).
  CloudWatch logs + a DLQ on the Lambda for retries.
- **Partial frame queue (< 144):** ffmpeg encodes whatever is present (page is live
  from frame 1; video lengthens until the 48h window fills).
- **Manifest race (7 concurrent Lambdas):** avoid lost updates — see component 3 note;
  simplest safe option is one fragment file per product (`manifest/<prod>.json`) and
  let the page merge, eliminating the shared-file race entirely. **Recommended.**

## Testing

- **Reducer:** unit-test the new filename mapping (`rhef_<prod>_4k.png`); a dry-run
  that asserts 7 expected 4k keys are produced and no bucket wipe occurs.
- **Lambda:** local test with a sample 4k PNG → assert 1k is exact ÷4, thumb is 256²,
  frame queue prunes to 144, ffmpeg emits a playable MP4 at 18 fps; assert manifest
  fragment is valid JSON.
- **End-to-end:** `workflow_dispatch` one run against a staging prefix/bucket; verify
  page renders 7 cards with working 4k/1k/video links.
- **Cost guardrail:** confirm no Actions-side frame downloads (the egress trap).

## Cost summary

- **Actions:** $0 (public repo, unlimited standard-runner minutes).
- **Lambda:** ~2,190 reducer runs × 7 products ≈ 15k invocations/mo, seconds each,
  small memory → a few dollars/mo at most; likely within free tier.
- **S3:** storage is ~7 × (4k+1k+video+144 frames) ≈ a few GB; egress = public
  page traffic only (no internal frame shuffling). Pennies to low single digits.
- **Net:** ~$0–5/mo, vs ~$30–100/mo if the video work ran on Actions.

## Out of scope / deferred

- Migrating the reducer itself off Actions (Fargate/always-on host) — not needed once
  the repo is public.
- Re-deriving 1k from FITS (explicitly block-reduced from 4k instead).
- Authentication / non-public assets (everything stays `public-read`).

## Open items to resolve in planning

1. Exact mechanism in the existing pipeline to emit per-channel 4k RHEF stills
   (iterate channels vs. a new processor path).
2. Manifest: per-product fragments (recommended) vs. single read-modify-write file.
3. Lambda packaging: container image vs. zip + ffmpeg layer.
4. Whether to keep the `image_times_readable.txt` time strip on the new page.

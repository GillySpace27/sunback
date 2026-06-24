# Sun-Right-Now bring-up & deploy

What's built (in this repo) and the remaining steps to make the live page work.
Spec: `docs/superpowers/specs/2026-06-24-sun-right-now-revamp-design.md`.

## Status

| Piece | State |
|---|---|
| NRT frame selection / hour-dir building | ✅ built + unit-tested (`sunback/fetcher/nrt_listing.py`) |
| Time integration (median/mean/sum) | ✅ built + unit-tested (`sunback/utils/time_integration.py`) |
| NRT→synoptic FITS integration | ✅ built + round-trip tested (`sunback/fetcher/nrt_integrate.py`) |
| `NRTFitsFetcher` | ✅ built (network shell — verify with a live run) |
| Reducer wired to NRT + integration | ✅ `run_server_github.py` |
| Bucket-wipe disabled | ✅ `AwsPutter.py` |
| Workflow cron `*/20` | ✅ `.github/workflows/GitCloudRunHourly.yml` |
| Lambda video builder | ✅ built (`handler.py`, queue/manifest unit-tested) |
| Landing page | ✅ `web/sun.html` |
| **Upload-key remap in `AwsPutter`** | ⚠️ **TODO — see below (page can't find images until done)** |
| AWS resources (Lambda, layer, S3 event, IAM) | ⚠️ manual, see below |
| Repo made public | ⚠️ manual |

## 1. Remaining code: `AwsPutter` upload-key remap (required)

The page + Lambda expect these keys; the reducer must produce them. This wasn't
auto-written because it depends on the exact local PNG filenames the pipeline emits
(per-wavelength vs. the composite), which needs a live run to confirm. Do this:

For each rendered PNG the reducer would upload:
- **Derive the product id** from the source name: `AIAsynoptic0171…` → `171`
  (strip leading zeros), the rainbow composite → `rainbow`.
- **Skip** any id not in the 8 served products
  (`rainbow,171,193,211,304,335,94,131`) — e.g. the composite-only 1600/1700 stills.
- Upload the full-res 1k PNG to **`1k/rhef_<id>_1k.png`**.
- Make a **256²** thumbnail (the global `make_thumbs` is 512² — pass a size or add a
  256 variant) and upload to **`thumb/rhef_<id>_thumb.png`**.
- Set S3 object **metadata `obstime`** = the frame's `T_REC` (ISO-8601). The Lambda
  reads this to timestamp queue frames; without it, it falls back to event time.
- **Do not upload any `.mp4`** — the Lambda owns `video/`. (Drop the mp4 branch in
  `do_upload` / `get_file_list`.)
- Keep `__save_times()` (writes `image_times_readable.txt`, used by the page).

`ContentType`/`ACL` stay as today (`image/png`, `public-read`, inline). The id and
key conventions match `aws_lambda/video_builder/manifest.py` — reuse those strings.

## 2. AWS resources (us-east-2, same region as bucket `the-sun-now`)

1. **ffmpeg layer:** publish a Lambda layer containing a static `ffmpeg` at
   `/opt/bin/ffmpeg` (e.g. from John Van Sickle's static build).
2. **Lambda function** `sun-video-builder`:
   - Runtime Python 3.12, handler `handler.handler`.
   - Package `aws_lambda/video_builder/*.py` (deps: boto3 is in the runtime).
   - Attach the ffmpeg layer. Memory ~1024 MB, timeout 120 s, ephemeral storage
     `/tmp` ≥ 512 MB (holds ≤144 small PNGs + one mp4 per product).
   - Env (optional overrides): `VIDEO_FPS=18`, `FRAME_WINDOW=144`,
     `INTEGRATION_FRAMES=3`, `INTEGRATION_METHOD=median`.
3. **IAM role** for the Lambda: `s3:GetObject,PutObject,DeleteObject,ListBucket`
   on `the-sun-now` (and `*/`). Public read is handled by object ACLs.
4. **S3 trigger:** bucket `the-sun-now` → event `s3:ObjectCreated:*`,
   prefix `1k/`, suffix `.png` → this Lambda. (8 stills/run ⇒ 8 invocations.)
5. **DLQ (optional):** an SQS dead-letter queue on the Lambda for retry visibility.

## 3. Make the repo public

Flip `GillySpace27/sunback` to public → unlimited free Actions minutes (the `*/20`
cadence is then $0). Nothing else depends on visibility.

## 4. Deploy the landing page

Publish `web/sun.html` to wherever `gilly.space/sun.html` is served. It is fully
static and reads only `https://the-sun-now.s3.us-east-2.amazonaws.com/` — no build.

## 5. End-to-end smoke test

1. `workflow_dispatch` one reducer run (or push to master).
2. Confirm `1k/rhef_171_1k.png` + `thumb/rhef_171_thumb.png` appear in the bucket.
3. Confirm the Lambda fired: `frames/171/<ts>_1k.png`, `video/rhef_171_1k.mp4`,
   `manifest/171.json` appear.
4. Open `sun.html` → 8 cards, thumbnails load, video plays, "Updated" shows now.
5. Let it run an hour → 3 frames/product accumulate; video lengthens toward 144.

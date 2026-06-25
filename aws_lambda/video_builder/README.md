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
| Upload-key remap in `AwsPutter` | ✅ done + unit-tested (`sunback/putter/serve_keys.py`) |
| `NRTFitsFetcher` live fetch+integrate | ✅ verified against live JSOC (2026-06-25) |
| AWS resources (Lambda, layer, S3 event, IAM) | ✅ **deployed** (us-east-2) + smoke-tested |
| Repo made public | ✅ done |

## 1. Cutover (remaining steps to go live)

All code is on branch `claude/amazing-wu-263fcf`; `master`/production is untouched
until you merge. The `AwsPutter` upload-key remap is **done** — it maps the real
reducer filenames (verified against the production bucket) to the served keys via
`sunback/putter/serve_keys.py`:

**12 served cards** (`serve_keys.py`), each → `1k/rhef_<id>_1k.png` +
`thumb/rhef_<id>_thumb.png` (256²) + an automatic 48h timelapse from the Lambda:
- `DrGilly_<wave>_ups(rhef).png` → ids `171,193,211,304,335,94,131,1600,1700`.
- `BGR_0171_0193_0211_ups(rhef).png` → **`rainbow`** (headline; swap `RAINBOW_SOURCE`
  for the other blend).
- `BGR_1700_1600_0304_ups(rhef).png` → **`composite_uv`**.
- `C_isothermal.png` → **`dem`** (isothermal temperature map).
- The DEM temperature-scan video (`a_temp_video_small.mp4`) is uploaded straight to
  `video/rhef_tscan.mp4` (bypasses the Lambda) and shown as a "T-scan" link on the
  DEM card.
- Skipped: visible-light 4500, and the still-loop ignores `.mp4`.
- `obstime` metadata = upload-time UTC (≈ observation time); the Lambda uses it to
  order the queue. `image_times_readable.txt` is still written.

To finish:
1. **Merge `claude/amazing-wu-263fcf` → `master`** (push triggers the `*/20` workflow).
2. **Deploy `web/sun.html`** to wherever `gilly.space/sun.html` is served.
3. **(optional) clean up old keys** — after cutover, the stale `renders/*` and
   `image_times.txt` from the old layout linger (not deleted, since we removed the
   bucket-wipe). Safe to delete once the new page is confirmed working.

## 2. AWS resources — automated by `deploy.sh`

Run `./deploy.sh` (awscli v2 + curl/tar/zip; creds that can manage Lambda/IAM/S3).
It is idempotent and does everything in this section: IAM role + S3 policy, the
ffmpeg layer, packaging the code as a `video_builder` package (handler
`video_builder.handler.handler`), create/update the function, and the S3
`1k/`→Lambda trigger. Override defaults via env (`REGION`, `FUNCTION`, etc.);
`SKIP_LAYER=1 ./deploy.sh` redeploys code only.

⚠️ The script **replaces** the bucket's notification config — if `the-sun-now`
already has notifications, merge them into the `notify.json` block first.

Manual reference (what the script sets up):

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

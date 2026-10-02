# video_builder: the sun-video-builder Lambda

## Current (2026-10-02, SB-1)

What `sun-video-builder` runs (us-east-2, python3.12, handler
`video_builder.handler.handler`), read from `handler.py`, `manifest.py` and
`frame_queue.py` in this directory. The zip holds exactly `__init__.py`,
`handler.py`, `frame_queue.py` and `manifest.py`.

- Trigger: S3 `ObjectCreated` on `1k/*.png` in `the-sun-now`, one invocation
  per still the reducer uploads (12 per reducer run).
- 12 products, in `manifest.PRODUCTS` order: `rainbow`, `171`, `193`, `211`,
  `304`, `335`, `94`, `131`, `1600`, `1700`, `composite_uv`, `dem`.
- Thumbnails are 512 x 512 (`THUMB_PX` in `sunback/putter/AwsPutter.py`); the
  reducer writes them, this Lambda does not.
- Per invocation: copy the still to `frames/<id>/<YYYYMMDDTHHMMSS>_1k.png`;
  prune frames more than `PRUNE_WINDOW_S` (49 h) older than the newest; snap
  the queue to a `GRID_CADENCE_S` (1200 s) grid of at most `FRAME_WINDOW` (144)
  slots, holding the previous frame through gaps; re-encode at most every
  `BUILD_THROTTLE_S` (7200 s) at `VIDEO_FPS` (18 fps); write
  `manifest/<id>.json`; rebuild `manifest/index.json`.
- Keys written, all public-read. Fixed keys, exactly as before:
  `video/rhef_<id>_1k.mp4` and `manifest/<id>.json` (`no-cache`). Immutable
  copies, never rewritten: `v/<id>/<stamp>.png` and `v/<id>/<stamp>.mp4`
  (`public, max-age=31536000, immutable`). One index of every fragment:
  `manifest/index.json` (`public, max-age=300`).
- Fragment fields: `id`, `label`, `thumb`, `img1k`, `video`, `updated`,
  `frame_count`, `integration` and, once known, `video_v`, `still_v`,
  `through`.
- Integration in each fragment comes from the Lambda environment:
  `INTEGRATION_FRAMES` (code default 5) and `INTEGRATION_METHOD` (default
  `median`). The reducer integrates 5 frames by median by default
  (`SUNBACK_INTEGRATION_FRAMES`, read in `sunback/run/run_server_github.py`).
- Colour: every MP4 is tagged bt709, tv range (`-vf setparams=...`,
  `-color_range tv`, `-colorspace bt709` in `_build_video`), so a player does
  not guess the matrix (Heliograph oversaturation, 2026-09-12). The argv is
  pinned by `sunback/__tests__/test_build_video_args.py`.
  `python aws_lambda/video_builder/colourcheck.py` measures it (not packed into
  the zip). Measured 2026-10-02 with ffmpeg version 6.1.1-3ubuntu5: mean
  |RGB| difference from the still 3.244 untagged, 1.013 tagged.
- Readers: gilly.space/sun.html and Heliograph 0.6 and older read the fixed
  keys; Heliograph 0.7+ and the R2 mirror read `v/` and `manifest/index.json`.
  Both sets are a public contract: add keys, never rename or remove one.
- Deploy: see "Deploying code changes" below. Never run `deploy.sh` or
  `deploy.py` against the live function: `deploy.py` publishes a new ffmpeg
  layer on every run, and both rewrite the function environment and the bucket
  notification configuration. Every real deploy is Gilly's call.
- Lifecycle: a 7-day expiry on `v/` is described in Heliograph's
  `infra/IMAGERY.md`; the rule is not in this repository.

## Deploying code changes (deploy_code.py)

Code reaches `sun-video-builder` only through `deploy_code.py`. It packs exactly
`__init__.py`, `handler.py`, `frame_queue.py` and `manifest.py` into a
deterministic zip and never touches the ffmpeg layer, the environment or the
bucket notification. Every real deploy and rollback needs Gilly's explicit yes.

1. After the change merges, tag the merge commit:
   `git tag -a lambda-YYYY-MM-DD -m "<what changed>"` (append `-2`, `-3` for a
   second deploy the same day). Pushing the tag is Gilly's call.
2. Read-only plan: `python aws_lambda/video_builder/deploy_code.py --plan`
   shows live versus new CodeSha256, the per-file diff and env drift against
   `lambda_env.json`, and prints `already live` when nothing differs.
3. Deploy (Gilly): `python aws_lambda/video_builder/deploy_code.py`, then type
   `sun-video-builder` at the prompt. It refuses a dirty tree, a HEAD not on
   `origin/master` and a HEAD without a `lambda-*` tag.
4. Commit the receipt it writes under `receipts/` (append-only).
5. Roll back (Gilly):
   `python aws_lambda/video_builder/deploy_code.py --rollback aws_lambda/video_builder/receipts/<stamp>.json`
   rebuilds the receipt's tag with `git archive` and goes through the same gate.

`lambda_env.json` records the live environment, one entry per name `handler.py`
reads. Changing an environment value is a separate, Gilly-approved
`aws lambda update-function-configuration`; new behaviour defaults correctly in
code instead.

The ffmpeg layer is pinned by `layer/ffmpeg.lock` (sha256 of the binary):
`python aws_lambda/video_builder/layer/build_layer.py --verify-only`.

`deploy.sh` and `deploy.py` bootstrap a new stack only. Never run them against
the live account: they publish a new layer, reset the environment and replace
the bucket notification configuration.

## History

The bring-up README below is moved here unchanged on 2026-10-02. It is out of
date: it describes 8 cards, 256 px thumbnails, `INTEGRATION_FRAMES=3` and a
cutover on `claude/amazing-wu-263fcf`. The "Deploying code changes" section
(SB-3, current) was kept above this heading, also unchanged.

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


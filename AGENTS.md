# AGENTS.md

Operating notes for any agent working in this repository. `CLAUDE.md` is a
symlink to this file: edit `AGENTS.md` only. Gilly owns this repository.

<!-- SU-6: the HelioSoftware suite preamble block goes directly below this line. -->

## What this repository is: one package, three roles

1. Research framework: runners in `sunback/run/` compose `Parameters`
   (`sunback/science/parameters.py`) with fetchers, processors and putters; Gilly runs them by hand.
2. Near-real-time publisher (production): `.github/workflows/GitCloudRunHourly.yml`
   runs `sunback/run/run_server_github.py` about every 20 minutes and writes
   to the public S3 bucket `the-sun-now`. The Lambda `sun-video-builder`
   (code in `aws_lambda/video_builder/`) turns each new 1k still into a
   48-hour movie plus the manifest JSON that gilly.space/sun.html, Heliograph
   and the R2 mirror read.
3. Wallpaper client: `pip install sunback`, console script `sunback-run`
   (`sunback/run/run_client_background.py`, `sunback/fetcher/S3ImgFetcher.py`,
   `sunback/putter/DesktopPutter.py`).

## Production files (a mistake here reaches the public within minutes)

- `.github/workflows/GitCloudRunHourly.yml`: until SB-2 lands, a push to master
  that touches code runs the production reducer against the live bucket.
- `sunback/run/run_server_github.py`, `sunback/run/run.py`, `sunback/science/parameters.py`
- `sunback/fetcher/NRTFitsFetcher.py`, `sunback/fetcher/nrt_listing.py`,
  `sunback/fetcher/nrt_integrate.py`, `sunback/utils/time_integration.py`
- `sunback/processor/SunPyProcessor.py`, `sunback/processor/ImageProcessorCV.py`,
  `sunback/processor/CompositeRainbowImageProcessor.py`, `sunback/processor/ScienceProcessor.py`
- `sunback/putter/AwsPutter.py`, `sunback/putter/serve_keys.py`
- `aws_lambda/video_builder/handler.py`, `aws_lambda/video_builder/manifest.py`,
  `aws_lambda/video_builder/frame_queue.py` (the Lambda zip)
- Public S3 key names and the manifest JSON shape are a contract with sun.html,
  Heliograph and the R2 mirror: add keys and fields, never rename or remove one.

## Never do these without Gilly's explicit yes, every time

- Push, merge, push a tag, deploy, publish or release. A yes covers one action only.
- Run `aws_lambda/video_builder/deploy.sh` or `aws_lambda/video_builder/deploy.py`
  against the live account: they publish a new ffmpeg layer, reset the Lambda
  environment and replace the bucket notification config.
- Run `sunback-serve` (`sunback/run/run_server_lingon.py`): it writes single,
  un-integrated frames over the production stills.
- Call `empty_the_bucket()` in `sunback/putter/AwsPutter.py`: it deletes every
  object in the bucket, including the Lambda frame queue.
- Dispatch the reducer workflow, or run anything else that writes to `the-sun-now`.
- Set the wallpaper on Gilly's Mac, or GUI-launch any Heliograph or Heliogram
  copy in Wall, Kiosk or Desktop mode.
- Read, print or commit a secret. The repository is public since 2026-06-25.

## Delete nothing

No `git rm`, no `rm` of a tracked file, no force-push, no history rewrite, no
branch, tag, S3 object, Lambda version or layer removal. Retire code with
`git mv` into `attic/` plus a line in its README; retire a branch with an
`archive/<branch>` tag. Before a multi-file refactor, tag `pre/<initiative-id>`.

## Local-only paths (gitignored; never commit them)

- `sunback_data/`: pipeline output rooted at the working directory (tens of GB on Gilly's Mac).
- `solar_archive_output/`: local Solar Archive output.
- `webapp/`: a separate git repository (the My Heliograph store, `sunback_webapp`).
  It is not part of this repository and must stay separate.

## How to check a change

- Every offline check: `bash devtools/check.sh` (prints `PASS`, `FAIL` or
  `UNCHECKED` per step; exits 1 only on a real failure).
- Use `python -m pytest`, not bare `pytest`: the stray `__init__.py` at the
  repository root breaks bare `pytest` collection. Until SB-2 lands,
  `sunback/__tests__/test_sunback.py` and `sunback/__tests__/test_parameters.py`
  cannot import and are skipped by path; every other test must pass.
- Offline fixtures: public-manifest fixtures (placeholders until recaptured, see
  the `_fixture_note` key) in `sunback/__tests__/fixtures/` and `sunback/__tests__/fixtures/make_fits.py` (a synthetic AIA-like FITS).
  Tests never need the network.
- A check that cannot fail proves nothing. Mark an unverifiable step UNCHECKED
  in the commit message instead of claiming it.

## Commands

- `bash devtools/check.sh`: every offline check (SB-4)
- `python devtools/scripts/smoke_public.py`: read-only probe of every public key of the-sun-now; exit 0 ok, 1 a failure, 3 unreachable (SB-6)
- `python infra/snapshot.py --out infra/live`: read-only, redacted snapshot of the AWS side; needs Gilly's AWS profile (SB-7)
- `python infra/diff.py`: drift between infra/declared/ and live AWS; exit 0 none, 1 drift, 3 unchecked (SB-7)
- `python infra/pat_expiry.py`: days until the dispatcher PAT expires; exit 1 within 14 days (SB-7)
- `python infra/rotate_dispatch_pat.py`: Gilly runs it in his own terminal; agents never run it (SB-7)
- `python devtools/reachability.py [--json]`: which tracked modules production, research or nothing reaches (SB-10)

## Where work lives

- One initiative per branch, `claude/<initiative-id>-<slug>`, started from
  master after its dependencies merge.
- The modernization plan for this repository lives in Gilly's research vault.
- Other sessions often work in this repository at the same time. Re-read a
  file right before editing it; never revert someone else's change.

## Writing rules

- No em dashes anywhere: code, comments, commit messages, docs.
- `sunback/sunback wishlist.txt` is canonical and must stay in sync with
  Gilly's vault. When your work completes or changes a wish-list item, say so
  in your report so Gilly can update both.
- RHEF output is visualization, not a calibrated radiance.

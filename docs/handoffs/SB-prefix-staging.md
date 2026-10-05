# SB-prefix hand-off: how to run a staging dispatch of the reducer

Written 2026-10-02 against claude/wave1-sunback. Decision (Gilly, 2026-10-02): wire
the input, add an offline audit, do not dispatch. Nothing in this branch dispatched
anything; the commands below are for Gilly.

## Warning: an empty prefix is production

`GitCloudRunHourly.yml` now has a `workflow_dispatch` input `prefix`. It is passed
to the reducer as `SUNBACK_PREFIX` (through `env:` on the "Execute the script" step).

- Empty (the default) means production: the reducer writes the live keys of
  `the-sun-now` (`1k/`, `thumb/`, `meta/`, `video/rhef_tscan.mp4`, `image_times.txt`,
  `image_times_readable.txt`) and the video Lambda fires.
- The EventBridge dispatcher sends no inputs, so it stays production every 20
  minutes, as intended. So does every push to master and every scheduled run.
- `gh workflow run GitCloudRunHourly.yml` without `-f prefix=...`, or the Actions
  "Run workflow" form with the box left blank, is a production run.

## Run a staging dispatch (Gilly's yes, every time)

```bash
gh workflow run GitCloudRunHourly.yml --ref <branch> -f prefix=staging/
gh run list --workflow GitCloudRunHourly.yml --branch <branch> --limit 3
```

- The prefix must be empty or like `staging/` or `staging/run-42/`: segments of letters,
  digits, `.`, `_`, `-`, each ending in `/`. `staging` (no slash), `/staging/`, `../`
  and spaces are refused by `NrtSettings.from_env` as the first thing the run does,
  before any download or upload (`test_workflow_prefix.py`).
- A dispatch always runs the reducer (the gate only skips scheduled runs).
- `--ref <branch>` runs that branch's code and workflow file, so the branch must
  contain this change. Whether the AWS role's trust policy accepts a non-master ref
  is UNCHECKED; if "Configure AWS credentials" fails, that is the first suspect.

## What to expect

- Written under `staging/` (nothing else): for each served product up to
  `staging/1k/rhef_<id>_1k.png`, `staging/thumb/rhef_<id>_thumb.png`,
  `staging/meta/rhef_<id>.json`; plus `staging/video/rhef_tscan.mp4` (when the DEM scan
  video exists), `staging/image_times.txt` and `staging/image_times_readable.txt`.
- No staging manifest, video or `status.json`: the video Lambda's S3 trigger matches
  prefix `1k/` only (`aws_lambda/video_builder/deploy.py`, notification filter), and
  `staging/1k/...` does not start with `1k/`. To exercise the Lambda under staging, invoke
  it by hand with a trigger key starting `staging/` (CONTRACT.md); that is Gilly's.
- Production keys are untouched. Check read-only:
  `python devtools/scripts/check_freshness.py --prefix staging/` for the staging side,
  and `python devtools/scripts/smoke_public.py` for the live side.
- Nothing cleans `staging/` up; the proposed lifecycle rule is in `infra/README.md`
  and is Gilly's decision. Agents never delete S3 objects.

## What the offline audit proves (and does not)

`sunback/__tests__/test_prefix_audit.py` drives `AwsPutter.put()` against a recorder of
every S3 call. With `SUNBACK_PREFIX=staging/` every key starts with `staging/`; with it
unset or empty the keys are exactly the production keys. A static guard fails when a new
S3 write call site appears under `sunback/` or `aws_lambda/` that is not on its audit
list. The audit found no write path that ignores the prefix. It does not prove what
GitHub passes in (that `inputs.prefix` is empty on push and schedule is GitHub's
documented behaviour, UNCHECKED here) or what the Lambda does in AWS.

## Correction to earlier task text

The task files for SB-8, SB-11 and SB-12 assumed a `prefix=staging/` dispatch input
already existed. It did not: before this change the workflow had `workflow_dispatch:`
with no inputs and never set `SUNBACK_PREFIX`, so a staging run was impossible and a
dispatch would have written production keys (found by the Mac session, 2026-10-02).

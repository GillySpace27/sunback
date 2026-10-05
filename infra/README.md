# infra: the AWS side of the live Sun, captured as data

The reducer runs on GitHub Actions and the video Lambda's code lives in
`aws_lambda/video_builder/`, but the rest of the live pipeline exists only in
the AWS account: the dispatcher Lambda, its EventBridge rule, the PAT secret,
the bucket's policy, lifecycle and notification, the two Lambda roles and the
Budget. This directory records them so they can be read, diffed and rebuilt.

Nothing here changes AWS. Every script is read-only except
`rotate_dispatch_pat.py`, which only Gilly runs.

## Files

| Path | What it is |
|---|---|
| `snapshot.py` | Reads every resource below (get, describe and list calls only) and writes one sorted JSON file per resource. Volatile fields are removed, the account id becomes `<ACCOUNT>`, token-shaped strings and sensitive environment values become `<REDACTED>`. Never reads a secret value or the Budget's subscribers. |
| `diff.py` | Takes a fresh snapshot in memory and diffs it against `declared/`. Exit 0 no drift, 1 drift, 3 UNCHECKED. |
| `pat_expiry.py` | Days until the dispatcher's GitHub token expires, from the `pat-expires` tag on its secret. Exit 1 inside 14 days. |
| `fetch_dispatcher.py` | Gilly runs it once (SB-7 Task 7). One `lambda:GetFunction` read, downloads the dispatcher's code zip to `/tmp/sb7-dispatcher` (refuses a folder inside the repository), checks it against the function's `CodeSha256`, unpacks it and scans it by count. Never prints the presigned URL. Exit 0 clean, 1 the scan found something or the zip does not match, 3 UNCHECKED. |
| `public_scan.py` | Counts token-, key-, account-id-, email- and `SecretString`-shaped text under the given paths and prints only counts per file, never a value. Exit 0 nothing, 1 something. Run it on `infra/live/` before any `declared/` commit. |
| `rotate_dispatch_pat.py` | Gilly only. Stores a new token, tags its expiry, fires one test dispatch. See `RUNBOOK-pat.md`. |
| `declared/` | The committed snapshot: what the account is meant to look like. Changed only by a PR that re-snapshots after a human change. NOT YET PRESENT: the first commit needs Gilly's yes (Q15), so `diff.py` answers UNCHECKED until then. |
| `live/` | Scratch output of `snapshot.py`. Gitignored; may hold drift you have not reviewed yet. |
| `dispatcher/` | A reviewed copy of the `sun-reducer-dispatcher` source, taken from the live function. Not deployed from here. NOT YET PRESENT: it needs a read of the live function with Gilly's AWS profile (SB-7 Task 7). |

## Commands (with Gilly's AWS profile; agents have no AWS access)

```bash
python infra/snapshot.py --out infra/live     # exit 0, or 3 with UNCHECKED sections
python infra/diff.py                          # exit 0 no drift, 1 drift, 3 unchecked
python infra/pat_expiry.py --warn-days 14     # exit 0 ok, 1 rotate soon or expired, 3 no tag
python infra/fetch_dispatcher.py              # once: the dispatcher code into /tmp/sb7-dispatcher, scanned
```

`python infra/public_scan.py <path>` needs no AWS and is safe for anyone to run.

Each takes `--region` (default `us-east-2`; the Budget is always read from
`us-east-1`, where the Budgets API lives).

## Resources captured

| File in `declared/` | Resource | Reads |
|---|---|---|
| `lambda-sun-video-builder.json` | Lambda `sun-video-builder` | configuration, resource policy (`s3invoke`), async invoke config |
| `lambda-sun-reducer-dispatcher.json` | Lambda `sun-reducer-dispatcher` | the same |
| `events-sun-reducer-20min.json` | EventBridge rule `sun-reducer-20min` | rule and targets |
| `s3-the-sun-now.json` | bucket `the-sun-now` | notification, policy, lifecycle, CORS, public-access block, ownership controls |
| `iam-role-<name>.json` | the role of each Lambda above | role and trust policy, inline policies, attached policies |
| `secret-github-actions-dispatch-token.json` | secret `github-actions-dispatch-token` | metadata and tags only |
| `budget-whole-account-monthly.json` | Budget `whole-account-monthly` | limit and notification thresholds, not subscribers |

Not captured: the OIDC role `Github_Web_Identity` the reducer assumes and its
identity provider (named in `.github/workflows/GitCloudRunHourly.yml`), the CI
image `ghcr.io/gillyspace27/sunback-ci` (SB-13), the R2 mirror and its Worker
(Heliogram repository, `infra/mirror/`), Cloudflare. Adding the OIDC role is
one more name in `snapshot.py`, on Gilly's word.

## How a change is made

1. Gilly says yes to the specific change.
2. A human makes it in the console or with the AWS CLI.
3. `python infra/snapshot.py --out infra/live`, review the diff
   (`python infra/diff.py`), then copy the changed files from `live/` to
   `declared/` in a PR. Never edit `declared/` by hand.

The repository is public. Before any `declared/` commit, check that no
12-digit number and nothing token-shaped is in the files:

```bash
grep -rEn '(^|[^0-9])[0-9]{12}([^0-9]|$)' infra/declared/ infra/live/
grep -rEn 'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|A[KS]IA[0-9A-Z]{16}' infra/declared/ infra/live/
```

Both must print nothing. `python infra/public_scan.py infra/live` runs the same
patterns plus email addresses and `SecretString` and must end `total: 0`.

## Rebuild order from an empty account

Read each step's values from the matching `declared/` file; replace `<ACCOUNT>`
with the new account id.

1. Bucket `the-sun-now` in `us-east-2`: ownership controls, public-access block,
   bucket policy, CORS (if any), lifecycle rules (`s3-the-sun-now.json`).
2. Role `sun-video-builder-role` with its inline policy `sun-bucket-access` and
   the managed `AWSLambdaBasicExecutionRole` (`iam-role-sun-video-builder-role.json`).
3. The ffmpeg layer: `python aws_lambda/video_builder/layer/build_layer.py --publish`
   (SB-3; gated on Gilly).
4. Lambda `sun-video-builder`: on an empty account only, the bootstrap path in
   `aws_lambda/video_builder/deploy.py`; otherwise `deploy_code.py` (SB-3).
   Environment from `aws_lambda/video_builder/lambda_env.json`.
5. The S3 to Lambda trigger: permission `s3invoke` and the bucket notification
   on prefix `1k/`, suffix `.png` (`lambda-sun-video-builder.json`, `s3-the-sun-now.json`).
6. Secret `github-actions-dispatch-token`: Gilly creates a token (`RUNBOOK-pat.md`)
   and stores it; tag `pat-expires`.
7. The dispatcher's role and Lambda `sun-reducer-dispatcher` from `dispatcher/`
   (`lambda-sun-reducer-dispatcher.json`, `iam-role-<dispatcher role>.json`).
8. EventBridge rule `sun-reducer-20min` with its target and the Lambda's invoke
   permission (`events-sun-reducer-20min.json`).
9. Budget `whole-account-monthly` with its thresholds; Gilly adds the email
   subscribers by hand.
10. Outside AWS: the OIDC role `Github_Web_Identity` (not captured), the CI image
    (SB-13), the R2 mirror (Heliogram repository).

## Proposed follow-ups (not applied; each needs Gilly's yes)

- A lifecycle rule that expires objects under `staging/` (SB-5's staging lane
  writes there and nothing cleans it). It deletes objects, so it is Gilly's
  decision alone (overview Q16); proposed expiry 7 days (estimated, matching
  the `v/` rule).
- Confirm the 7-day `v/` expiry that Heliogram's `infra/IMAGERY.md` describes
  against the `lifecycle` section of `declared/s3-the-sun-now.json`, and whether
  it should apply to new objects only (Q16).
- Add `Github_Web_Identity` to the snapshot.

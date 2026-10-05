# The reducer image

`ghcr.io/gillyspace27/sunback-ci` runs the production reducer
(`.github/workflows/GitCloudRunHourly.yml`, job `run-task`) and the `tests`
workflow. Plan: SB-13 in Gilly's research vault.

## State today (2026-10-05)

- Both workflows still use `ghcr.io/gillyspace27/sunback-ci:latest`, an image
  built by hand from `Dockerfile` (unpinned `python:3.12-slim`, apt `ffmpeg`,
  `requirements.txt` and `requirements-server.txt` resolved at build time),
  and both still run `pip install .`, which may resolve packages at run time.
- `.github/workflows/container.yml` has only the `freeze` job, which runs on
  manual dispatch only (Gilly's answer to Q3, 2026-10-02).
- `requirements-reducer.txt` does not exist yet. Everything below that needs it
  waits for the first freeze.

## Step 1: the freeze (Gilly)

```bash
gh workflow run container.yml -R GillySpace27/sunback
```

The job pulls the image production uses today inside a GitHub runner and keeps
its digest, Python version, `pip freeze --all`, ffmpeg line, Debian ffmpeg
version and `pip check` output as the artifact `freeze`. It writes nothing to
GHCR, S3 or AWS. Then, read only:

```bash
RUN=$(gh run list -R GillySpace27/sunback --workflow container.yml --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download "$RUN" -R GillySpace27/sunback -n freeze -D /tmp/sb13-freeze
```

## Step 2: lock, Dockerfile and pins (one PR, no behaviour change)

`devtools/scripts/reducer_lock.py` turns the freeze into the change. It edits
files in the checkout only.

```bash
python devtools/scripts/reducer_lock.py make-lock /tmp/sb13-freeze
python devtools/scripts/reducer_lock.py dockerfile /tmp/sb13-freeze --base-digest sha256:<python:3.12.N-slim index digest>
python devtools/scripts/reducer_lock.py pin "$(cat /tmp/sb13-freeze/image-digest.txt)"
python devtools/scripts/reducer_lock.py check      # PASS, or FAIL lines naming each problem
python -m pytest -q sunback/__tests__/test_reducer_lock.py
```

- `make-lock` writes `requirements-reducer.txt`: every line `name==version` or a
  git pin with a 40-hex commit, sunback itself left out, and a header with the
  source image digest, the date, ffmpeg, the Debian ffmpeg version, Python and
  the sunkit-image lineage (which RHEF build is canonical is Q6, RH-9). It stops
  on an editable, local-file or unpinned line: that package cannot be
  reinstalled from an index, and Gilly says where it came from.
- `dockerfile` pins `python:3.12.<n>-slim` by digest (the `<n>` comes from the
  freeze), pins apt `ffmpeg` to the frozen Debian version and installs only the
  lock with `--no-deps`. The old file stays below as comments. If the frozen
  image's `pip check` was not clean, `RUN pip check` is written as a comment
  that quotes the first problem, so the build reproduces the image.
- `pin` points both workflows at the hand-built image's own digest and changes
  `pip install .` to `pip install --no-deps .`. Production keeps running the
  same image, and a CI-built `:latest` can no longer move it.

The base digest is a public, read-only Docker Hub read of the
`python:3.12.<n>-slim` manifest index (`docker-content-digest` header).

## Step 3: verify and publish jobs (with the same PR)

Not written yet, because they need the lock and the pinned digest:

- `verify` (pull requests and pushes that touch the Dockerfile, the lock or
  `container.yml`): builds the candidate without pushing; its `pip freeze --all`
  must equal the lock, its `ffmpeg -version` the header, and
  `python -m devtools.scripts.pixel_probe` must render the same bytes in the
  candidate as in the image `GitCloudRunHourly.yml` pins (`--compare` prints
  `max_abs_diff`). A difference changes public imagery; only Gilly accepts it.
- `publish` (push to master only): pushes `sha-<gitsha>` and `latest` with an
  SBOM and provenance and prints the digest. Before it can push, Gilly gives the
  `sunback` repository Write access to the `sunback-ci` package (GitHub,
  Packages, sunback-ci, Manage Actions access).

## Step 4: move production to the CI-built image (second PR)

After the merge publishes a digest: Gilly dispatches a freeze of that digest
(`gh workflow run container.yml -f image=ghcr.io/gillyspace27/sunback-ci@sha256:<digest>`),
its `pip-freeze.txt` must diff empty against the lock, then
`python devtools/scripts/reducer_lock.py pin ghcr.io/gillyspace27/sunback-ci@sha256:<digest>`,
a staging run on that branch
(`gh workflow run GitCloudRunHourly.yml --ref <branch> -f prefix=staging/`, Gilly's yes),
and his merge.

## Rolling back

Revert the digest commit (`git revert <sha>`) and merge the revert. Every image
ever pushed stays in GHCR; no image or tag is deleted.

## Dependabot and alerts

`ci-pending/dependabot.yml` moves to `.github/dependabot.yml` in the same PR that
adds `requirements-reducer.txt` (`ci-pending/README.md`).
`python devtools/scripts/alert_triage.py` (needs `gh` with access to Dependabot
alerts) lists every open alert as `in-lock`, `pyproject`, `lambda` or
`not-installed`. Dismissing alerts and closing Dependabot PRs is Gilly's.

## Merging changes here runs the reducer

`GitCloudRunHourly.yml` runs on every push to master except changes under
`web/`, `docs/`, `aws_lambda/` or to `*.md`. A merge that touches the
Dockerfile, a workflow, `devtools/` or tests starts one production reducer run.

# Releasing sunback

## sunback on PyPI (SB-12)

Releases go to PyPI from a bare version tag through `.github/workflows/release.yml`.
An agent prepares the release commit and stops; every tag push, approval and upload
is Gilly's, one release at a time.

### One-time setup (Gilly, in web UIs)

1. GitHub, sunback, Settings, Environments: create `testpypi` (no reviewers) and
   `pypi` (Required reviewers: Gilly; deployment branches and tags: selected, tag
   rule `[0-9]*`).
2. pypi.org, project sunback, Publishing: add a trusted publisher with owner
   `GillySpace27`, repository `sunback`, workflow `release.yml`, environment `pypi`.
3. test.pypi.org, project sunback, Publishing: the same with environment `testpypi`.

### Each release

1. On a branch: set `version` in `pyproject.toml` (the only place it lives), move
   the `## Unreleased` lines in `CHANGELOG.md` under a heading for that version,
   run `bash devtools/check.sh` (needs `pip install build`), and merge by PR.
2. Gilly tags the merge commit on master and pushes the tag:
   `git tag -a 0.6.17.4 -m 0.6.17.4 <merge sha> && git push origin 0.6.17.4`.
3. The workflow checks that the tag equals the version and that the commit is on
   master, builds, audits the wheel, installs it outside the checkout, uploads to
   TestPyPI, then waits on the `pypi` environment.
4. Check the TestPyPI upload in a fresh venv outside the repository:
   `python3.11 -m venv /tmp/sb-rel && /tmp/sb-rel/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sunback==<version>`
   then `cd /tmp && /tmp/sb-rel/bin/python -c 'import sunback, sunback.science.color_tables, sunback.run.run_client_background; print(sunback.__version__)'`.
   Import only; running `sunback-run` changes the desktop picture.
5. Gilly approves the `pypi` job (Actions, the run, Review deployments) or rejects it.

### Rules

- Tags are bare versions (`0.6.17.4`), never `v*`. `v0.2.0` and `v1.0.0` are older
  tags and stay as they are.
- Never delete or move a tag, never yank or delete a PyPI or TestPyPI release. A bad
  release is followed by a new version.
- A release candidate (`0.6.17.4rc1`) goes through the same tag flow; reject its
  `pypi` job if TestPyPI was all you wanted.
- The production reducer installs from its checkout (`pip install .` in
  `GitCloudRunHourly.yml`), not from PyPI. A release changes only what
  `pip install sunback` users get.
- Build the wheel with plain `python -m build` (sdist first). `python -m build --wheel`
  in a checkout with an old `build/` directory reuses `build/lib` and ships stale files.
- `devtools/scripts/*.bat` are superseded and kept for reference.
- Author metadata in `pyproject.toml` and the meaning of `v1.0.0` are Gilly's to decide.

## Staging run of the reducer (SB-prefix)

A merge to master runs the production reducer, and so does a dispatch with an empty
`prefix`. To try a branch's reducer without touching production keys, Gilly dispatches
it with a prefix: `gh workflow run GitCloudRunHourly.yml --ref <branch> -f prefix=staging/`.
An empty prefix is production. What to expect and what not to: `docs/handoffs/SB-prefix-staging.md`.

## Release feed record (SU-11)

After Gilly has approved the `pypi` job and `https://pypi.org/project/sunback/<version>/` exists, record the
release in the HelioSoftware family feed (the Website checkout is `$SITE`, default `~/vscode/Website`; the
record format is `heliosoftware/spec/release-feed.md` there). The block reads the files and checksums from
PyPI's JSON and calls the append-only writer; it records nothing that PyPI does not list.

`CONFIRMED` stays `false` until someone has run this build: the TestPyPI install check in step 4 above counts,
because it installs the same files. The feed rule is that a page links an asset only when it is confirmed.

~~~bash
V=0.6.17.4        # the version just published
export SITE="${SITE:-$HOME/vscode/Website}" V
export CONFIRMED="${CONFIRMED:-false}"
curl -fsS "https://pypi.org/pypi/sunback/$V/json" | python3 -c '
import json, os, subprocess, sys, tempfile
d = json.load(sys.stdin)
v = d["info"]["version"]
files = [u for u in d["urls"] if u["packagetype"] in ("sdist", "bdist_wheel")]
args = []
for u in files:
    args += ["--asset", "name=%s,url=%s,sha256=%s,bytes=%d,platform=python,confirmed=%s"
             % (u["filename"], u["url"], u["digests"]["sha256"], u["size"], os.environ["CONFIRMED"])]
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(os.environ.get("NOTES", "sunback %s on PyPI." % v) + "\n")
cmd = [sys.executable, os.environ["SITE"] + "/heliosoftware/feed/append_record.py", "--product", "sunback",
       "--version", v, "--date", files[0]["upload_time_iso_8601"][:10], "--channel", "pypi", "--tag", v,
       "--url", "https://pypi.org/project/sunback/%s/" % v, "--notes-file", f.name] + args
sys.exit(subprocess.call(cmd))
'
python3 "$SITE/heliosoftware/feed/build_feed.py"
~~~

Exit 0 appended, 3 already recorded, 1 refused (the message names why). Set `NOTES` to a sentence without an
em dash to replace the default. Committing and pushing the Website is a separate yes from Gilly. The field
names (`urls`, `packagetype`, `digests`, `size`, `upload_time_iso_8601`) are PyPI's JSON API as read from its
documentation, not run against pypi.org: if the block fails on a missing key, print `d["urls"][0]` and adjust.

"""Invariants of .github/workflows/container.yml (SB-13). Stdlib only: regexes over the YAML text."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
WF = ROOT / ".github" / "workflows" / "container.yml"


def _text():
    assert WF.exists(), "container.yml missing"
    return WF.read_text()


def _jobs(text):
    body = text.split("\njobs:\n", 1)[1]
    return dict(re.findall(r"^  ([a-z-]+):\n(.*?)(?=^  [a-z-]+:\n|\Z)", body, re.M | re.S))


def test_freeze_job_records_the_image_without_publishing():
    text = _text()
    freeze = _jobs(text)["freeze"]
    for needle in ("docker pull", "pip freeze --all", "ffmpeg -version", "image-digest.txt", "name: freeze"):
        assert needle in freeze, needle
    assert re.search(r"^permissions:\n  contents: read$", text, re.M)
    assert "packages: write" not in text and "push: true" not in text
    assert "secrets." not in text


def test_freeze_waits_for_gillys_yes_before_it_runs_on_pull_requests():
    """Q3: a job that pulls the production image needs Gilly's yes. Until he gives it, only an explicit
    workflow_dispatch runs the freeze; no push or pull request pulls anything."""
    freeze = _jobs(_text())["freeze"]
    assert re.search(r"^    if: github\.event_name == 'workflow_dispatch'$", freeze, re.M), freeze[:200]


def test_actions_pinned_by_full_sha():
    for line in re.findall(r"^\s*-?\s*uses:\s*(\S+.*)$", _text(), re.M):
        assert re.match(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}\s+# v\d", line), line

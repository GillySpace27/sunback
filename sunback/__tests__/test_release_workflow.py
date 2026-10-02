"""Invariants of .github/workflows/release.yml (SB-12). Stdlib only: regexes over the YAML text."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
WF = ROOT / ".github" / "workflows" / "release.yml"


def _text():
    assert WF.exists(), "release.yml missing"
    return WF.read_text()


def _jobs(text):
    body = text.split("\njobs:\n", 1)[1]
    return dict(re.findall(r"^  ([a-z-]+):\n(.*?)(?=^  [a-z-]+:\n|\Z)", body, re.M | re.S))


def test_triggers_only_on_bare_version_tags():
    text = _text()
    on = text.split("\non:\n", 1)[1].split("\n\n", 1)[0]
    assert re.fullmatch(r"  push:\n    tags:\n      - '\[0-9\]\*'", on), on


def test_jobs_run_build_then_testpypi_then_pypi():
    jobs = _jobs(_text())
    assert list(jobs) == ["build", "testpypi", "pypi"]
    assert "needs: build" in jobs["testpypi"]
    assert "needs: testpypi" in jobs["pypi"]


def test_build_job_checks_before_anything_is_uploaded():
    build = _jobs(_text())["build"]
    for needle in ("python -m build", "check_wheel.py dist/*.whl", "GITHUB_REF_NAME", "merge-base --is-ancestor"):
        assert needle in build, needle
    assert "id-token" not in build


def test_publish_jobs_use_trusted_publishing_and_their_environments():
    text = _text()
    jobs = _jobs(text)
    assert "name: testpypi" in jobs["testpypi"] and "repository-url: https://test.pypi.org/legacy/" in jobs["testpypi"]
    assert "name: pypi" in jobs["pypi"] and "repository-url" not in jobs["pypi"]
    for name in ("testpypi", "pypi"):
        assert "id-token: write" in jobs[name], name
    assert "secrets." not in text and "password" not in text
    assert re.search(r"^permissions:\n  contents: read$", text, re.M)


def test_actions_pinned_by_full_sha():
    for line in re.findall(r"^\s*-?\s*uses:\s*(\S+.*)$", _text(), re.M):
        assert re.match(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}\s+# v\d", line), line

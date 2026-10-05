"""devtools/scripts/alert_triage.py classifies alerts without the network (SB-13)."""

import json
import subprocess
import sys

from devtools.scripts import alert_triage as at

LOCK = "# source-image: x\npip==25.0\nPillow==11.1.0\nsunkit-image @ git+https://github.com/o/r.git@" + "a" * 40 + "\n"
PYPROJECT = """
[project]
name = "sunback"
dependencies = ["boto3", "tqdm", "pyobjc-framework-Cocoa; sys_platform == 'darwin'"]
[project.optional-dependencies]
server = ["twine"]
"""


def test_lock_names_normalize_and_skip_comments():
    assert at.lock_names(LOCK) == {"pip", "pillow", "sunkit-image"}


def test_pyproject_names_include_extras():
    assert at.pyproject_names(PYPROJECT) == {"boto3", "tqdm", "pyobjc-framework-cocoa", "twine"}


def test_classes_first_match_wins():
    alerts = [
        {"number": 1, "package": "pillow", "severity": "high"},
        {"number": 2, "package": "twine", "severity": "low"},
        {"number": 3, "package": "botocore", "severity": "medium"},
        {"number": 4, "package": "Flask", "severity": "high"},
        {"number": 5, "package": "boto3", "severity": "low"},
    ]
    rows = at.classify(alerts, at.lock_names(LOCK), at.pyproject_names(PYPROJECT))
    assert [r["class"] for r in rows] == ["in-lock", "pyproject", "lambda", "not-installed", "pyproject"]
    summary = at.summarize(rows)
    assert summary["in-lock"] == {"count": 1, "high_or_critical": 1}
    assert summary["not-installed"] == {"count": 1, "high_or_critical": 1}


def test_without_gh_the_result_is_unchecked(tmp_path):
    code = "import sys; from devtools.scripts import alert_triage as at; sys.exit(at.main(['--json']))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PATH": str(tmp_path)})
    assert r.returncode == 3, r.stdout + r.stderr
    assert json.loads(r.stdout)["status"] == "UNCHECKED"

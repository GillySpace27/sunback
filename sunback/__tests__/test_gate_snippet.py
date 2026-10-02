"""The hourly workflow's gate step must decide the same under `bash -e` (the default shell)
and `bash -eo pipefail` (what `shell: bash` adds). Review nit on GitCloudRunHourly.yml.

The step's script is cut out of the workflow text and run with a fake check_freshness.py
in a scratch directory. Nothing touches the network or AWS.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / ".github" / "workflows" / "GitCloudRunHourly.yml"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

SHELLS = {"bash -e": ["bash", "-e"], "bash -eo pipefail": ["bash", "-eo", "pipefail"]}

# (name, fake stdout, fake exit code, expected run=). Stale and missing exit 1, unreachable 3,
# like the real script; a crash prints nothing.
FAKES = [
    ("stale", '{"worst_age_s": 5000}', 1, "true"),
    ("fresh", '{"worst_age_s": 100}', 0, "false"),
    ("unreachable", '{"worst_age_s": null}', 3, "true"),
    ("crash", "", 2, "true"),
]


def gate_script():
    text = WF.read_text(encoding="utf-8")
    m = re.search(r"- name: Decide whether the reducer needs to run\n(?:.*\n)*?        run: \|\n((?:          .*\n|\n)+)", text)
    assert m, "decide step not found"
    return "\n".join(line[10:] for line in m.group(1).splitlines())


def run_gate(tmp_path, shell, stdout, code, event="schedule"):
    scripts = tmp_path / "devtools" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "check_freshness.py").write_text(
        f"import sys\nprint({stdout!r})\nsys.exit({code})\n", encoding="utf-8")
    out = tmp_path / "github_output"
    out.write_text("", encoding="utf-8")
    link = tmp_path / "bin"  # python3 on PATH is the interpreter running the tests
    link.mkdir()
    (link / "python3").symlink_to(sys.executable)
    env = {"PATH": f"{link}:/usr/bin:/bin", "GITHUB_EVENT_NAME": event, "GITHUB_OUTPUT": str(out)}
    proc = subprocess.run(shell + ["-c", gate_script()], cwd=tmp_path, env=env, capture_output=True, text=True)
    return proc, out.read_text(encoding="utf-8")


@pytest.mark.parametrize("name,stdout,code,expected", FAKES)
@pytest.mark.parametrize("shell", SHELLS)
def test_gate_decides_the_same_under_both_shells(tmp_path, shell, name, stdout, code, expected):
    proc, outputs = run_gate(tmp_path, SHELLS[shell], stdout, code)
    assert proc.returncode == 0, proc.stderr
    assert outputs.strip() == f"run={expected}", (name, shell, proc.stdout, proc.stderr)


@pytest.mark.parametrize("shell", SHELLS)
def test_gate_always_runs_on_dispatch(tmp_path, shell):
    proc, outputs = run_gate(tmp_path, SHELLS[shell], '{"worst_age_s": 1}', 0, event="workflow_dispatch")
    assert proc.returncode == 0 and outputs.strip() == "run=true"


def test_gate_job_is_read_only():
    text = WF.read_text(encoding="utf-8")
    gate = text[text.index("  gate:\n"):text.index("  run-task:\n")]
    assert re.search(r"^    permissions:\n      contents: read$", gate, re.M)

"""SB-prefix: GitCloudRunHourly.yml passes a `prefix` dispatch input to the reducer as SUNBACK_PREFIX.

Decision (Gilly, 2026-10-02): wire the input, add an offline audit, do not dispatch.
Before this, the workflow declared `workflow_dispatch:` with no inputs and never set
SUNBACK_PREFIX, so a staging run was impossible and a plain dispatch wrote production keys.

The workflow is parsed as YAML; nothing is dispatched and nothing reaches AWS or GitHub.
"""
import importlib
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from sunback.settings import NrtSettings

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / ".github" / "workflows" / "GitCloudRunHourly.yml"
BAD_PREFIXES = ["/staging/", "staging", "staging//", "../", "a/../", " staging/", "staging/ ", "stag ing/"]


def load():
    doc = yaml.safe_load(WF.read_text(encoding="utf-8"))
    triggers = doc.get("on", doc.get(True))   # PyYAML reads the bare key `on` as boolean True
    return doc, triggers


def execute_step(doc):
    steps = [s for s in doc["jobs"]["run-task"]["steps"] if s.get("name") == "Execute the script"]
    assert len(steps) == 1
    return steps[0]


def test_dispatch_declares_an_optional_prefix_input_defaulting_to_empty():
    _, triggers = load()
    spec = triggers["workflow_dispatch"]["inputs"]["prefix"]
    assert spec["required"] is False
    assert spec["default"] == ""
    assert "SUNBACK_PREFIX" in spec["description"] and "staging/" in spec["description"]
    assert "empty means production" in spec["description"]
    assert list(triggers["workflow_dispatch"]["inputs"]) == ["prefix"]


def test_prefix_reaches_the_reducer_only_through_env():
    doc, _ = load()
    step = execute_step(doc)
    assert step["env"] == {"SUNBACK_PREFIX": "${{ inputs.prefix }}"}
    assert step["run"].strip() == "python sunback/run/run_server_github.py"
    # never interpolated into a script (shell injection) and never set anywhere else
    for job_name, job in doc["jobs"].items():
        assert "SUNBACK_PREFIX" not in (job.get("env") or {}), job_name
        for s in job["steps"]:
            assert "inputs." not in s.get("run", ""), (job_name, s.get("name"))
            assert "${{" not in s.get("run", "") or "inputs" not in s["run"], (job_name, s.get("name"))
            if s is not step and s.get("name") != "Refuse a staging prefix on a scheduled run":
                assert "SUNBACK_PREFIX" not in (s.get("env") or {}), (job_name, s.get("name"))
    assert "SUNBACK_PREFIX" not in (doc.get("env") or {})


def test_schedule_trigger_is_unchanged_and_push_is_gone():
    """SB-2 Task 6 (Gilly, 2026-10-12) removed the push trigger; test_reducer_workflow.py pins the rest."""
    _, triggers = load()
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "0 * * * *"}]


def test_rest_of_the_workflow_is_unchanged():
    doc, _ = load()
    assert list(doc["jobs"]) == ["gate", "run-task"]
    assert [s.get("name") or s.get("uses") for s in doc["jobs"]["run-task"]["steps"]] == [
        "Refuse a staging prefix on a scheduled run",
        "Check out repository", "Install sunback package from checked out code",
        "Configure AWS credentials for OIDC", "Execute the script"]
    assert doc["jobs"]["run-task"]["needs"] == "gate"
    assert doc["jobs"]["run-task"]["if"] == "needs.gate.outputs.run == 'true'"
    assert doc["jobs"]["run-task"]["permissions"] == {"id-token": "write", "contents": "read"}
    assert [s.get("name") for s in doc["jobs"]["gate"]["steps"]] == [
        "Check out repository (for devtools/scripts/check_freshness.py)",
        "Decide whether the reducer needs to run"]


@pytest.mark.parametrize("value,prefix", [("", ""), ("staging/", "staging/"), ("staging/run-42/", "staging/run-42/")])
def test_what_the_workflow_passes_becomes_the_reducer_prefix(value, prefix):
    """A schedule passes '' (inputs.prefix is empty there): production, byte for byte."""
    assert NrtSettings.from_env({"SUNBACK_PREFIX": value}).prefix == prefix
    assert NrtSettings.from_env({"SUNBACK_PREFIX": value}).prefixed("1k/rhef_171_1k.png") == prefix + "1k/rhef_171_1k.png"


@pytest.mark.parametrize("bad", BAD_PREFIXES)
def test_a_bad_prefix_is_refused_before_any_write(monkeypatch, bad):
    with pytest.raises(ValueError, match="SUNBACK_PREFIX"):
        NrtSettings.from_env({"SUNBACK_PREFIX": bad})

    rsg = importlib.import_module("sunback.run.run_server_github")
    aws = importlib.import_module("sunback.putter.AwsPutter")
    touched = []

    class Refusing:
        def __getattr__(self, name):
            touched.append(name)
            raise AssertionError(f"S3 call {name} after a bad prefix")

    class NoRunner:
        def __init__(self, *a, **k):
            touched.append("SingleRunner")

    monkeypatch.setattr(aws, "_S3_CLIENT", Refusing())
    monkeypatch.setattr(rsg, "SingleRunner", NoRunner)
    monkeypatch.setenv("SUNBACK_PREFIX", bad)
    with pytest.raises(ValueError, match="SUNBACK_PREFIX"):
        rsg.run_server_github()
    assert touched == []


# SB-5 Task 6: a scheduled run refuses a non-empty prefix. A schedule has no inputs, so
# inputs.prefix is '' there today; the guard keeps it that way if that ever changes.
def _guard_step(doc):
    steps = [s for s in doc["jobs"]["run-task"]["steps"]
             if s.get("name") == "Refuse a staging prefix on a scheduled run"]
    assert len(steps) == 1, f"expected one guard step in {WF.name}, found {len(steps)}"
    return steps[0]


def test_guard_is_the_first_step_and_reads_the_prefix_from_env():
    doc, _ = load()
    step = _guard_step(doc)
    assert doc["jobs"]["run-task"]["steps"][0] is step
    assert step["env"] == {"SUNBACK_PREFIX": "${{ inputs.prefix }}"}


@pytest.mark.parametrize("event,prefix,code", [
    ("schedule", "staging/", 1),
    ("schedule", "", 0),
    ("workflow_dispatch", "staging/", 0),
    ("workflow_dispatch", "", 0),
])
def test_guard_refuses_a_prefix_on_schedule(event, prefix, code):
    doc, _ = load()
    env = {"PATH": os.environ.get("PATH", ""), "GITHUB_EVENT_NAME": event, "SUNBACK_PREFIX": prefix}
    result = subprocess.run(["bash", "-c", _guard_step(doc)["run"]], env=env, capture_output=True, text=True)
    assert result.returncode == code, result.stdout + result.stderr

"""SB-prefix: GitCloudRunHourly.yml passes a `prefix` dispatch input to the reducer as SUNBACK_PREFIX.

Decision (Gilly, 2026-10-02): wire the input, add an offline audit, do not dispatch.
Before this, the workflow declared `workflow_dispatch:` with no inputs and never set
SUNBACK_PREFIX, so a staging run was impossible and a plain dispatch wrote production keys.

The workflow is parsed as YAML; nothing is dispatched and nothing reaches AWS or GitHub.
"""
import importlib
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
            if s is not step:
                assert "SUNBACK_PREFIX" not in (s.get("env") or {}), (job_name, s.get("name"))
    assert "SUNBACK_PREFIX" not in (doc.get("env") or {})


def test_push_and_schedule_triggers_are_unchanged():
    _, triggers = load()
    assert set(triggers) == {"push", "schedule", "workflow_dispatch"}
    assert triggers["push"] == {
        "branches": ["master"],
        "paths-ignore": ["web/**", "docs/**", "aws_lambda/**", "**.md"],
    }
    assert triggers["schedule"] == [{"cron": "0 * * * *"}]


def test_rest_of_the_workflow_is_unchanged():
    doc, _ = load()
    assert list(doc["jobs"]) == ["gate", "run-task"]
    assert [s.get("name") or s.get("uses") for s in doc["jobs"]["run-task"]["steps"]] == [
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
    """Push and schedule pass '' (inputs.prefix is empty there): production, byte for byte."""
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

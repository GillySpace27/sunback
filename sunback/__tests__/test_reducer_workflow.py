"""SB-2 Task 6: the production reducer runs on schedule and dispatch only, one at a time, with pinned actions.

Decision (Gilly, 2026-10-12): remove the push trigger, add a concurrency group that never
cancels an in-flight production run, and pin every action by full commit SHA.
The workflow is parsed as text and YAML; nothing is dispatched.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / ".github" / "workflows" / "GitCloudRunHourly.yml"


def load():
    doc = yaml.safe_load(WF.read_text(encoding="utf-8"))
    return doc, doc.get("on", doc.get(True))   # PyYAML reads the bare key `on` as boolean True


def test_no_push_trigger():
    _, triggers = load()
    assert set(triggers) == {"schedule", "workflow_dispatch"}


def test_one_reducer_at_a_time_and_never_cancel_a_running_one():
    doc, _ = load()
    assert doc["concurrency"] == {"group": "reducer", "cancel-in-progress": False}


def test_every_action_is_pinned_by_full_sha_with_a_version_comment():
    uses = [ln.strip() for ln in WF.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("uses:") or ln.strip().startswith("- uses:")]
    assert uses, "no uses: lines found"
    bad = [u for u in uses if not re.search(r"@[0-9a-f]{40} # v\d+\.\d+\.\d+$", u)]
    assert bad == []

"""Tests for devtools/scripts/nrt_status.py (suite SU-5). No network, no AWS."""
import datetime as dt

from devtools.scripts import nrt_status
from devtools.scripts.freshness_probe import Check, run_checks

NOW = dt.datetime(2026, 10, 1, 18, 0, 0, tzinfo=dt.timezone.utc)
NO_AWS = lambda: ("UNCHECKED", "no AWS credentials in this run")
ORRERY_KEYS = {"name", "title", "checked_at", "complete", "total", "next",
               "external_state", "external_label", "milestones"}
MILESTONE_KEYS = {"key", "label", "done", "note", "gated", "how_kind", "how"}


def test_incomplete_state_shows_less_than_total():
    checks = [Check("s3-image-times", "FAIL", "age 7200 s (limit 3600 s)"),
              Check("sun-html", "PASS", "HTTP 200, 31000 bytes"),
              Check("appcast-heliogram", "UNCHECKED", "HTTP 404; not live before Heliogram 0.8")]
    snap = nrt_status.snapshot(checks, NOW)
    assert (snap["complete"], snap["total"]) == (1, 3)
    assert snap["external_state"] == "FAIL"
    assert snap["next"] == nrt_status.LABELS["s3-image-times"]


def test_unchecked_is_neither_done_nor_fail():
    snap = nrt_status.snapshot([Check("appcast-heliogram", "UNCHECKED", "HTTP 404")], NOW)
    m = snap["milestones"][0]
    assert m["done"] is False and m["note"].startswith("UNCHECKED: ")
    assert snap["external_state"] == "UNCHECKED"


def test_snapshot_has_the_orrery_shape():
    snap = nrt_status.snapshot([Check("sun-html", "PASS", "HTTP 200, 1 bytes")], NOW)
    assert set(snap) == ORRERY_KEYS
    assert set(snap["milestones"][0]) == MILESTONE_KEYS
    assert snap["name"] == "sunback-nrt" and snap["checked_at"] == "2026-10-01T18:00:00+00:00"


def test_dark_chain_end_to_end_is_incomplete():
    checks = run_checks(NOW, lambda url, timeout: (500, {}, b""), env={}, budget=NO_AWS)
    snap = nrt_status.snapshot(checks, NOW)
    assert snap["total"] == len(checks) == 7
    assert snap["complete"] == 0
    assert all(c in nrt_status.LABELS for c in (m["key"] for m in snap["milestones"]))

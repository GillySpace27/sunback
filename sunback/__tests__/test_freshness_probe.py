"""Tests for devtools/scripts/freshness_probe.py (suite SU-2). No network."""
import datetime as dt
import json
import pathlib

from devtools.scripts.freshness_probe import (
    APPCAST_HELIOGRAM_URL, APPCAST_URL, R2_INDEX_URL, S3_IMAGE_TIMES_URL,
    STORE_BUILD_INFO_URL, SUN_HTML_URL, aws_budget, exit_code, run_checks,
)

NOW = dt.datetime(2026, 10, 1, 15, 0, 0, tzinfo=dt.timezone.utc)
NO_AWS = lambda: ("UNCHECKED", "no AWS credentials in this run")


def healthy():
    """URL -> (status, headers, body) for a chain where everything is fresh."""
    return {
        S3_IMAGE_TIMES_URL: (200, {}, b"2026-10-01T14:40:00.570\n"),
        R2_INDEX_URL: (200, {}, json.dumps({"generated": "2026-10-01T12:17:00Z", "products": []}).encode()),
        SUN_HTML_URL: (200, {}, b"<!doctype html><title>The Sun Right Now</title>"),
        APPCAST_URL: (200, {}, b"<rss><item><sparkle:version>7</sparkle:version></item></rss>"),
        APPCAST_HELIOGRAM_URL: (404, {}, b""),
        STORE_BUILD_INFO_URL: (200, {}, b'{"built": "2026-09-30T20:00:00Z"}'),
    }


def fetcher(table, seen=None):
    def fetch(url, timeout):
        if seen is not None:
            seen[url] = timeout
        return table[url]
    return fetch


def by_id(checks):
    return {c.id: c for c in checks}


def test_healthy_chain_passes_and_exits_zero():
    checks = run_checks(NOW, fetcher(healthy()), env={}, budget=NO_AWS)
    got = {c.id: c.status for c in checks}
    assert got == {
        "s3-image-times": "PASS", "r2-index": "PASS", "sun-html": "PASS",
        "appcast": "PASS", "appcast-heliogram": "UNCHECKED",
        "store-build-info": "PASS", "aws-budget": "UNCHECKED",
    }
    assert exit_code(checks) == 0


def test_stale_s3_fails_and_exits_one():
    table = healthy()
    table[S3_IMAGE_TIMES_URL] = (200, {}, b"2026-10-01T13:00:00.000")
    checks = run_checks(NOW, fetcher(table), env={}, budget=NO_AWS)
    assert by_id(checks)["s3-image-times"].status == "FAIL"
    assert exit_code(checks) == 1


def test_s3_threshold_override_turns_fresh_into_fail():
    checks = run_checks(NOW, fetcher(healthy()), env={"FRESHNESS_S3_MAX_AGE_S": "1"}, budget=NO_AWS)
    assert by_id(checks)["s3-image-times"].status == "FAIL"


def test_r2_404_is_unchecked_until_armed():
    table = healthy()
    table[R2_INDEX_URL] = (404, {}, b"")
    assert by_id(run_checks(NOW, fetcher(table), env={}, budget=NO_AWS))["r2-index"].status == "UNCHECKED"
    armed = run_checks(NOW, fetcher(table), env={"FRESHNESS_ARM_R2_INDEX": "1"}, budget=NO_AWS)
    assert by_id(armed)["r2-index"].status == "FAIL"


def test_r2_stale_generated_fails():
    table = healthy()
    table[R2_INDEX_URL] = (200, {}, b'{"generated": "2026-09-30T23:00:00Z"}')
    assert by_id(run_checks(NOW, fetcher(table), env={}, budget=NO_AWS))["r2-index"].status == "FAIL"


def test_heliogram_appcast_passes_once_live():
    table = healthy()
    table[APPCAST_HELIOGRAM_URL] = (200, {}, b"<sparkle:version>8</sparkle:version>")
    assert by_id(run_checks(NOW, fetcher(table), env={}, budget=NO_AWS))["appcast-heliogram"].status == "PASS"


def test_appcast_without_sparkle_version_fails():
    table = healthy()
    table[APPCAST_URL] = (200, {}, b"<html>404</html>")
    assert by_id(run_checks(NOW, fetcher(table), env={}, budget=NO_AWS))["appcast"].status == "FAIL"


def test_empty_sun_html_and_dead_store_fail():
    table = healthy()
    table[SUN_HTML_URL] = (200, {}, b"")
    table[STORE_BUILD_INFO_URL] = (0, {}, b"timed out")
    got = by_id(run_checks(NOW, fetcher(table), env={}, budget=NO_AWS))
    assert got["sun-html"].status == "FAIL"
    assert got["store-build-info"].status == "FAIL"


def test_store_gets_a_30_second_timeout():
    seen = {}
    run_checks(NOW, fetcher(healthy(), seen), env={}, budget=NO_AWS)
    assert seen[STORE_BUILD_INFO_URL] == 30


def test_unchecked_never_fails_the_run():
    table = healthy()
    table[R2_INDEX_URL] = (404, {}, b"")
    assert exit_code(run_checks(NOW, fetcher(table), env={}, budget=NO_AWS)) == 0


def test_budget_without_credentials_is_unchecked_and_runs_nothing():
    def run(*a, **k):
        raise AssertionError("aws CLI must not run without credentials")
    assert aws_budget(env={}, run=run) == ("UNCHECKED", "no AWS credentials in this run")


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_freshness_workflow_is_scheduled_and_separate():
    wf = (ROOT / ".github" / "workflows" / "freshness.yml").read_text()
    assert "cron: '41 */2 * * *'" in wf
    assert "workflow_dispatch:" in wf
    assert "run: python3 -m devtools.scripts.freshness_probe" in wf
    assert "contents: read" in wf
    reducer = (ROOT / ".github" / "workflows" / "GitCloudRunHourly.yml").read_text()
    # The reducer gate already calls SB-8's check_freshness.py; it must never run this probe.
    assert "freshness_probe" not in reducer

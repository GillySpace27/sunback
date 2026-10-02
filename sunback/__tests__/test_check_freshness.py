"""SB-8: devtools/scripts/check_freshness.py against a file:// copy of the bucket (no network)."""
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_freshness", ROOT / "devtools" / "scripts" / "check_freshness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bucket(tmp_path):
    """image_times.txt last modified 10 min before NOW; 171 updated 20 min before; 193 3 h before."""
    (tmp_path / "manifest").mkdir()
    times = tmp_path / "image_times.txt"
    times.write_text("2026-09-28T11:50:00.000")
    stamp = NOW.timestamp() - 600
    os.utime(times, (stamp, stamp))
    (tmp_path / "manifest" / "171.json").write_text(json.dumps({"id": "171", "updated": "2026-09-28T11:40:00Z"}))
    (tmp_path / "manifest" / "193.json").write_text(json.dumps({"id": "193", "updated": "2026-09-28T09:00:00Z"}))
    return tmp_path.as_uri() + "/"


def test_fresh_keys_pass(bucket, capsys):
    cf = _load()
    code = cf.run(["--base-url", bucket, "--key", "image_times.txt", "--key", "manifest/171.json", "--json"], now=NOW)
    doc = json.loads(capsys.readouterr().out)
    assert code == 0, doc
    assert [(r["key"], r["age_s"], r["status"]) for r in doc["results"]] == [
        ("image_times.txt", 600, "fresh"), ("manifest/171.json", 1200, "fresh")]
    assert doc["worst_age_s"] == 1200


def test_threshold_one_second_fails(bucket, capsys):
    cf = _load()
    code = cf.run(["--base-url", bucket, "--key", "manifest/171.json", "--threshold", "1", "--json"], now=NOW)
    assert code == 1
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "stale"


def test_stale_fragment_fails(bucket, capsys):
    cf = _load()
    code = cf.run(["--base-url", bucket, "--key", "manifest/193.json"], now=NOW)
    out = capsys.readouterr().out
    assert code == 1
    assert "STALE" in out and "FAIL: stale or missing" in out


def test_missing_key_fails(bucket, capsys):
    cf = _load()
    assert cf.run(["--base-url", bucket, "--key", "manifest/304.json", "--json"], now=NOW) == 1
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "missing"


def test_prefix_is_prepended(bucket, tmp_path, capsys):
    cf = _load()
    (tmp_path / "staging" / "manifest").mkdir(parents=True)
    (tmp_path / "staging" / "manifest" / "171.json").write_text(json.dumps({"updated": "2026-09-28T11:55:00Z"}))
    code = cf.run(["--base-url", bucket, "--prefix", "staging/", "--key", "manifest/171.json", "--json"], now=NOW)
    doc = json.loads(capsys.readouterr().out)
    assert code == 0 and doc["results"][0] == {"key": "staging/manifest/171.json", "age_s": 300, "status": "fresh"}


def test_unreachable_is_unchecked(capsys):
    cf = _load()
    code = cf.run(["--base-url", "http://127.0.0.1:9/", "--key", "image_times.txt", "--json"], now=NOW)
    assert code == 3
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "unreachable"


def test_default_keys_cover_every_product():
    cf = _load()
    from aws_lambda.video_builder.manifest import PRODUCTS
    assert cf.default_keys() == ["image_times.txt"] + [f"manifest/{p['id']}.json" for p in PRODUCTS]

"""The contract fixture tool refuses bad data and detects edits; the captured contract-v1 fixture is valid (SU-9)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aws_lambda.video_builder import manifest
from aws_lambda.video_builder.fixtures import capture_contract as cc
from aws_lambda.video_builder.schema.validate import validate

BASE = cc.DEFAULT_BASE_URL
V1 = Path(__file__).resolve().parents[2] / "aws_lambda" / "video_builder" / "fixtures" / "contract-v1"
NOW = datetime(2026, 10, 1, 15, 0, 0, tzinfo=timezone.utc)
V1_IDS = ["rainbow", "171", "193", "211", "304", "335", "94", "131", "1600", "1700", "composite_uv", "dem"]


def _frag(pid):
    return manifest.build_manifest_fragment(pid, updated="2026-10-01T15:00:00Z", frame_count=144,
                                            integration={"frames": 5, "method": "median"})


def fake_bucket(index=True, mutate=None):
    frags = [_frag(p["id"]) for p in manifest.PRODUCTS]
    docs = {f"manifest/{f['id']}.json": json.dumps(f).encode() for f in frags}
    if index:
        docs["manifest/index.json"] = json.dumps(manifest.build_index(frags, "2026-10-01T15:00:00Z")).encode()
    docs["image_times.txt"] = b"2026-10-01T15:03:02.081\n"
    if mutate:
        mutate(docs)

    def fetch(url):
        key = url[len(BASE):]
        return (200, docs[key]) if key in docs else (404, b"")
    return fetch


def test_capture_writes_a_verifiable_fixture(tmp_path, capsys):
    out = tmp_path / "contract-v9"
    assert cc.capture(fake_bucket(), out, version=9) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(cc.FILES + ("SHA256SUMS",))
    assert (out / "README.md").read_text().startswith("# contract-v9")
    assert cc.verify(out) == 0
    assert "OK" in capsys.readouterr().out


def test_capture_refuses_an_existing_destination(tmp_path, capsys):
    out = tmp_path / "contract-v1"
    out.mkdir()
    assert cc.capture(fake_bucket(), out, version=1) == 2
    assert "kept forever" in capsys.readouterr().out
    assert list(out.iterdir()) == []


def test_missing_index_needs_the_flag(tmp_path, capsys):
    out = tmp_path / "v"
    assert cc.capture(fake_bucket(index=False), out, version=1) == 1
    assert "--synthesize-index" in capsys.readouterr().out
    assert not out.exists()
    assert cc.capture(fake_bucket(index=False), out, version=1, synthesize=True) == 0
    assert "synthesized" in (out / "README.md").read_text()
    assert cc.verify(out) == 0


def test_a_fragment_that_breaks_the_schema_is_not_written(tmp_path, capsys):
    def rename(docs):
        frag = json.loads(docs["manifest/171.json"])
        frag["img_1k"] = frag.pop("img1k")
        docs["manifest/171.json"] = json.dumps(frag).encode()
    out = tmp_path / "v"
    assert cc.capture(fake_bucket(mutate=rename), out, version=1) == 1
    text = capsys.readouterr().out
    assert "fragment-171.json: $: missing required key 'img1k'" in text
    assert "NOT WRITTEN" in text
    assert not out.exists()


def test_a_bad_capture_time_is_not_written(tmp_path, capsys):
    out = tmp_path / "v"
    assert cc.capture(fake_bucket(mutate=lambda d: d.update({"image_times.txt": b"yesterday\n"})), out, version=1) == 1
    assert "image_times.txt" in capsys.readouterr().out
    assert not out.exists()


def test_verify_and_same_detect_an_edit(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    cc.capture(fake_bucket(), a, version=1, now=NOW)
    cc.capture(fake_bucket(), b, version=1, now=NOW)
    assert cc.same(a, b) == 0
    (b / "image_times.txt").write_bytes(b"2026-10-01T15:03:03.000\n")
    assert cc.verify(b) == 1
    assert cc.same(a, b) == 1
    out = capsys.readouterr().out
    assert "MISMATCH image_times.txt" in out
    assert "DIFFERENT image_times.txt" in out


def test_contract_v1_fixture_is_valid_and_pinned():
    assert V1.is_dir(), "contract-v1 has not been captured yet (Task 3 Step 3)"
    assert cc.verify(V1) == 0
    index = json.loads((V1 / "index.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in index["products"]] == V1_IDS
    ids_now = [p["id"] for p in manifest.PRODUCTS]
    assert set(V1_IDS) <= set(ids_now), "a v1 product id was removed from PRODUCTS: that needs contract-v2"
    for name, pid in (("fragment-171.json", "171"), ("fragment-rainbow.json", "rainbow")):
        frag = json.loads((V1 / name).read_text(encoding="utf-8"))
        assert frag["id"] == pid
        assert validate(frag, json.loads((cc.SCHEMA_DIR / "manifest-fragment.schema.json").read_text())) == []
    sums = (V1 / "SHA256SUMS").read_text().splitlines()
    assert [ln.split("  ", 1)[1] for ln in sums] == sorted(cc.FILES)
    for line in sums:
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((V1 / name).read_bytes()).hexdigest() == digest


def test_offline_capture_is_valid_and_says_it_is_not_live(tmp_path):
    out = tmp_path / "contract-v7"
    assert cc.capture(cc.offline_fetch(), out, version=7, base_url=cc.OFFLINE_BASE_URL) == 0
    assert cc.verify(out) == 0
    readme = (out / "README.md").read_text()
    assert "NOT captured from the live bucket" in readme
    assert "the-sun-now" not in readme
    index = json.loads((out / "index.json").read_text())
    assert [p["id"] for p in index["products"]] == [p["id"] for p in manifest.PRODUCTS]
    assert all("video_v" in p and "obs_end" in p for p in index["products"])


def test_urllib_fetch_reads_a_local_http_server(tmp_path):
    import functools
    import http.server
    import threading
    served = tmp_path / "www"
    (served / "manifest").mkdir(parents=True)
    fetch_off = cc.offline_fetch()
    for name in ["image_times.txt", "manifest/index.json"] + [f"manifest/{p['id']}.json" for p in manifest.PRODUCTS]:
        (served / name).write_bytes(fetch_off(cc.OFFLINE_BASE_URL + name)[1])
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        assert cc.urllib_fetch(base + "nothing.json")[0] == 404
        out = tmp_path / "contract-v8"
        assert cc.capture(cc.urllib_fetch, out, version=8, base_url=base) == 0
    finally:
        server.shutdown()
        server.server_close()
    assert cc.verify(out) == 0
    assert "NOT captured from the live bucket" in (out / "README.md").read_text()

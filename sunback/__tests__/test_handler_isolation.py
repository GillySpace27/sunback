"""SB-8: one bad record must not stop the others; status.json; staging keys stay under staging/."""
import pytest

from sunback.__tests__.lambda_harness import FakeS3, fake_ffmpeg, load_handler

STILL = "1k/rhef_171_1k.png"


def _record(key, event_time="2026-09-28T12:00:05.000Z"):
    return {"eventTime": event_time, "s3": {"object": {"key": key}}}


def _seed_still(fake, key=STILL, obstime="2026-09-28T12:00:00Z"):
    fake.objects[key] = {"Body": b"png", "Metadata": {"obstime": obstime},
                         "LastModified": None}


@pytest.fixture
def lam(monkeypatch, tmp_path):
    fake = FakeS3()
    handler = load_handler(monkeypatch, fake, fake_ffmpeg(tmp_path))
    return handler, fake


def test_bad_record_does_not_stop_good_one(lam):
    handler, fake = lam
    _seed_still(fake)
    event = {"Records": [_record("1k/rhef_193_1k.png"),  # no such object: head_object fails
                         _record(STILL)]}
    with pytest.raises(RuntimeError) as exc:
        handler.handler(event, None)
    assert str(exc.value) == "1 record(s) failed: ['1k/rhef_193_1k.png']"
    assert "manifest/171.json" in fake.objects  # the good product still published
    assert fake.json("manifest/171.json")["updated"] == "2026-09-28T12:00:00Z"


def test_all_good_returns_processed(lam):
    handler, fake = lam
    _seed_still(fake)
    assert handler.handler({"Records": [_record(STILL)]}, None) == {"processed": ["171"]}


def test_status_json_written_beside_index(lam):
    handler, fake = lam
    _seed_still(fake)
    handler.handler({"Records": [_record(STILL)]}, None)
    status = fake.json("status.json")
    assert [p["id"] for p in status["products"]] == ["171"]
    assert status["products"][0]["updated"] == "2026-09-28T12:00:00Z"
    assert status["worst_age_s"] == status["products"][0]["age_s"]
    assert fake.objects["status.json"]["CacheControl"] == "public, max-age=60"
    assert fake.objects["status.json"]["ACL"] == "public-read"


def test_staging_key_stays_under_staging(lam):
    handler, fake = lam
    _seed_still(fake, key="staging/" + STILL)
    assert handler.handler({"Records": [_record("staging/" + STILL)]}, None) == {"processed": ["171"]}
    written = [k for op, k in fake.calls if op in ("put_object", "copy_object", "upload_file")]
    assert written and all(k.startswith("staging/") for k in written), written
    assert not [k for op, k in fake.calls if op == "delete_object"]
    frag = fake.json("staging/manifest/171.json")
    assert frag["still_v"] == "v/171/20260928T120000.png"  # keys inside a fragment stay root-relative


def test_production_keys_unchanged(lam):
    handler, fake = lam
    _seed_still(fake)
    handler.handler({"Records": [_record(STILL)]}, None)
    assert sorted(fake.objects) == [
        "1k/rhef_171_1k.png",
        "frames/171/20260928T120000_1k.png",
        "manifest/171.json",
        "manifest/index.json",
        "status.json",
        "v/171/20260928T120000.mp4",
        "v/171/20260928T120000.png",
        "video/rhef_171_1k.mp4",
    ]

"""The video Lambda end to end, offline (SB-11).

An S3 ObjectCreated event for a 1k still runs the real handler against an
in-memory S3 and a stand-in ffmpeg (sunback/__tests__/lambda_harness.py, created
by SB-8 and reused here). The tests pin what production writes today: the
frame-queue copy, the immutable v/ copies, the video keys, manifest/<id>.json
and manifest/index.json.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from aws_lambda.video_builder.manifest import IMMUTABLE
from sunback.__tests__.lambda_harness import FakeS3, fake_ffmpeg, load_handler

STILL = "1k/rhef_171_1k.png"
OBSTIME = "2026-09-28T12:00:00Z"
STAMP = "20260928T120000"
PNG_BYTES = b"\x89PNG\r\n\x1a\nsb11"


def s3_event(key, event_time="2026-09-28T12:03:10.123Z"):
    return {"Records": [{"eventName": "ObjectCreated:Put", "eventTime": event_time,
                         "s3": {"bucket": {"name": "the-sun-now"}, "object": {"key": key}}}]}


@pytest.fixture
def lam(monkeypatch, tmp_path):
    fake = FakeS3()
    fake.put_object(Bucket="the-sun-now", Key=STILL, Body=PNG_BYTES, Metadata={"obstime": OBSTIME},
                    ContentType="image/png", ACL="public-read")
    fake.calls.clear()
    ffmpeg = fake_ffmpeg(tmp_path)
    handler = load_handler(monkeypatch, fake, ffmpeg,
                           env={"INTEGRATION_FRAMES": "5", "INTEGRATION_METHOD": "median"})
    return handler, fake, tmp_path


def test_new_still_builds_queue_video_and_manifests(lam):
    handler, fake, tmp_path = lam
    assert handler.handler(s3_event(STILL), None) == {"processed": ["171"]}

    frame = fake.objects[f"frames/171/{STAMP}_1k.png"]
    assert frame["Body"] == PNG_BYTES
    assert frame["Metadata"] == {"obstime": OBSTIME}

    still_v = fake.objects[f"v/171/{STAMP}.png"]
    assert (still_v["ContentType"], still_v["CacheControl"], still_v["ACL"]) == ("image/png", IMMUTABLE, "public-read")
    assert still_v["Metadata"] == {}

    video_v = fake.objects[f"v/171/{STAMP}.mp4"]
    assert video_v["Body"] == b"fake mp4"
    assert (video_v["ContentType"], video_v["CacheControl"], video_v["ACL"]) == ("video/mp4", IMMUTABLE, "public-read")
    video = fake.objects["video/rhef_171_1k.mp4"]
    assert video["Metadata"] == {"through": STAMP}
    assert (video["ContentType"], video["ContentDisposition"]) == ("video/mp4", "inline")

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0].endswith("fake_ffmpeg") and argv[-1].endswith("171.mp4")
    assert argv[argv.index("-r") + 1] == "18"

    fragment = fake.json("manifest/171.json")
    assert fragment["id"] == "171"
    assert fragment["label"] == "AIA 171 Å"
    assert fragment["thumb"] == "thumb/rhef_171_thumb.png"
    assert fragment["img1k"] == STILL
    assert fragment["video"] == "video/rhef_171_1k.mp4"
    assert fragment["updated"] == OBSTIME
    assert fragment["frame_count"] == 1
    assert fragment["integration"] == {"frames": 5, "method": "median"}
    assert fragment["video_v"] == f"v/171/{STAMP}.mp4"
    assert fragment["still_v"] == f"v/171/{STAMP}.png"
    assert fragment["through"] == OBSTIME
    assert fake.objects["manifest/171.json"]["CacheControl"] == "no-cache"
    index = fake.json("manifest/index.json")
    assert [p["id"] for p in index["products"]] == ["171"]
    assert index["products"][0] == fragment


def test_recent_video_is_not_re_encoded(lam):
    handler, fake, tmp_path = lam
    fake.put_object(Bucket="the-sun-now", Key="video/rhef_171_1k.mp4", Body=b"old mp4",
                    Metadata={"through": "20260928T100000"})
    handler.handler(s3_event(STILL), None)
    assert not (tmp_path / "argv.json").exists()
    assert fake.objects["video/rhef_171_1k.mp4"]["Body"] == b"old mp4"
    frag = fake.json("manifest/171.json")
    assert (frag["video_v"], frag["through"]) == ("v/171/20260928T100000.mp4", "2026-09-28T10:00:00Z")
    assert f"v/171/{STAMP}.png" in fake.objects


def test_frames_older_than_the_window_are_pruned(lam):
    handler, fake, _ = lam
    old = (datetime(2026, 9, 28, 12, tzinfo=timezone.utc) - timedelta(hours=50)).strftime("%Y%m%dT%H%M%S")
    kept = (datetime(2026, 9, 28, 12, tzinfo=timezone.utc) - timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
    for stamp in (old, kept):
        fake.put_object(Bucket="the-sun-now", Key=f"frames/171/{stamp}_1k.png", Body=PNG_BYTES)
    handler.handler(s3_event(STILL), None)
    queue = sorted(k for k in fake.objects if k.startswith("frames/171/"))
    assert queue == [f"frames/171/{kept}_1k.png", f"frames/171/{STAMP}_1k.png"]
    assert ("delete_object", f"frames/171/{old}_1k.png") in fake.calls


def test_event_for_another_prefix_is_ignored(lam):
    handler, fake, _ = lam
    assert handler.handler(s3_event("thumb/rhef_171_thumb.png"), None) == {"processed": []}
    assert fake.calls == []


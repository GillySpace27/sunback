"""SB-5 production-safety rails: today's keys by default, staging/ on request,
no bucket wipe, no unflagged sunback-serve.

Every S3 call goes to a stand-in; nothing here reaches AWS.
"""
import importlib
import os
import sys
from unittest.mock import MagicMock

import boto3
import numpy as np
import pytest

aws = importlib.import_module("sunback.putter.AwsPutter")


class FakeS3Client:
    """Records upload_file; any other S3 call fails the test."""

    def __init__(self):
        self.uploads = []

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, **kwargs):
        assert os.path.exists(Filename), Filename
        self.uploads.append((Bucket, Key, dict(ExtraArgs or {})))

    def __getattr__(self, name):
        raise AssertionError(f"unexpected S3 call: {name}")


class StandInBucket:
    """Takes the place of the pre-SB-5 module-level bucket; records a wipe instead of doing it."""

    def __init__(self):
        self.wiped = 0

    @property
    def objects(self):
        return self

    def all(self):
        return self

    def delete(self):
        self.wiped += 1
        return []


def test_empty_the_bucket_always_raises(monkeypatch):
    stand_in = StandInBucket()
    monkeypatch.setattr(aws, "bucket", stand_in, raising=False)
    monkeypatch.setattr(aws, "_S3_CLIENT", FakeS3Client(), raising=False)
    putter = aws.AwsPutter.__new__(aws.AwsPutter)
    with pytest.raises(RuntimeError, match="empty_the_bucket is fenced"):
        putter.empty_the_bucket()
    assert stand_in.wiped == 0


# Recorded from master before SB-5 (Task 0 of the SB-5 task file): bucket, key,
# ContentType, and whether obstime metadata rides along. Order is upload order.
TODAYS_KEYS = [
    ("the-sun-now", "1k/rhef_171_1k.png", "image/png", True),
    ("the-sun-now", "thumb/rhef_171_thumb.png", "image/png", False),
    ("the-sun-now", "meta/rhef_171.json", "application/json", False),  # SB-9 sidecar
    ("the-sun-now", "1k/rhef_rainbow_1k.png", "image/png", True),
    ("the-sun-now", "thumb/rhef_rainbow_thumb.png", "image/png", False),
    ("the-sun-now", "meta/rhef_rainbow.json", "application/json", False),  # SB-9 sidecar
    ("the-sun-now", "1k/rhef_composite_uv_1k.png", "image/png", True),
    ("the-sun-now", "thumb/rhef_composite_uv_thumb.png", "image/png", False),
    ("the-sun-now", "meta/rhef_composite_uv.json", "application/json", False),  # SB-9 sidecar
    ("the-sun-now", "1k/rhef_dem_1k.png", "image/png", True),
    ("the-sun-now", "thumb/rhef_dem_thumb.png", "image/png", False),
    ("the-sun-now", "meta/rhef_dem.json", "application/json", False),  # SB-9 sidecar
    ("the-sun-now", "video/rhef_tscan.mp4", "video/mp4", False),
    ("the-sun-now", "image_times.txt", "text/plain", False),
    ("the-sun-now", "image_times_readable.txt", "text/plain", False),
]

LOCAL_PNGS = [
    "DrGilly_0171_ups(rhef).png",
    "BGR_0171_0193_0211_ups(rhef).png",
    "BGR_1700_1600_0304_ups(rhef).png",
    "C_isothermal.png",
    "DrGilly_1234_ups(rhef).png",  # not a served channel: never uploaded
]


@pytest.fixture
def fake_s3(monkeypatch):
    fake = FakeS3Client()
    monkeypatch.setattr(aws, "_S3_CLIENT", fake)
    for name in ("SUNBACK_BUCKET", "SUNBACK_PREFIX", "SUNBACK_WRITE_READABLE_TIMES"):
        monkeypatch.delenv(name, raising=False)
    return fake


def _run_put(tmp_path):
    import cv2

    pngs = []
    for name in LOCAL_PNGS:
        path = tmp_path / name
        cv2.imwrite(str(path), np.zeros((16, 16), np.uint8))
        pngs.append(str(path))
    (tmp_path / "dem").mkdir()
    (tmp_path / "dem" / "a_temp_video_small.mp4").write_bytes(b"mp4")

    params = MagicMock()
    params.multi_pool = None
    params.local_imgs_paths.return_value = pngs
    params.imgs_top_directory.return_value = str(tmp_path)
    params.base_directory.return_value = str(tmp_path)
    params.time_path.return_value = str(tmp_path / "image_times.txt")
    params.local_fits_paths.return_value = [str(tmp_path / "AIAsynoptic0171.fits")]
    params.fits_directory.return_value = str(tmp_path / "no-fits")  # SB-9: no FITS, so upload-time fallback

    putter = aws.AwsPutter.__new__(aws.AwsPutter)  # skip Processor.__init__ (no FITS needed)
    putter.params = params
    putter.ii = 0
    putter.pbar = None
    putter.to_upload = None
    putter.load_this_fits_frame = lambda fits_path=None, in_name=None, quiet=False: (
        None, "0171", "2026-09-28T12:00:00.00", None, None, None)
    putter.clean_time_string = lambda t, zone=None, out_fmt=None: f"{zone}|{t}"
    putter.put()


def _summary(uploads):
    return [(b, k, x.get("ContentType"), "obstime" in x.get("Metadata", {})) for b, k, x in uploads]


def test_default_run_writes_todays_keys(fake_s3, tmp_path):
    _run_put(tmp_path)
    assert _summary(fake_s3.uploads) == TODAYS_KEYS
    for _, _, extra in fake_s3.uploads:
        assert extra["ACL"] == "public-read"
        assert extra["ContentDisposition"] == "inline"


def test_staging_prefix_prefixes_every_key(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SUNBACK_PREFIX", "staging/")
    _run_put(tmp_path)
    assert _summary(fake_s3.uploads) == [(b, "staging/" + k, t, m) for b, k, t, m in TODAYS_KEYS]


def test_bucket_comes_from_settings(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SUNBACK_BUCKET", "sunback-test-bucket")
    _run_put(tmp_path)
    assert {b for b, _, _ in fake_s3.uploads} == {"sunback-test-bucket"}


def test_readable_times_can_be_switched_off(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setenv("SUNBACK_WRITE_READABLE_TIMES", "0")
    _run_put(tmp_path)
    assert [k for _, k, _, _ in _summary(fake_s3.uploads)] == [k for _, k, _, _ in TODAYS_KEYS][:-1]


def test_upload_public_returns_full_key(fake_s3, tmp_path):
    from sunback.settings import NrtSettings

    path = tmp_path / "x.png"
    path.write_bytes(b"x")
    key = aws.upload_public(str(path), "overlay/grid_1k.png", "image/png",
                            cache_control="max-age=60", settings=NrtSettings(prefix="staging/"))
    assert key == "staging/overlay/grid_1k.png"
    assert fake_s3.uploads == [("the-sun-now", "staging/overlay/grid_1k.png", {
        "ACL": "public-read", "ContentDisposition": "inline",
        "ContentType": "image/png", "CacheControl": "max-age=60"})]


def test_no_boto3_call_at_import(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("boto3 called at import time")

    monkeypatch.setattr(boto3, "resource", refuse)
    monkeypatch.setattr(boto3, "client", refuse)
    importlib.reload(aws)
    assert aws._S3_CLIENT is None


def _lingon_with_fakes(monkeypatch, argv):
    lingon = importlib.import_module("sunback.run.run_server_lingon")
    started = []

    class RecordingRunner:
        def __init__(self, params):
            self.params = params

        def start(self):
            started.append(self.params)

    monkeypatch.setattr(lingon, "Parameters", MagicMock)
    monkeypatch.setattr(lingon, "SingleRunner", RecordingRunner)
    monkeypatch.setattr(sys, "argv", argv)
    return lingon, started


def test_sunback_serve_refuses_without_flag(monkeypatch, capsys):
    lingon, started = _lingon_with_fakes(monkeypatch, ["sunback-serve"])
    with pytest.raises(SystemExit) as exc:
        lingon.run_server_lingon()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "sunback-serve is deprecated" in err
    assert started == []


def test_sunback_serve_runs_with_flag(monkeypatch):
    lingon, started = _lingon_with_fakes(monkeypatch, ["sunback-serve", "--force-production"])
    lingon.run_server_lingon()
    assert len(started) == 1

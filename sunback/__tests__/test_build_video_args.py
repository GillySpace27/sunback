"""Pin the ffmpeg argv of the video Lambda's _build_video (SB-1).

The bt709 tags fix the Heliograph oversaturation of 2026-09-12: without them a
player reads the untagged 1024x1024 H.264 stream with its own default matrix.
The test fails on any handler.py whose argv lacks the tags (master at a429bc1,
claude/versioned-keys at d031fd1).
"""
# handler.py creates its boto3 S3 client at import, so boto3 must be importable
# (it is a sunback dependency and is in the reducer image).
from aws_lambda.video_builder import handler

BT709_TOKENS = [
    "-vf",
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
    "-color_range", "tv",
    "-colorspace", "bt709",
]


class _FakeS3:
    def download_file(self, bucket, key, local):
        with open(local, "wb") as fp:
            fp.write(b"png")


def _argv_of_build_video(monkeypatch, tmp_path):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))

    monkeypatch.setattr(handler, "s3", _FakeS3())
    monkeypatch.setattr(handler.subprocess, "run", fake_run)
    keys = ["frames/171/20260928T120000_1k.png", "frames/171/20260928T122000_1k.png"]
    out, n = handler._build_video("171", keys, str(tmp_path))
    assert n == 2
    assert out == str(tmp_path / "171.mp4")
    assert len(calls) == 1, calls
    return calls[0]


def test_build_video_tags_bt709(monkeypatch, tmp_path):
    argv = _argv_of_build_video(monkeypatch, tmp_path)
    missing = [t for t in BT709_TOKENS if t not in argv]
    assert not missing, f"bt709 tokens missing from _build_video argv: {missing}"
    vf = argv.index("-vf")
    assert argv[vf + 1] == BT709_TOKENS[1]
    assert argv[argv.index("-color_range") + 1] == "tv"
    assert argv[argv.index("-colorspace") + 1] == "bt709"


def test_build_video_keeps_existing_args(monkeypatch, tmp_path):
    argv = _argv_of_build_video(monkeypatch, tmp_path)
    for flag, value in (("-c:v", "libx264"), ("-pix_fmt", "yuv420p"),
                        ("-f", "concat")):
        assert argv[argv.index(flag) + 1] == value, (flag, argv)
    assert argv[-1] == str(tmp_path / "171.mp4")
    # SB-9: faststart is kept; use_metadata_tags is added so the custom provenance tag is written
    assert argv[argv.index("-movflags") + 1].startswith("+faststart")


# --- SB-9: provenance in every MP4 and in the fragment ---------------------------
import json as _json

from sunback.__tests__.lambda_harness import FakeS3, fake_ffmpeg, load_handler


def _metadata_of(argv):
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == "-metadata"]


def test_build_video_writes_json_tag_and_creation_time(monkeypatch, tmp_path):
    """Decision A4: the JSON rides in its own tag; `comment` is reserved for the RH-3 stamp string."""
    argv = _argv_of_build_video(monkeypatch, tmp_path)
    metas = _metadata_of(argv)
    assert [m.split("=", 1)[0] for m in metas] == ["sunback_provenance", "creation_time"], argv
    info = _json.loads(metas[0].split("=", 1)[1])
    assert info["product"] == "171"
    assert (info["first"], info["through"]) == ("2026-09-28T12:00:00Z", "2026-09-28T12:20:00Z")
    assert info["frames"] == 2 and info["fps"] == handler.FPS
    assert metas[1] == "creation_time=2026-09-28T12:20:00Z"
    assert argv[argv.index("-movflags") + 1] == "+faststart+use_metadata_tags"
    assert argv[-1] == str(tmp_path / "171.mp4")  # output path stays last


def test_build_video_comment_is_the_stamp_string(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(handler, "s3", _FakeS3())
    monkeypatch.setattr(handler.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    keys = ["frames/171/20260928T120000_1k.png", "frames/171/20260928T122000_1k.png"]
    stamp = "RHEF sunkit-0.7 via sunkit-image 0.6.1; upsilon=0.35,0.35; deviations=none"
    handler._build_video("171", keys, str(tmp_path), {"frames": 5, "method": "median"}, stamp)
    metas = _metadata_of(calls[0])
    assert metas[0] == "comment=" + stamp
    assert [m.split("=", 1)[0] for m in metas] == ["comment", "sunback_provenance", "creation_time"]
    assert "comment=" not in metas[1]  # the JSON never lands in comment


def test_build_video_drops_a_stamp_that_is_not_one_ascii_line(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(handler, "s3", _FakeS3())
    monkeypatch.setattr(handler.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    keys = ["frames/171/20260928T120000_1k.png"]
    handler._build_video("171", keys, str(tmp_path), None, "line one\nline two")
    assert [m.split("=", 1)[0] for m in _metadata_of(calls[0])] == ["sunback_provenance", "creation_time"]


def test_fragment_carries_obs_window_from_still_metadata(monkeypatch, tmp_path):
    fake = FakeS3()
    lam = load_handler(monkeypatch, fake, fake_ffmpeg(tmp_path))
    fake.objects["1k/rhef_171_1k.png"] = {
        "Body": b"png", "LastModified": None,
        "Metadata": {"obstime": "2026-09-28T12:00:00Z", "obstime_source": "header",
                     "obs_start": "2026-09-28T11:52:00Z", "obs_end": "2026-09-28T12:00:00Z",
                     "tint_n": "5", "tint_m": "median",
                     "rhef_stamp": "RHEF oRHEF-2.0 via orhef 0.1.0.dev0; upsilon=0.35,0.35; deviations=none"}}
    lam.handler({"Records": [{"eventTime": "2026-09-28T12:31:00Z",
                              "s3": {"object": {"key": "1k/rhef_171_1k.png"}}}]}, None)
    frag = fake.json("manifest/171.json")
    assert (frag["obs_start"], frag["obs_end"]) == ("2026-09-28T11:52:00Z", "2026-09-28T12:00:00Z")
    recorded = _json.load(open(tmp_path / "argv.json"))
    assert "creation_time=2026-09-28T12:00:00Z" in recorded
    assert "comment=RHEF oRHEF-2.0 via orhef 0.1.0.dev0; upsilon=0.35,0.35; deviations=none" in recorded


def test_fragment_without_window_metadata_is_unchanged(monkeypatch, tmp_path):
    fake = FakeS3()
    lam = load_handler(monkeypatch, fake, fake_ffmpeg(tmp_path))
    fake.objects["1k/rhef_171_1k.png"] = {"Body": b"png", "LastModified": None,
                                          "Metadata": {"obstime": "2026-09-28T12:00:00Z"}}
    lam.handler({"Records": [{"eventTime": "", "s3": {"object": {"key": "1k/rhef_171_1k.png"}}}]}, None)
    frag = fake.json("manifest/171.json")
    assert "obs_start" not in frag and "obs_end" not in frag

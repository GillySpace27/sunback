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
                        ("-movflags", "+faststart"), ("-f", "concat")):
        assert argv[argv.index(flag) + 1] == value, (flag, argv)
    assert argv[-1] == str(tmp_path / "171.mp4")

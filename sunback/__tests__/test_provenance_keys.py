"""SU-18: additive provenance keys (heliosoftware/spec/provenance.md, version 1).

The fragment gains ``credit`` (always written by the Lambda) and ``rhef`` (the
RH-3 stamp line, only when the reducer sent one); the movies gain the MP4
``description`` tag holding the credit line. Every older key is unchanged.
"""
import ast
import json as _json
from pathlib import Path

from aws_lambda.video_builder import handler, manifest
from sunback.__tests__.lambda_harness import FakeS3, fake_ffmpeg, load_handler
from sunback.__tests__.test_build_video_args import _FakeS3, _metadata_of

ROOT = Path(__file__).resolve().parents[2]
STAMP = "RHEF oRHEF-2.0 via orhef 0.1.0.dev0; upsilon=0.35,0.35; deviations=none"
# heliosoftware/spec/attribution.json "credit" (GillySpace27.github.io PR #43), copied by hand
SPEC_CREDIT = "Imagery courtesy of NASA/SDO and the AIA science team."


def test_credit_is_the_spec_line_and_the_putter_line():
    tree = ast.parse((ROOT / "sunback" / "putter" / "AwsPutter.py").read_text(encoding="utf-8"))
    putter = [ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CREDIT"]
    assert manifest.CREDIT == SPEC_CREDIT and putter == [SPEC_CREDIT]


def test_builder_adds_the_keys_only_when_given():
    base = manifest.build_manifest_fragment("171", "2026-09-28T12:00:00Z", 2, {"frames": 5, "method": "median"})
    assert "rhef" not in base and "credit" not in base
    frag = manifest.build_manifest_fragment("171", "2026-09-28T12:00:00Z", 2, {"frames": 5, "method": "median"},
                                            rhef=STAMP, credit=manifest.CREDIT)
    assert (frag["rhef"], frag["credit"]) == (STAMP, SPEC_CREDIT)
    assert {k: v for k, v in frag.items() if k not in ("rhef", "credit")} == base
    assert manifest.validate_fragment(frag) == []


def _trigger(monkeypatch, tmp_path, meta):
    fake = FakeS3()
    lam = load_handler(monkeypatch, fake, fake_ffmpeg(tmp_path))
    fake.objects["1k/rhef_171_1k.png"] = {"Body": b"png", "LastModified": None, "Metadata": meta}
    lam.handler({"Records": [{"eventTime": "2026-09-28T12:31:00Z",
                              "s3": {"object": {"key": "1k/rhef_171_1k.png"}}}]}, None)
    return fake.json("manifest/171.json"), _json.load(open(tmp_path / "argv.json"))


def test_lambda_writes_credit_and_the_stamp(monkeypatch, tmp_path):
    frag, argv = _trigger(monkeypatch, tmp_path, {"obstime": "2026-09-28T12:31:00Z", "rhef_stamp": STAMP})
    assert (frag["rhef"], frag["credit"]) == (STAMP, SPEC_CREDIT)
    assert "description=" + SPEC_CREDIT in argv


def test_lambda_without_a_stamp_writes_credit_only(monkeypatch, tmp_path):
    frag, _ = _trigger(monkeypatch, tmp_path, {"obstime": "2026-09-28T12:31:00Z"})
    assert "rhef" not in frag and frag["credit"] == SPEC_CREDIT


def test_lambda_drops_a_stamp_that_is_not_one_ascii_line(monkeypatch, tmp_path):
    frag, _ = _trigger(monkeypatch, tmp_path, {"obstime": "2026-09-28T12:31:00Z", "rhef_stamp": "a\nb"})
    assert "rhef" not in frag


def test_mp4_description_is_the_credit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(handler, "s3", _FakeS3())
    monkeypatch.setattr(handler.subprocess, "run", lambda argv, **kw: calls.append(list(argv)))
    handler._build_video("171", ["frames/171/20260928T120000_1k.png"], str(tmp_path), None, None)
    assert _metadata_of(calls[0])[-1] == "description=" + SPEC_CREDIT

"""deploy_code.py: deterministic zip, and no write call unless the function name is typed (SB-3)."""
import base64
import hashlib
import io
import json
import pathlib
import zipfile
from datetime import datetime, timezone

import pytest

botocore_session = pytest.importorskip("botocore.session")
from botocore.stub import ANY, Stubber  # noqa: E402

from aws_lambda.video_builder import deploy_code  # noqa: E402

VB = pathlib.Path(deploy_code.__file__).resolve().parent
FN = "sun-video-builder"
LAYER = "arn:aws:lambda:us-east-2:123456789012:layer:ffmpeg-static:3"


def _lambda_client():
    return botocore_session.get_session().create_client(
        "lambda", region_name="us-east-2",
        aws_access_key_id="testing", aws_secret_access_key="testing")


def _get_function(code_sha):
    return {
        "Configuration": {
            "FunctionName": FN, "CodeSha256": code_sha, "RevisionId": "rev-1",
            "LastModified": "2026-10-01T00:00:00.000+0000",
            "Environment": {"Variables": {"VIDEO_FPS": "18"}},
            "Layers": [{"Arn": LAYER}],
        },
        "Code": {"Location": "https://example.invalid/live.zip"},
    }


def _old_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("video_builder/", b"")
        for name in deploy_code.PACKED_FILES:
            z.writestr(f"video_builder/{name}", b"# older code\n")
    return buf.getvalue()


def _clean_preflight():
    return "a" * 40, "lambda-2026-10-01", []


def test_build_zip_is_deterministic_and_sorted():
    a = deploy_code.build_zip(VB)
    b = deploy_code.build_zip(VB)
    assert a == b
    with zipfile.ZipFile(io.BytesIO(a)) as z:
        infos = z.infolist()
    assert [i.filename for i in infos] == [f"video_builder/{n}" for n in sorted(deploy_code.PACKED_FILES)]
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in infos)
    assert all((i.external_attr >> 16) & 0o777 == 0o644 for i in infos)


def test_code_sha256_is_base64_of_sha256():
    assert deploy_code.code_sha256(b"") == base64.b64encode(hashlib.sha256(b"").digest()).decode()
    assert deploy_code.code_sha256(b"") == "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="


def test_redact_hides_account_id():
    assert deploy_code.redact(LAYER) == "arn:aws:lambda:us-east-2:<ACCOUNT>:layer:ffmpeg-static:3"


def test_declined_gate_makes_no_write_call(tmp_path, capsys):
    lam = _lambda_client()
    with Stubber(lam) as stub:
        stub.add_response("get_function", _get_function("live-sha"), {"FunctionName": FN})
        rc = deploy_code.run([], lam=lam, input_fn=lambda prompt: "no",
                             fetch_zip=lambda url: _old_zip(), preflight_fn=_clean_preflight,
                             receipts_dir=tmp_path)
        stub.assert_no_pending_responses()
    assert rc == 2
    assert "declined: nothing was changed" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_refusal_stops_before_the_gate(tmp_path):
    lam = _lambda_client()
    asked = []
    with Stubber(lam) as stub:
        stub.add_response("get_function", _get_function("live-sha"), {"FunctionName": FN})
        rc = deploy_code.run([], lam=lam, input_fn=asked.append,
                             fetch_zip=lambda url: _old_zip(),
                             preflight_fn=lambda: ("a" * 40, "", ["HEAD carries no lambda-* tag"]),
                             receipts_dir=tmp_path)
    assert rc == 1
    assert asked == []


def test_already_live_when_contents_match(tmp_path, capsys):
    lam = _lambda_client()
    same = deploy_code.build_zip(VB)
    with Stubber(lam) as stub:
        stub.add_response("get_function", _get_function("different-bytes-same-files"), {"FunctionName": FN})
        rc = deploy_code.run([], lam=lam, input_fn=lambda prompt: pytest.fail("asked"),
                             fetch_zip=lambda url: same, preflight_fn=_clean_preflight,
                             receipts_dir=tmp_path)
    assert rc == 0
    assert "already live" in capsys.readouterr().out


def test_plan_json_is_read_only_and_reports_pending(tmp_path, capsys):
    lam = _lambda_client()
    with Stubber(lam) as stub:
        stub.add_response("get_function", _get_function("live-sha"), {"FunctionName": FN})
        rc = deploy_code.run(["--plan", "--json"], lam=lam, fetch_zip=lambda url: _old_zip(),
                             preflight_fn=_clean_preflight, receipts_dir=tmp_path)
    plan = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert plan["already_live"] is False
    assert "video_builder/handler.py" in plan["changed_files"]
    assert plan["layers"] == ["arn:aws:lambda:us-east-2:<ACCOUNT>:layer:ffmpeg-static:3"]


def test_typed_name_deploys_and_writes_receipt(tmp_path):
    lam = _lambda_client()
    new_sha = deploy_code.code_sha256(deploy_code.build_zip(VB))
    with Stubber(lam) as stub:
        stub.add_response("get_function", _get_function("live-sha"), {"FunctionName": FN})
        stub.add_response(
            "update_function_code", {"FunctionName": FN, "CodeSha256": new_sha, "Version": "42"},
            {"FunctionName": FN, "ZipFile": ANY, "Publish": True, "RevisionId": "rev-1"})
        stub.add_response("get_function_configuration",
                          {"FunctionName": FN, "LastUpdateStatus": "Successful"}, {"FunctionName": FN})
        stub.add_response("get_function", _get_function(new_sha), {"FunctionName": FN})
        rc = deploy_code.run([], lam=lam, input_fn=lambda prompt: FN,
                             fetch_zip=lambda url: _old_zip(), preflight_fn=_clean_preflight,
                             receipts_dir=tmp_path,
                             now=datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc))
        stub.assert_no_pending_responses()
    assert rc == 0
    receipt = json.loads((tmp_path / "20261001T120000Z.json").read_text())
    assert set(receipt) == {"function", "git_sha", "tag", "code_sha256", "version",
                            "ffmpeg_version", "deployed_at"}
    assert receipt["code_sha256"] == new_sha
    assert receipt["version"] == "42"
    assert "123456789012" not in json.dumps(receipt)

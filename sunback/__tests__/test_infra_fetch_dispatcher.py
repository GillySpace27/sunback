"""infra/public_scan.py and infra/fetch_dispatcher.py against fakes (SB-7 Task 7 tooling). No AWS, no network."""
import base64
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[2] / "infra"
sys.path.insert(0, str(INFRA))
import fetch_dispatcher  # noqa: E402
import public_scan  # noqa: E402

# Built at runtime so no token-, key- or account-shaped literal is committed.
FAKE_TOKEN = "ghp_" + "0" * 36
FAKE_KEY = "AKIA" + "A" * 16
FAKE_ACCOUNT = "1234567890" + "12"
URL = "https://example.invalid/presigned?X-Amz-Security-Token=" + "z" * 20
HANDLER = (
    "import json, os, urllib.request, boto3\n"
    "def handler(event, context):\n"
    "    s = boto3.client('secretsmanager').get_secret_value(SecretId='github-actions-dispatch-token')\n"
    "    token = s['SecretString']\n"
    "    return {'repo': 'GillySpace27/sunback', 'workflow': 'GitCloudRunHourly.yml'}\n"
)


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeLambda:
    def __init__(self, blob=None, error=None, sha=None):
        self.blob, self.error = blob, error
        self.sha = sha or base64.b64encode(hashlib.sha256(blob or b"").digest()).decode()
        self.calls = []

    def __getattr__(self, name):
        raise AssertionError(f"unexpected AWS call: {name}")

    def get_function(self, **kwargs):
        self.calls.append(("get_function", kwargs))
        if self.error:
            raise self.error
        return {"Configuration": {"Runtime": "python3.12", "Handler": "lambda_function.handler",
                                  "CodeSha256": self.sha, "LastModified": "2026-06-01T00:00:00.000+0000",
                                  "Role": "arn:aws:iam::" + FAKE_ACCOUNT + ":role/x"},
                "Code": {"Location": URL, "RepositoryType": "S3"}}


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _run(tmp_path, client, argv=()):
    lines = []
    code = fetch_dispatcher.run(["--dest", str(tmp_path / "d"), *argv], client=client,
                                fetch=lambda url: client.blob, out=lines.append)
    return code, "\n".join(lines)


def test_scan_counts_each_kind_and_prints_no_value(tmp_path, capsys):
    (tmp_path / "a.py").write_text(f"t = '{FAKE_TOKEN}'\nk = '{FAKE_KEY}'\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.json").write_text(f'{{"Account": "{FAKE_ACCOUNT}", "to": "x@example.org"}}\n')
    (tmp_path / "clean.txt").write_text("nothing here; 1234567890123 is 13 digits\n")
    assert public_scan.scan([tmp_path]) == {
        "github-token": 1, "aws-access-key-id": 1, "private-key": 0, "aws-secret-key-name": 0,
        "secret-string": 0, "12-digit-number": 1, "email": 1}
    assert public_scan.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "total: 4"
    for value in (FAKE_TOKEN, FAKE_KEY, FAKE_ACCOUNT, "x@example.org"):
        assert value not in out


def test_scan_clean_tree_and_missing_path(tmp_path, capsys):
    (tmp_path / "ok.py").write_text("print('hello')\n")
    assert public_scan.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == "total: 0\n"
    assert public_scan.main([str(tmp_path / "nope")]) == 2


def test_fetch_clean_code(tmp_path):
    client = FakeLambda(_zip({"lambda_function.py": HANDLER}))
    code, out = _run(tmp_path, client)
    assert code == 0, out
    assert client.calls == [("get_function", {"FunctionName": "sun-reducer-dispatcher"})]
    assert "lambda_function.py" in out and "scan total: 0" in out
    assert URL not in out and "X-Amz-Security-Token" not in out
    d = tmp_path / "d"
    assert (d / "src" / "lambda_function.py").read_text() == HANDLER
    config = (d / "config.json").read_text()
    assert '"Handler": "lambda_function.handler"' in config
    assert URL not in config and FAKE_ACCOUNT not in config  # only the four fields, never the role ARN


def test_fetch_stops_on_a_token_without_printing_it(tmp_path):
    client = FakeLambda(_zip({"lambda_function.py": HANDLER + f"TOKEN = '{FAKE_TOKEN}'\n"}))
    code, out = _run(tmp_path, client)
    assert code == 1
    assert "hits in lambda_function.py: 1 (github-token 1)" in out and "STOP" in out
    assert FAKE_TOKEN not in out


def test_fetch_refuses_a_zip_that_is_not_the_live_code(tmp_path):
    client = FakeLambda(_zip({"lambda_function.py": HANDLER}), sha="bm90IHRoZSBzYW1l")
    code, out = _run(tmp_path, client)
    assert code == 1 and "is not the function's CodeSha256" in out
    assert not (tmp_path / "d" / "src").exists()


def test_fetch_refuses_an_entry_outside_the_destination(tmp_path):
    client = FakeLambda(_zip({"../escape.py": "x = 1\n"}))
    code, out = _run(tmp_path, client)
    assert code == 1 and "outside the destination" in out
    assert not (tmp_path / "escape.py").exists()


@pytest.mark.parametrize("err", ["ResourceNotFoundException", "AccessDeniedException", "UnrecognizedClientException"])
def test_fetch_is_unchecked_on_any_aws_error(tmp_path, err):
    code, out = _run(tmp_path, FakeLambda(error=ClientError(err)), ["--region", "us-west-1"])
    assert code == 3 and out == f"UNCHECKED: cannot read sun-reducer-dispatcher in us-west-1 ({err})"


def test_fetch_refuses_a_destination_inside_the_repository(tmp_path):
    lines = []
    code = fetch_dispatcher.run(["--dest", str(INFRA / "dispatcher-download")], client=FakeLambda(b""),
                                fetch=lambda url: b"", out=lines.append)
    assert code == 1 and "inside the repository" in lines[0]
    assert not (INFRA / "dispatcher-download").exists()


def test_neither_script_writes_to_aws():
    import re

    for name in ("public_scan.py", "fetch_dispatcher.py"):
        assert not re.search(r"\b(put_|create_|delete_|update_|tag_|untag_|invoke)", (INFRA / name).read_text()), name

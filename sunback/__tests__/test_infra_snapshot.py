"""infra/snapshot.py against fake AWS clients (SB-7). No AWS, no network."""
import json
import re
import sys
from pathlib import Path

INFRA = Path(__file__).resolve().parents[2] / "infra"
sys.path.insert(0, str(INFRA))
import snapshot  # noqa: E402

ACCOUNT = "123456789012"
FAKE_TOKEN = "ghp_" + "0" * 36  # built at runtime so no token-shaped literal is committed


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class Fake:
    """A boto3 client stand-in: only the listed methods exist; anything else fails the test."""

    def __init__(self, **methods):
        self._methods = methods
        self.calls = []

    def __getattr__(self, name):
        if name not in self._methods:
            raise AssertionError(f"unexpected AWS call: {name}")

        def call(**kwargs):
            self.calls.append(name)
            result = self._methods[name]
            if isinstance(result, Exception):
                raise result
            return result(**kwargs) if callable(result) else json.loads(json.dumps(result))
        return call


def fake_clients(rule_state="ENABLED"):
    video_cfg = {"FunctionName": "sun-video-builder", "Role": f"arn:aws:iam::{ACCOUNT}:role/sun-video-builder-role",
                 "Runtime": "python3.12", "CodeSha256": "abc=", "LastModified": "2026-09-24", "State": "Active",
                 "Environment": {"Variables": {"SUN_BUCKET": "the-sun-now", "FFMPEG_PATH": "/opt/bin/ffmpeg"}},
                 "Layers": [{"Arn": f"arn:aws:lambda:us-east-2:{ACCOUNT}:layer:ffmpeg-static:3"}],
                 "ResponseMetadata": {"RequestId": "x"}}
    dispatch_cfg = {"FunctionName": "sun-reducer-dispatcher",
                    "Role": f"arn:aws:iam::{ACCOUNT}:role/service-role/sun-reducer-dispatcher-role",
                    "Environment": {"Variables": {"SECRET_NAME": "github-actions-dispatch-token",
                                                  "REPO": "GillySpace27/sunback", "LEAK": FAKE_TOKEN}}}
    configs = {"sun-video-builder": video_cfg, "sun-reducer-dispatcher": dispatch_cfg}
    policy = {"Statement": [{"Sid": "s3invoke", "Condition": {"StringEquals": {"AWS:SourceAccount": ACCOUNT}}}]}
    return {
        "sts": Fake(get_caller_identity={"Account": ACCOUNT}),
        "lambda": Fake(get_function_configuration=lambda FunctionName: configs[FunctionName],
                       get_policy={"Policy": json.dumps(policy), "RevisionId": "r1"},
                       get_function_event_invoke_config=ClientError("ResourceNotFoundException")),
        "events": Fake(describe_rule={"Name": "sun-reducer-20min", "State": rule_state,
                                      "ScheduleExpression": "rate(20 minutes)",
                                      "Arn": f"arn:aws:events:us-east-2:{ACCOUNT}:rule/sun-reducer-20min"},
                       list_targets_by_rule={"Targets": [{"Id": "1", "Arn": f"arn:aws:lambda:us-east-2:{ACCOUNT}:"
                                                                         "function:sun-reducer-dispatcher"}]}),
        "s3": Fake(get_bucket_notification_configuration={"LambdaFunctionConfigurations": []},
                   get_bucket_policy={"Policy": json.dumps({"Statement": [{"Effect": "Allow"}]})},
                   get_bucket_lifecycle_configuration={"Rules": [{"ID": "v-7d", "Filter": {"Prefix": "v/"},
                                                                  "Expiration": {"Days": 7}}]},
                   get_bucket_cors=ClientError("NoSuchCORSConfiguration"),
                   get_public_access_block={"PublicAccessBlockConfiguration": {"BlockPublicAcls": False}},
                   get_bucket_ownership_controls=ClientError("OwnershipControlsNotFoundError")),
        "iam": Fake(get_role=lambda RoleName: {"Role": {"RoleName": RoleName, "RoleLastUsed": {"x": 1},
                                                        "AssumeRolePolicyDocument": "{\"Version\": \"2012-10-17\"}"}},
                    list_role_policies={"PolicyNames": ["sun-bucket-access"]},
                    get_role_policy={"PolicyDocument": {"Statement": []}},
                    list_attached_role_policies={"AttachedPolicies": [
                        {"PolicyArn": "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"}]}),
        "secretsmanager": Fake(describe_secret={"Name": "github-actions-dispatch-token",
                                                "ARN": f"arn:aws:secretsmanager:us-east-2:{ACCOUNT}:secret:x",
                                                "Tags": [{"Key": "pat-expires", "Value": "2026-12-31"}],
                                                "VersionIdsToStages": {"v1": ["AWSCURRENT"]},
                                                "LastAccessedDate": "2026-10-01"}),
        "budgets": Fake(describe_budget=ClientError("AccessDeniedException"),
                        describe_notifications_for_budget={"Notifications": [{"Threshold": 20.0}]}),
    }


def test_snapshot_reads_every_section_and_writes_sorted_json(tmp_path):
    clients = fake_clients()
    assert snapshot.main(["--out", str(tmp_path)], clients=clients) == 3  # the Budget read is denied
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["budget-whole-account-monthly.json", "events-sun-reducer-20min.json",
                     "iam-role-sun-reducer-dispatcher-role.json", "iam-role-sun-video-builder-role.json",
                     "lambda-sun-reducer-dispatcher.json", "lambda-sun-video-builder.json",
                     "s3-the-sun-now.json", "secret-github-actions-dispatch-token.json"]
    video = json.loads((tmp_path / "lambda-sun-video-builder.json").read_text())
    assert "CodeSha256" not in video["configuration"] and "ResponseMetadata" not in video["configuration"]
    assert video["event_invoke_config"] is None
    assert video["policy"]["Statement"][0]["Condition"]["StringEquals"]["AWS:SourceAccount"] == "<ACCOUNT>"
    secret = json.loads((tmp_path / "secret-github-actions-dispatch-token.json").read_text())
    assert "VersionIdsToStages" not in secret and secret["Tags"][0]["Key"] == "pat-expires"
    budget = json.loads((tmp_path / "budget-whole-account-monthly.json").read_text())
    assert budget["budget"] == {"unchecked": "AccessDeniedException"}


def test_no_account_id_secret_value_or_token_is_written(tmp_path):
    clients = fake_clients()
    snapshot.main(["--out", str(tmp_path)], clients=clients)
    text = "".join(p.read_text() for p in tmp_path.iterdir())
    assert ACCOUNT not in text
    assert FAKE_TOKEN not in text
    assert re.search(r"(^|[^0-9])[0-9]{12}([^0-9]|$)", text) is None
    dispatch = json.loads((tmp_path / "lambda-sun-reducer-dispatcher.json").read_text())
    assert dispatch["configuration"]["Environment"]["Variables"] == {
        "SECRET_NAME": "<REDACTED>", "REPO": "GillySpace27/sunback", "LEAK": "<REDACTED>"}
    assert "get_secret_value" not in clients["secretsmanager"].calls
    assert clients["budgets"].calls == ["describe_budget", "describe_notifications_for_budget"]


def test_snapshot_makes_no_write_calls():
    source = (INFRA / "snapshot.py").read_text()
    assert re.findall(r"put_|create_|delete_|update_", source) == []


def test_no_credentials_is_unchecked(tmp_path, capsys):
    clients = fake_clients()
    clients["sts"] = Fake(get_caller_identity=ClientError("NoCredentialsError"))
    assert snapshot.main(["--out", str(tmp_path / "live")], clients=clients) == 3
    assert "UNCHECKED: cannot read the AWS account (NoCredentialsError)" in capsys.readouterr().err
    assert not (tmp_path / "live").exists()

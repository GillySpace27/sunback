#!/usr/bin/env python3
"""Read-only snapshot of the AWS side of the live Sun pipeline (SB-7).

Writes one sorted JSON file per resource, with volatile fields removed and the
AWS account id replaced by <ACCOUNT>: both Lambdas (configuration, resource
policy, async invoke config), the EventBridge rule and its targets, the bucket
(notification, policy, lifecycle, CORS, public-access block, ownership), the
two Lambda roles with their inline and attached policies, the dispatch secret's
metadata and tags (never its value) and the account Budget with its
notification thresholds (never its subscribers, which are email addresses).

Only get, describe and list calls; the repository is public, so every value is
also scrubbed of anything shaped like a GitHub token or an AWS access key.

    python infra/snapshot.py --out infra/live      # Gilly's AWS profile

Exit 0 when every section was read, 3 when any section is UNCHECKED (no
credentials, or a permission is missing; the section records the error code).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REGION = "us-east-2"
BUCKET = "the-sun-now"
VIDEO_FUNCTION = "sun-video-builder"
DISPATCH_FUNCTION = "sun-reducer-dispatcher"
RULE = "sun-reducer-20min"
SECRET_ID = "github-actions-dispatch-token"
BUDGET = "whole-account-monthly"
ACCOUNT_PLACEHOLDER = "<ACCOUNT>"
REDACTED = "<REDACTED>"

# Fields that change without anyone changing the configuration.
VOLATILE = {
    "lambda": {"LastModified", "RevisionId", "LastUpdateStatus", "LastUpdateStatusReason",
               "LastUpdateStatusReasonCode", "State", "StateReason", "StateReasonCode",
               "CodeSha256", "CodeSize", "RuntimeVersionConfig"},
    "secret": {"LastAccessedDate", "LastChangedDate", "LastRotatedDate", "NextRotationDate",
               "VersionIdsToStages"},
    "budget": {"CalculatedSpend", "LastUpdatedTime", "HealthStatus"},
    "iam": {"RoleLastUsed", "UpdateDate", "AttachmentCount", "DefaultVersionId"},
}
TOKEN_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|A[KS]IA[0-9A-Z]{16}")
SENSITIVE_NAME_RE = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE|CREDENTIAL|API_?KEY|ACCESS_?KEY|(^|_)PAT($|_)",
                               re.IGNORECASE)
ARN_ACCOUNT_RE = re.compile(r"(arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:)\d{12}(:)")


def error_code(exc):
    """The AWS error code of a botocore ClientError, else the exception's class name."""
    response = getattr(exc, "response", None) or {}
    return response.get("Error", {}).get("Code") or type(exc).__name__


def read(call, absent=()):
    """call(); None when AWS says the thing is not configured; {"unchecked": code} on any other error."""
    try:
        return call()
    except Exception as exc:  # recorded, never raised: one unreadable section must not hide the rest
        code = error_code(exc)
        return None if code in absent else {"unchecked": code}


def strip(obj, keys):
    """Copy of obj without ResponseMetadata and without the given keys, at any depth."""
    if isinstance(obj, dict):
        return {k: strip(v, keys) for k, v in obj.items() if k != "ResponseMetadata" and k not in keys}
    if isinstance(obj, list):
        return [strip(v, keys) for v in obj]
    return obj


def parse_json_text(value):
    """Policies come back as JSON strings; parse them so the snapshot diffs line by line."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def redact(obj, account):
    """Account id to <ACCOUNT>, token-shaped strings and sensitive environment values to <REDACTED>."""
    def walk(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "Variables" and isinstance(v, dict):
                    v = {name: REDACTED if SENSITIVE_NAME_RE.search(name) else val for name, val in v.items()}
                out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            text = TOKEN_RE.sub(REDACTED, node)
            text = ARN_ACCOUNT_RE.sub(lambda m: m.group(1) + ACCOUNT_PLACEHOLDER + m.group(2), text)
            return text.replace(account, ACCOUNT_PLACEHOLDER) if account else text
        return node
    return walk(json.loads(json.dumps(obj, default=str)))


def role_name(role_arn):
    return role_arn.rsplit("/", 1)[-1] if isinstance(role_arn, str) and ":role/" in role_arn else None


def lambda_section(lam, name):
    config = read(lambda: lam.get_function_configuration(FunctionName=name))
    policy = read(lambda: lam.get_policy(FunctionName=name), absent=("ResourceNotFoundException",))
    if isinstance(policy, dict) and "Policy" in policy:
        policy = parse_json_text(policy["Policy"])
    invoke = read(lambda: lam.get_function_event_invoke_config(FunctionName=name),
                  absent=("ResourceNotFoundException",))
    return strip({"configuration": config, "policy": policy, "event_invoke_config": invoke}, VOLATILE["lambda"])


def role_section(iam, name):
    role = read(lambda: iam.get_role(RoleName=name))
    if isinstance(role, dict) and "Role" in role:
        role = role["Role"]
        role["AssumeRolePolicyDocument"] = parse_json_text(role.get("AssumeRolePolicyDocument"))
    inline = {}
    names = read(lambda: iam.list_role_policies(RoleName=name, MaxItems=100))
    for pol in (names or {}).get("PolicyNames", []):
        doc = read(lambda: iam.get_role_policy(RoleName=name, PolicyName=pol))
        inline[pol] = parse_json_text(doc.get("PolicyDocument")) if "PolicyDocument" in (doc or {}) else doc
    attached = {}
    listed = read(lambda: iam.list_attached_role_policies(RoleName=name, MaxItems=100))
    for pol in (listed or {}).get("AttachedPolicies", []):
        arn = pol["PolicyArn"]
        if arn.startswith("arn:aws:iam::aws:policy/"):
            attached[arn] = "aws-managed"
            continue
        meta = read(lambda: iam.get_policy(PolicyArn=arn))
        version = (meta or {}).get("Policy", {}).get("DefaultVersionId")
        doc = read(lambda: iam.get_policy_version(PolicyArn=arn, VersionId=version)) if version else meta
        attached[arn] = parse_json_text((doc or {}).get("PolicyVersion", {}).get("Document", doc))
    if isinstance(names, dict) and "unchecked" in names:
        inline = names
    if isinstance(listed, dict) and "unchecked" in listed:
        attached = listed
    return strip({"role": role, "inline_policies": inline, "attached_policies": attached}, VOLATILE["iam"])


# --- SB-8: pipeline health resources (read-only: describe/get/list calls only) ---
HEALTH_ALARMS = (
    "sun-video-builder-no-invocations",
    "sun-video-builder-errors",
    "sun-reducer-dispatcher-errors",
)
ALERT_TOPIC_NAME = "sun-pipeline-alerts"
FAILURE_FUNCTION = "sun-video-builder"
_ALARM_FIELDS = (
    "AlarmName", "Namespace", "MetricName", "Dimensions", "Statistic", "Period",
    "EvaluationPeriods", "DatapointsToAlarm", "Threshold", "ComparisonOperator",
    "TreatMissingData", "ActionsEnabled", "AlarmActions", "OKActions",
)


def snapshot_alarms(cw, names=HEALTH_ALARMS):
    """The SB-8 alarms' definitions (no state, no timestamps), sorted by name."""
    resp = cw.describe_alarms(AlarmNames=list(names))
    alarms = [{k: a.get(k) for k in _ALARM_FIELDS} for a in resp.get("MetricAlarms", [])]
    return sorted(alarms, key=lambda a: a["AlarmName"])


def snapshot_alert_topic(sns, name=ALERT_TOPIC_NAME):
    """The alert topic and its subscriptions; e-mail endpoints become <EMAIL> (public repo)."""
    arns = []
    token = None
    while True:
        resp = sns.list_topics(**({"NextToken": token} if token else {}))
        arns += [t["TopicArn"] for t in resp.get("Topics", []) if t["TopicArn"].rsplit(":", 1)[-1] == name]
        token = resp.get("NextToken")
        if not token:
            break
    if not arns:
        return {"TopicName": name, "exists": False}
    subs = sns.list_subscriptions_by_topic(TopicArn=arns[0]).get("Subscriptions", [])
    return {
        "TopicName": name,
        "exists": True,
        "TopicArn": arns[0],
        "Subscriptions": sorted(
            ({"Protocol": s["Protocol"],
              "Endpoint": "<EMAIL>" if s["Protocol"] in ("email", "email-json") else s["Endpoint"],
              "Confirmed": s["SubscriptionArn"] not in ("PendingConfirmation", "Deleted")}
             for s in subs),
            key=lambda s: (s["Protocol"], s["Endpoint"])),
    }


def snapshot_event_invoke_config(lam, function=FAILURE_FUNCTION):
    """Async retry settings and the on-failure destination of the video Lambda."""
    try:
        cfg = lam.get_function_event_invoke_config(FunctionName=function)
    except Exception as exc:  # botocore ClientError; matched by code so fakes and old clients behave alike
        if error_code(exc) != "ResourceNotFoundException":
            raise
        return {"FunctionName": function, "configured": False}
    return {
        "FunctionName": function,
        "configured": True,
        "MaximumRetryAttempts": cfg.get("MaximumRetryAttempts"),
        "MaximumEventAgeInSeconds": cfg.get("MaximumEventAgeInSeconds"),
        "DestinationConfig": cfg.get("DestinationConfig", {}),
    }


def take(clients):
    """{filename: data} for every resource, redacted. Raises when the account id cannot be read."""
    account = clients["sts"].get_caller_identity()["Account"]
    lam, s3, iam = clients["lambda"], clients["s3"], clients["iam"]
    snap = {}
    roles = []
    for name in (VIDEO_FUNCTION, DISPATCH_FUNCTION):
        section = lambda_section(lam, name)
        snap[f"lambda-{name}.json"] = section
        role = role_name((section["configuration"] or {}).get("Role"))
        if role and role not in roles:
            roles.append(role)
    events = clients["events"]
    snap[f"events-{RULE}.json"] = {
        "rule": strip(read(lambda: events.describe_rule(Name=RULE)), set()),
        "targets": strip(read(lambda: events.list_targets_by_rule(Rule=RULE)), set()),
    }
    bucket = {
        "notification": read(lambda: s3.get_bucket_notification_configuration(Bucket=BUCKET)),
        "policy": read(lambda: s3.get_bucket_policy(Bucket=BUCKET), absent=("NoSuchBucketPolicy",)),
        "lifecycle": read(lambda: s3.get_bucket_lifecycle_configuration(Bucket=BUCKET),
                          absent=("NoSuchLifecycleConfiguration",)),
        "cors": read(lambda: s3.get_bucket_cors(Bucket=BUCKET), absent=("NoSuchCORSConfiguration",)),
        "public_access_block": read(lambda: s3.get_public_access_block(Bucket=BUCKET),
                                    absent=("NoSuchPublicAccessBlockConfiguration",)),
        "ownership_controls": read(lambda: s3.get_bucket_ownership_controls(Bucket=BUCKET),
                                   absent=("OwnershipControlsNotFoundError",)),
    }
    if isinstance(bucket["policy"], dict) and "Policy" in bucket["policy"]:
        bucket["policy"] = parse_json_text(bucket["policy"]["Policy"])
    snap[f"s3-{BUCKET}.json"] = strip(bucket, set())
    for role in roles:
        snap[f"iam-role-{role}.json"] = role_section(iam, role)
    secret = read(lambda: clients["secretsmanager"].describe_secret(SecretId=SECRET_ID))
    snap[f"secret-{SECRET_ID}.json"] = strip(secret, VOLATILE["secret"])
    # SB-8: pipeline health resources. A missing resource is recorded as the function returns it
    # ([], {"exists": false}, {"configured": false}); only an unreadable one is "unchecked".
    snap["cloudwatch_alarms.json"] = read(lambda: snapshot_alarms(clients["cloudwatch"]))
    snap[f"sns_{ALERT_TOPIC_NAME}.json"] = read(lambda: snapshot_alert_topic(clients["sns"]))
    snap[f"lambda_{FAILURE_FUNCTION}_event_invoke_config.json"] = read(lambda: snapshot_event_invoke_config(lam))
    budgets = clients["budgets"]
    snap[f"budget-{BUDGET}.json"] = strip({
        "budget": read(lambda: budgets.describe_budget(AccountId=account, BudgetName=BUDGET)),
        "notifications": read(lambda: budgets.describe_notifications_for_budget(
            AccountId=account, BudgetName=BUDGET)),
    }, VOLATILE["budget"])
    return {name: redact(data, account) for name, data in snap.items()}


def render(data):
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def unchecked_sections(snap):
    """Paths like 'budget-whole-account-monthly.json:budget' whose read failed."""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            if set(node) == {"unchecked"}:
                found.append(f"{path} ({node['unchecked']})")
                return
            for k, v in node.items():
                walk(v, f"{path}:{k}")
    for name, data in sorted(snap.items()):
        walk(data, name)
    return found


def make_clients(region=REGION):
    import boto3  # imported here so the tests and --help need no AWS libraries

    session = boto3.session.Session(region_name=region)
    clients = {name: session.client(name)
               for name in ("lambda", "events", "s3", "iam", "secretsmanager", "sts", "cloudwatch", "sns")}
    clients["budgets"] = session.client("budgets", region_name="us-east-1")
    return clients


def main(argv=None, clients=None):
    parser = argparse.ArgumentParser(description="Read-only, redacted snapshot of the AWS side (SB-7).")
    parser.add_argument("--out", default="infra/live", help="directory to write (default infra/live, gitignored)")
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args(argv)
    try:
        snap = take(clients or make_clients(args.region))
    except Exception as exc:  # no boto3, no credentials, no network
        print(f"UNCHECKED: cannot read the AWS account ({error_code(exc)})", file=sys.stderr)
        return 3
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(snap.items()):
        (out / name).write_text(render(data), encoding="utf-8")
    print(f"wrote {len(snap)} files to {out}")
    missing = unchecked_sections(snap)
    for item in missing:
        print(f"UNCHECKED: {item}")
    return 3 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

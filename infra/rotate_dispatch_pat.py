#!/usr/bin/env python3
"""Rotate the GitHub PAT that sun-reducer-dispatcher uses (SB-7). Gilly runs this; agents never do.

Reads the new token with getpass (never echoed, never printed, never written to
disk), shows a plan, and changes nothing until Gilly types ``rotate``. Then it
stores the token as the new AWSCURRENT version of the secret (the old token
stays as AWSPREVIOUS), tags the secret pat-expires=<date>, invokes the
dispatcher once (this starts ONE real production reducer run) and waits for a
new workflow_dispatch run of GitCloudRunHourly.yml to appear.

    python infra/rotate_dispatch_pat.py       # in Gilly's own terminal, his AWS profile, gh logged in

Exit 0 a new run appeared, 1 the dispatcher failed or no run appeared (the
rollback command is printed), 2 refused or declined (nothing changed), 3 gh is
unavailable after the secret changed (check the Actions tab by hand).
See infra/RUNBOOK-pat.md.
"""
from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone

SECRET_ID = "github-actions-dispatch-token"
DISPATCH_FUNCTION = "sun-reducer-dispatcher"
REPO = "GillySpace27/sunback"
WORKFLOW = "GitCloudRunHourly.yml"
REGION = "us-east-2"
TAG_KEY = "pat-expires"
CONFIRM_WORD = "rotate"
# How the dispatcher reads the secret, set from the reviewed copy in infra/dispatcher/:
# None means the SecretString is the bare token; a name means {"<name>": "<token>"}.
SECRET_JSON_FIELD = None
TOKEN_SHAPE = re.compile(r"(github_pat_|ghp_)[A-Za-z0-9_]{16,}")


def gh_runs(repo=REPO, workflow=WORKFLOW):
    """The five newest workflow_dispatch runs, as gh reports them."""
    out = subprocess.run(
        ["gh", "run", "list", "--repo", repo, "--workflow", workflow, "--event", "workflow_dispatch",
         "--limit", "5", "--json", "databaseId,createdAt,url,status"],
        capture_output=True, text=True, check=True, timeout=60)
    return json.loads(out.stdout or "[]")


def _clients(region):
    import boto3

    return boto3.client("secretsmanager", region_name=region), boto3.client("lambda", region_name=region)


def _when(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def run(argv=None, sm=None, lam=None, ask=input, secret_prompt=getpass.getpass, gh=gh_runs,
        now=lambda: datetime.now(timezone.utc), sleep=time.sleep):
    parser = argparse.ArgumentParser(description="Rotate the dispatcher PAT (Gilly only, SB-7).")
    parser.add_argument("--secret-id", default=SECRET_ID)
    parser.add_argument("--function", default=DISPATCH_FUNCTION)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--wait", type=int, default=300, help="seconds to wait for the test run")
    args = parser.parse_args(argv)

    expires_text = ask("Expiry date GitHub shows for the new token (YYYY-MM-DD): ").strip()
    try:
        expires = date.fromisoformat(expires_text)
    except ValueError:
        print(f"REFUSED: {expires_text!r} is not YYYY-MM-DD. Nothing changed.")
        return 2
    if expires <= now().date():
        print(f"REFUSED: {expires.isoformat()} is not in the future. Nothing changed.")
        return 2
    token = secret_prompt("New token (input hidden): ").strip()
    if not TOKEN_SHAPE.fullmatch(token):
        print("REFUSED: that does not look like a GitHub token (github_pat_... or ghp_...). Nothing changed.")
        return 2

    print("Plan:")
    print(f"  1. store the new token as the AWSCURRENT version of {args.secret_id} "
          "(the old token stays as AWSPREVIOUS)")
    print(f"  2. tag {args.secret_id} with {TAG_KEY}={expires.isoformat()}")
    print(f"  3. invoke {args.function} once: this starts ONE real production reducer run")
    print(f"  4. wait up to {args.wait} s for a new workflow_dispatch run of {WORKFLOW} in {REPO}")
    if ask(f"Type {CONFIRM_WORD} to go ahead: ").strip() != CONFIRM_WORD:
        print("Declined. Nothing changed.")
        return 2

    if sm is None or lam is None:
        sm, lam = _clients(args.region)
    stages = sm.describe_secret(SecretId=args.secret_id).get("VersionIdsToStages", {})
    previous = next((vid for vid, st in stages.items() if "AWSCURRENT" in st), "<previous version id>")
    value = token if SECRET_JSON_FIELD is None else json.dumps({SECRET_JSON_FIELD: token})
    started = now()
    new_version = sm.put_secret_value(SecretId=args.secret_id, SecretString=value)["VersionId"]
    del token, value
    print(f"stored version {new_version}; previous version {previous} is AWSPREVIOUS")
    sm.tag_resource(SecretId=args.secret_id, Tags=[{"Key": TAG_KEY, "Value": expires.isoformat()}])
    print(f"tagged {TAG_KEY}={expires.isoformat()}")
    rollback = (f"To go back to the previous token: aws secretsmanager update-secret-version-stage "
                f"--region {args.region} --secret-id {args.secret_id} --version-stage AWSCURRENT "
                f"--move-to-version-id {previous} --remove-from-version-id {new_version}")

    reply = lam.invoke(FunctionName=args.function, InvocationType="RequestResponse", Payload=b"{}")
    if reply.get("FunctionError"):
        print(f"FAIL: {args.function} returned FunctionError={reply['FunctionError']}; read its CloudWatch log.")
        print(rollback)
        return 1
    print(f"invoked {args.function} (status {reply.get('StatusCode')}); waiting for the run")

    deadline = started + timedelta(seconds=args.wait)
    while now() < deadline:
        try:
            runs = gh()
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"UNCHECKED: gh run list failed ({type(exc).__name__}); look for a workflow_dispatch run "
                  f"of {WORKFLOW} created after {started.isoformat()} in the Actions tab.")
            return 3
        fresh = [r for r in runs if _when(r["createdAt"]) >= started - timedelta(seconds=5)]
        if fresh:
            print(f"OK: run {fresh[0]['url']} created {fresh[0]['createdAt']} (after {started.isoformat()})")
            return 0
        sleep(10)
    print(f"FAIL: no new workflow_dispatch run of {WORKFLOW} within {args.wait} s.")
    print(rollback)
    return 1


if __name__ == "__main__":
    sys.exit(run())

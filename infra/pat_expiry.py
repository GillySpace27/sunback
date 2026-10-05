#!/usr/bin/env python3
"""Days until the dispatcher's GitHub PAT expires, from a tag on its secret (SB-7). Read-only.

The secret github-actions-dispatch-token carries the tag pat-expires=YYYY-MM-DD
(written by infra/rotate_dispatch_pat.py). Only describe-secret is called; the
secret value is never read.

    python infra/pat_expiry.py [--warn-days 14] [--json]

Exit 0 more than --warn-days left, 1 inside the window or expired, 3 UNCHECKED
(no tag, a malformed tag, no permission or no credentials).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone

SECRET_ID = "github-actions-dispatch-token"
TAG_KEY = "pat-expires"
REGION = "us-east-2"
RUNBOOK = "infra/RUNBOOK-pat.md"


def expiry_date(secret_id=SECRET_ID, client=None):
    """The pat-expires tag as a date, or None when the tag is absent. ValueError if malformed."""
    if client is None:
        import boto3

        client = boto3.client("secretsmanager", region_name=REGION)
    tags = client.describe_secret(SecretId=secret_id).get("Tags", [])
    value = next((t["Value"] for t in tags if t.get("Key") == TAG_KEY), None)
    return None if value is None else date.fromisoformat(value)


def days_until_expiry(secret_id=SECRET_ID, now=None, client=None):
    """Whole days from today (UTC) to the pat-expires date; negative once expired; None without a tag."""
    expires = expiry_date(secret_id, client)
    if expires is None:
        return None
    today = (now or datetime.now(timezone.utc)).date()
    return (expires - today).days


def main(argv=None, client=None, now=None):
    parser = argparse.ArgumentParser(description="Dispatcher PAT expiry check (SB-7).")
    parser.add_argument("--warn-days", type=int, default=14)
    parser.add_argument("--secret-id", default=SECRET_ID)
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)
    result = {"secret": args.secret_id, "expires": None, "days": None, "warn_days": args.warn_days}
    try:
        expires = expiry_date(args.secret_id, client)
    except ValueError as exc:
        result.update(status="UNCHECKED", detail=f"{TAG_KEY} tag is not YYYY-MM-DD ({exc})")
    except Exception as exc:  # no boto3, no credentials, no permission
        code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code") or type(exc).__name__
        result.update(status="UNCHECKED", detail=f"cannot describe {args.secret_id} ({code})")
    else:
        if expires is None:
            result.update(status="UNCHECKED", detail=f"{args.secret_id} has no {TAG_KEY} tag; see {RUNBOOK}")
        else:
            days = (expires - (now or datetime.now(timezone.utc)).date()).days
            result.update(expires=expires.isoformat(), days=days)
            if days < 0:
                result.update(status="FAIL", detail=f"dispatcher PAT expired {-days} day(s) ago; see {RUNBOOK}")
            elif days <= args.warn_days:
                result.update(status="FAIL", detail=f"dispatcher PAT expires in {days} day(s); rotate, see {RUNBOOK}")
            else:
                result.update(status="OK", detail=f"dispatcher PAT expires in {days} day(s)")
    if args.json:
        print(json.dumps(result))
    else:
        print(f"{result['status']}: {result['detail']}" + (f" ({result['expires']})" if result["expires"] else ""))
    return {"OK": 0, "FAIL": 1, "UNCHECKED": 3}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())

"""Freshness probe for the live imagery chain (suite SU-2).

Read-only GETs of Gilly's own public endpoints; no viewer telemetry. Each check
reports PASS, FAIL or UNCHECKED; UNCHECKED never fails the run. Exit 1 when any
check is FAIL, else 0.

    python3 -m devtools.scripts.freshness_probe [--json]

Environment:
    FRESHNESS_S3_MAX_AGE_S   image_times.txt age limit in seconds (default 3600)
    FRESHNESS_R2_MAX_AGE_S   R2 manifest/index.json "generated" age limit (default 28800)
    FRESHNESS_ARM_R2_INDEX=1 an R2 index 404 counts as FAIL (set only once the
                             in-flight ledger marks claude/versioned-keys live)
"""
import argparse
import collections
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

Check = collections.namedtuple("Check", "id status detail")

S3_IMAGE_TIMES_URL = "https://the-sun-now.s3.us-east-2.amazonaws.com/image_times.txt"
R2_INDEX_URL = "https://imagery.myheliograph.com/manifest/index.json"
SUN_HTML_URL = "https://gilly.space/sun.html"
APPCAST_URL = "https://gilly.space/heliograph/appcast.xml"
APPCAST_HELIOGRAM_URL = "https://gilly.space/heliogram/appcast.xml"
STORE_BUILD_INFO_URL = "https://myheliograph.com/api/build-info"
BUDGET_NAME = "whole-account-monthly"

S3_MAX_AGE_S = 3600     # estimated; the reducer gate uses 1800 for its own decision
R2_MAX_AGE_S = 28800    # estimated; the mirror Worker runs "17 */6 * * *"
TIMEOUT_S = 20
STORE_TIMEOUT_S = 30    # the store's Fly machine scales to zero and wakes slowly


def urllib_fetch(url, timeout):
    """GET url; return (status, headers, body). Network errors return status 0."""
    req = urllib.request.Request(url, headers={"User-Agent": "sunback-freshness-probe"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() or b""
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {}, str(e).encode()


def _why(code, body):
    """HTTP status for a detail line; the error text when the GET never got one."""
    return f"HTTP {code}" if code else "unreachable: " + body.decode("utf-8", "replace")[:120]


def _utc(text):
    """Parse an ISO time; a missing zone means UTC (image_times.txt has none)."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1]
    t = dt.datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _age_check(cid, now, stamp, limit):
    try:
        age = int((now - _utc(stamp)).total_seconds())
    except (TypeError, ValueError) as e:
        return Check(cid, "FAIL", f"unreadable time {stamp!r}: {e}")
    status = "PASS" if age < limit else "FAIL"
    return Check(cid, status, f"age {age} s (limit {limit} s)")


def aws_budget(env=None, run=subprocess.run):
    """("PASS"|"FAIL"|"UNCHECKED", detail) for the AWS Budget's presence.

    UNCHECKED without AWS credentials, without the aws CLI, or when the
    caller lacks budgets:ViewBudget. Never creates or changes anything.
    """
    env = os.environ if env is None else env
    if not any(env.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_PROFILE")):
        return "UNCHECKED", "no AWS credentials in this run"
    if shutil.which("aws") is None:
        return "UNCHECKED", "aws CLI not installed"
    ident = run(["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
                capture_output=True, text=True)
    if ident.returncode != 0:
        return "UNCHECKED", "sts get-caller-identity failed"
    names = run(["aws", "budgets", "describe-budgets", "--account-id", ident.stdout.strip(),
                 "--query", "Budgets[].BudgetName", "--output", "text"],
                capture_output=True, text=True)
    if names.returncode != 0:
        return "UNCHECKED", "no budgets:ViewBudget permission"
    if BUDGET_NAME in names.stdout.split():
        return "PASS", f"budget {BUDGET_NAME} present"
    return "FAIL", f"budget {BUDGET_NAME} missing"


def run_checks(now, fetch, env=None, budget=None):
    """Run every check at `now` (aware datetime) with `fetch(url, timeout)`."""
    env = os.environ if env is None else env
    s3_limit = int(env.get("FRESHNESS_S3_MAX_AGE_S", S3_MAX_AGE_S))
    r2_limit = int(env.get("FRESHNESS_R2_MAX_AGE_S", R2_MAX_AGE_S))
    armed = env.get("FRESHNESS_ARM_R2_INDEX") == "1"
    checks = []

    code, _, body = fetch(S3_IMAGE_TIMES_URL, TIMEOUT_S)
    if code != 200:
        checks.append(Check("s3-image-times", "FAIL", _why(code, body)))
    else:
        checks.append(_age_check("s3-image-times", now, body.decode("utf-8", "replace"), s3_limit))

    code, _, body = fetch(R2_INDEX_URL, TIMEOUT_S)
    if code == 404 and not armed:
        checks.append(Check("r2-index", "UNCHECKED", "HTTP 404; not armed (FRESHNESS_ARM_R2_INDEX)"))
    elif code != 200:
        checks.append(Check("r2-index", "FAIL", _why(code, body)))
    else:
        try:
            generated = json.loads(body)["generated"]
        except (ValueError, KeyError, TypeError) as e:
            checks.append(Check("r2-index", "FAIL", f"no generated field: {e}"))
        else:
            checks.append(_age_check("r2-index", now, generated, r2_limit))

    code, _, body = fetch(SUN_HTML_URL, TIMEOUT_S)
    ok = code == 200 and len(body) > 0
    checks.append(Check("sun-html", "PASS" if ok else "FAIL", f"{_why(code, body)}, {len(body)} bytes"))

    code, _, body = fetch(APPCAST_URL, TIMEOUT_S)
    ok = code == 200 and b"sparkle:version" in body
    checks.append(Check("appcast", "PASS" if ok else "FAIL", f"{_why(code, body)}, sparkle:version {'found' if ok else 'missing'}"))

    code, _, body = fetch(APPCAST_HELIOGRAM_URL, TIMEOUT_S)
    if code != 200:
        checks.append(Check("appcast-heliogram", "UNCHECKED", f"{_why(code, body)}; not live before Heliogram 0.8"))
    else:
        ok = b"sparkle:version" in body
        checks.append(Check("appcast-heliogram", "PASS" if ok else "FAIL",
                            f"HTTP 200, sparkle:version {'found' if ok else 'missing'}"))

    code, _, body = fetch(STORE_BUILD_INFO_URL, STORE_TIMEOUT_S)
    try:
        ok = code == 200 and "built" in json.loads(body)
    except (ValueError, TypeError):
        ok = False
    checks.append(Check("store-build-info", "PASS" if ok else "FAIL", _why(code, body)))

    status, detail = (budget or (lambda: aws_budget(env)))()
    checks.append(Check("aws-budget", status, detail))
    return checks


def exit_code(checks):
    return 1 if any(c.status == "FAIL" for c in checks) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print a JSON list instead of lines")
    args = ap.parse_args(argv)
    checks = run_checks(dt.datetime.now(dt.timezone.utc), urllib_fetch)
    if args.json:
        print(json.dumps([c._asdict() for c in checks], indent=2))
    else:
        for c in checks:
            print(f"{c.status} {c.id} {c.detail}")
    return exit_code(checks)


if __name__ == "__main__":
    sys.exit(main())

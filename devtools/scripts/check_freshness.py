"""Is the live Sun fresh? Read-only GETs against the public bucket (SB-8).

    python devtools/scripts/check_freshness.py [--threshold 3600] [--key KEY ...]
        [--prefix P] [--base-url URL] [--check-pat] [--json]

Default keys: image_times.txt plus manifest/<id>.json for every id in
aws_lambda/video_builder/manifest.py PRODUCTS. A fragment's age comes from its
``updated`` field; any other key's age from the HTTP Last-Modified header.
Exit 0 fresh, 1 stale or missing, 3 unreachable (UNCHECKED). With --json,
exactly one JSON object is printed to stdout. The 3600 s default is estimated:
the 20-minute cadence plus one missed run plus margin.
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aws_lambda.video_builder.manifest import PRODUCTS, manifest_key  # noqa: E402

DEFAULT_BASE_URL = "https://the-sun-now.s3.us-east-2.amazonaws.com/"
DEFAULT_THRESHOLD_S = 3600
TIMEOUT_S = 20
# manifest/<id>.json (any prefix), but not manifest/index.json
FRAGMENT_RE = re.compile(r"(^|/)manifest/(?!index\.json$)[^/]+\.json$")


class Unreachable(Exception):
    """The source could not be asked (network, DNS, TLS): UNCHECKED, not stale."""


def default_keys():
    return ["image_times.txt"] + [manifest_key(p["id"]) for p in PRODUCTS]


def _get(url):
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "sunback-check-freshness"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return resp.read(), resp.headers.get("Last-Modified")
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None, None  # missing (403 counts too: S3 can answer 403 for a key that does not exist)
        raise Unreachable(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, FileNotFoundError):
            return None, None  # file:// base URL in tests
        raise Unreachable(str(exc.reason)) from exc
    except OSError as exc:
        raise Unreachable(str(exc)) from exc


def _parse_updated(text):
    text = text.strip()
    when = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def key_age_s(base_url, key, now):
    """Age in seconds of one public key, or None when it is missing or has no time.

    Raises Unreachable when the source cannot be asked.
    """
    body, last_modified = _get(base_url + key)
    if body is None:
        return None
    if FRAGMENT_RE.search(key):
        try:
            when = _parse_updated(json.loads(body)["updated"])
        except (ValueError, KeyError, TypeError):
            return None
    else:
        if not last_modified:
            return None
        when = parsedate_to_datetime(last_modified)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
    return max(0, int((now - when).total_seconds()))


def check_pat():
    """Run SB-7's infra/pat_expiry.py as a subprocess: (exit code, parsed JSON or an error dict)."""
    script = REPO_ROOT / "infra" / "pat_expiry.py"
    if not script.exists():
        return 3, {"error": "infra/pat_expiry.py not present (SB-7 not landed)"}
    proc = subprocess.run([sys.executable, str(script), "--json"], capture_output=True, text=True, timeout=60)
    try:
        detail = json.loads(proc.stdout)
    except ValueError:
        detail = {"stdout": proc.stdout.strip()[-200:], "stderr": proc.stderr.strip()[-200:]}
    return (proc.returncode if proc.returncode in (0, 1, 3) else 3), detail


def run(argv=None, now=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD_S, help="seconds (default 3600)")
    ap.add_argument("--key", action="append", dest="keys", help="key to check (repeatable)")
    ap.add_argument("--prefix", default="", help='key prefix, for example "staging/"')
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--check-pat", action="store_true", help="also run infra/pat_expiry.py (needs AWS read)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    now = now or datetime.now(timezone.utc)
    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    results = []
    for key in args.keys or default_keys():
        full = args.prefix + key
        try:
            age = key_age_s(base, full, now)
            status = "missing" if age is None else ("stale" if age > args.threshold else "fresh")
        except Unreachable as exc:
            age, status = None, "unreachable"
            print(f"UNCHECKED: {full}: {exc}", file=sys.stderr)
        results.append({"key": full, "age_s": age, "status": status})

    ages = [r["age_s"] for r in results if r["age_s"] is not None]
    statuses = {r["status"] for r in results}
    if statuses & {"stale", "missing"}:
        code = 1
    elif "unreachable" in statuses:
        code = 3
    else:
        code = 0

    pat = None
    if args.check_pat:
        pat_code, pat_detail = check_pat()
        pat = {"exit": pat_code, "detail": pat_detail}
        if pat_code == 1:
            code = 1
        elif pat_code == 3 and code == 0:
            code = 3

    doc = {"ok": code == 0, "exit": code, "threshold_s": args.threshold,
           "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "worst_age_s": max(ages) if ages else None, "results": results, "pat": pat}
    if args.json:
        print(json.dumps(doc))
    else:
        for r in results:
            age = "-" if r["age_s"] is None else f"{r['age_s']}s"
            print(f"{r['status'].upper():11} {age:>8}  {r['key']}")
        if pat is not None:
            print(f"PAT        exit {pat['exit']}: {json.dumps(pat['detail'])}")
        print({0: "OK: fresh", 1: "FAIL: stale or missing", 3: "UNCHECKED: source unreachable"}[code])
    return code


if __name__ == "__main__":
    sys.exit(run())

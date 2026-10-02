#!/usr/bin/env python3
"""Classify open Dependabot alerts by whether production installs the package (SB-13).

Read only: one `gh api --paginate` GET of the repository's open Dependabot alerts.
Classes, first match wins:
  in-lock        the package is pinned in requirements-reducer.txt (the reducer image)
  pyproject      declared in pyproject.toml (dependencies or an extra), not in the lock
  lambda         boto3 or its dependency closure, which the Lambda Python runtime provides
  not-installed  none of the above: the alert concerns a file nothing installs
Usage: python devtools/scripts/alert_triage.py [--json] [--count] [--repo OWNER/NAME]
Default output: a Markdown table and one summary line per class.
Exit: 0 no high or critical alert in the in-lock class, 1 at least one, 3 UNCHECKED
(gh missing, not logged in, or no permission to read Dependabot alerts).
Until requirements-reducer.txt exists the in-lock class is empty.
"""

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements-reducer.txt"
REPO = "GillySpace27/sunback"
CLASSES = ("in-lock", "pyproject", "lambda", "not-installed")
# boto3's install requirements and theirs (read from package metadata on 2026-10-01).
LAMBDA_PACKAGES = frozenset({"boto3", "botocore", "s3transfer", "jmespath", "python-dateutil", "urllib3", "six"})
JQ = (".[] | {number, state, severity: .security_advisory.severity, ghsa: .security_advisory.ghsa_id, "
      "package: .dependency.package.name, ecosystem: .dependency.package.ecosystem, "
      "manifest: .dependency.manifest_path} | @json")


def normalize(name):
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(line):
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    return normalize(match.group(1)) if match else None


def lock_names(text):
    """Names pinned in a requirements file body (comments and blank lines skipped)."""
    names = set()
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith(("#", "-")):
            name = _requirement_name(line)
            if name:
                names.add(name)
    return names


def pyproject_names(text):
    project = tomllib.loads(text)["project"]
    reqs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        reqs.extend(extra)
    return {n for n in (_requirement_name(r) for r in reqs) if n}


def classify(alerts, locked, declared):
    """Return the alerts, each with a "class" key added."""
    rows = []
    for alert in alerts:
        name = normalize(alert.get("package") or "")
        if name in locked:
            cls = "in-lock"
        elif name in declared:
            cls = "pyproject"
        elif name in LAMBDA_PACKAGES:
            cls = "lambda"
        else:
            cls = "not-installed"
        rows.append({**alert, "class": cls})
    return rows


def fetch_alerts(repo):
    """Open alerts as a list of dicts, or None when gh cannot read them."""
    if shutil.which("gh") is None:
        return None
    proc = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{repo}/dependabot/alerts?state=open&per_page=100", "--jq", JQ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return None
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def summarize(rows):
    out = {}
    for cls in CLASSES:
        members = [r for r in rows if r["class"] == cls]
        out[cls] = {"count": len(members),
                    "high_or_critical": sum(r.get("severity") in ("high", "critical") for r in members)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify open Dependabot alerts (SB-13).")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    parser.add_argument("--count", action="store_true", help="print only the number of open alerts")
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args(argv)

    alerts = fetch_alerts(args.repo)
    if alerts is None:
        if args.json:
            print(json.dumps({"status": "UNCHECKED", "reason": "gh missing or no access to Dependabot alerts"}))
        else:
            print("UNCHECKED: alert_triage (gh missing or no access to Dependabot alerts)")
        return 3
    locked = lock_names(LOCK.read_text()) if LOCK.exists() else set()
    declared = pyproject_names((ROOT / "pyproject.toml").read_text())
    rows = classify(alerts, locked, declared)
    summary = summarize(rows)
    status = 1 if summary["in-lock"]["high_or_critical"] else 0

    if args.count:
        print(len(rows))
    elif args.json:
        print(json.dumps({"status": "FAIL" if status else "PASS", "total": len(rows), "summary": summary,
                          "alerts": rows}))
    else:
        print("| # | Severity | Package | Manifest | Class |")
        print("|---|---|---|---|---|")
        order = {c: i for i, c in enumerate(CLASSES)}
        for r in sorted(rows, key=lambda r: (order[r["class"]], r.get("package") or "", r.get("number") or 0)):
            print(f"| {r.get('number')} | {r.get('severity')} | {r.get('package')} | {r.get('manifest')} | "
                  f"{r['class']} |")
        print()
        for cls in CLASSES:
            print(f"{cls}: {summary[cls]['count']} open, {summary[cls]['high_or_critical']} high or critical")
        print(f"total: {len(rows)}")
    return status


if __name__ == "__main__":
    sys.exit(main())

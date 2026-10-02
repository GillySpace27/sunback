#!/usr/bin/env python3
"""Drift between the committed AWS snapshot and the live account (SB-7). Read-only.

Takes a fresh snapshot in memory with infra/snapshot.py (same reads, same
redaction) and compares it file by file with infra/declared/.

    python infra/diff.py [--json] [--declared infra/declared]   # Gilly's AWS profile

Exit 0 no drift, 1 drift (the unified diff is printed), 3 UNCHECKED: no
declared snapshot yet, no credentials, or a section could not be read and
nothing else drifted.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import snapshot  # noqa: E402

DECLARED = Path(__file__).resolve().parent / "declared"


def load_declared(path):
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(path).glob("*.json"))}


def compare(declared, live):
    """{filename: unified diff lines} for every file that differs, is new or is gone."""
    drift = {}
    for name in sorted(set(declared) | set(live)):
        old, new = declared.get(name, ""), live.get(name, "")
        if old != new:
            drift[name] = list(difflib.unified_diff(
                old.splitlines(), new.splitlines(), f"declared/{name}", f"live/{name}", lineterm=""))
    return drift


def main(argv=None, clients=None):
    parser = argparse.ArgumentParser(description="Drift between infra/declared/ and live AWS (SB-7).")
    parser.add_argument("--declared", default=str(DECLARED))
    parser.add_argument("--region", default=snapshot.REGION)
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)
    result = {"declared": args.declared, "drift": {}, "unchecked": [], "status": "OK"}
    declared = load_declared(args.declared)
    if not declared:
        result.update(status="UNCHECKED", unchecked=[f"no *.json in {args.declared} (first commit is Gilly's)"])
    else:
        try:
            snap = snapshot.take(clients or snapshot.make_clients(args.region))
        except Exception as exc:  # no boto3, no credentials, no network
            result.update(status="UNCHECKED", unchecked=[f"cannot read the AWS account ({snapshot.error_code(exc)})"])
        else:
            live = {name: snapshot.render(data) for name, data in snap.items()}
            result["drift"] = compare(declared, live)
            result["unchecked"] = snapshot.unchecked_sections(snap)
            result["status"] = "DRIFT" if result["drift"] else ("UNCHECKED" if result["unchecked"] else "OK")
    code = {"OK": 0, "DRIFT": 1, "UNCHECKED": 3}[result["status"]]
    if args.json:
        print(json.dumps(result))
    else:
        for name, lines in result["drift"].items():
            print("\n".join(lines))
        for item in result["unchecked"]:
            print(f"UNCHECKED: {item}")
        print(f"{result['status']}: {len(result['drift'])} file(s) drifted, {len(result['unchecked'])} unchecked")
    return code


if __name__ == "__main__":
    sys.exit(main())

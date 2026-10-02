#!/usr/bin/env python3
"""Audit built sunback wheels (SB-12). Stdlib only; reads the wheel, writes nothing.

A wheel fails when it has a top-level entry other than sunback/ and
<name>.dist-info/, any .dll, .ipynb or .timestamp file, or any file over
1 MB (1,000,000 bytes uncompressed).

Usage: python devtools/scripts/check_wheel.py dist/*.whl [--json]
Exit codes: 0 every wheel clean, 1 a rule broken, 3 UNCHECKED (no wheel given,
or a path that is not a .whl file, such as an unexpanded dist/*.whl glob).
With --json, stdout is exactly one JSON object:
{"status": "PASS" | "FAIL" | "UNCHECKED", "wheels": [{"wheel", "entries", "bytes", "problems"}]}.
"""

import argparse
import json
import os
import sys
import zipfile

MAX_BYTES = 1_000_000
BANNED_SUFFIXES = (".dll", ".ipynb", ".timestamp")


def audit(path):
    """Return {"wheel", "entries", "bytes", "problems"} for one wheel file."""
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
    problems = []
    for info in infos:
        name = info.filename
        top = name.split("/", 1)[0]
        if top != "sunback" and not top.endswith(".dist-info"):
            problems.append(f"top-level entry outside sunback/: {name}")
        if name.lower().endswith(BANNED_SUFFIXES):
            problems.append(f"banned file type: {name}")
        if info.file_size > MAX_BYTES:
            problems.append(f"file over 1 MB: {name} ({info.file_size} bytes)")
    return {"wheel": str(path), "entries": len(infos), "bytes": sum(i.file_size for i in infos), "problems": problems}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit built sunback wheels (SB-12).")
    parser.add_argument("wheels", nargs="*", help="wheel files, for example dist/*.whl")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)

    missing = [w for w in args.wheels if not (w.endswith(".whl") and os.path.isfile(w))]
    if not args.wheels or missing:
        why = f"not a wheel file: {', '.join(missing)}" if missing else "no wheel given"
        if args.json:
            print(json.dumps({"status": "UNCHECKED", "reason": why, "wheels": []}))
        else:
            print(f"UNCHECKED: wheel ({why}; run python -m build first)")
        return 3

    reports = [audit(w) for w in args.wheels]
    status = "FAIL" if any(r["problems"] for r in reports) else "PASS"
    if args.json:
        print(json.dumps({"status": status, "wheels": reports}))
    else:
        for r in reports:
            print(f"{r['wheel']}: {r['entries']} entries, {r['bytes']} bytes uncompressed")
            for problem in r["problems"]:
                print(f"  {problem}")
        print(f"{status}: wheel")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

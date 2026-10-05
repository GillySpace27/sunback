#!/usr/bin/env python3
"""Download the live sun-reducer-dispatcher code for review, read only (SB-7 Task 7, Gilly runs it).

One `lambda:GetFunction` call with Gilly's AWS profile, then a download of the
code zip from the presigned URL that call returns. The URL carries temporary
credentials: it is held in memory and never printed or written. The zip's
SHA-256 is checked against the function's CodeSha256, the zip is unpacked into
--dest (outside the repository; refused inside it), and every file is scanned
by count with infra/public_scan.py before anyone opens one.

    python infra/fetch_dispatcher.py                   # --dest /tmp/sb7-dispatcher

Writes <dest>/config.json (Runtime, Handler, CodeSha256, LastModified),
<dest>/code.zip and <dest>/src/. Prints the file list and, per file with a hit,
the scan counts; never a matched value.

Exit 0 downloaded and the scan total is 0 (go on with Task 7 Step 3), 1 the scan
found something (do not open the files; Gilly reviews them himself) or the zip
does not match CodeSha256, 3 UNCHECKED (no credentials, no permission, function
not found in --region: ask Gilly for its region).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_scan  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FUNCTION = "sun-reducer-dispatcher"
REGION = "us-east-2"
DEST = Path("/tmp/sb7-dispatcher")
CONFIG_FIELDS = ("Runtime", "Handler", "CodeSha256", "LastModified")
# Code that reads the secret names the field it reads; that is expected here (Task 7 Step 3 checks how).
SCAN_KINDS = {k: v for k, v in public_scan.KINDS.items() if k != "secret-string"}


def error_code(exc):
    response = getattr(exc, "response", None) or {}
    return response.get("Error", {}).get("Code") or type(exc).__name__


def download(url):
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (presigned https URL from AWS)
        return response.read()


def safe_extract(zip_path, dest):
    """Unpack, refusing absolute paths and parent references. Returns the sorted relative file names."""
    names = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            rel = Path(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"zip entry outside the destination: {info.filename}")
            if not info.is_dir():
                names.append(info.filename)
        zf.extractall(dest)
    return sorted(names)


def run(argv=None, client=None, fetch=download, out=print):
    parser = argparse.ArgumentParser(description="Download the dispatcher code for review (SB-7, read only).")
    parser.add_argument("--function", default=FUNCTION)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args(argv)

    dest = args.dest.resolve()
    if dest == REPO or REPO in dest.parents:
        out(f"FAIL: --dest {dest} is inside the repository; use a folder outside it")
        return 1

    try:
        if client is None:
            import boto3

            client = boto3.client("lambda", region_name=args.region)
        response = client.get_function(FunctionName=args.function)
    except Exception as exc:  # no credentials, no permission, wrong region: reported, never raised
        out(f"UNCHECKED: cannot read {args.function} in {args.region} ({error_code(exc)})")
        return 3

    config = {k: response["Configuration"].get(k) for k in CONFIG_FIELDS}
    blob = fetch(response["Code"]["Location"])
    del response  # the presigned URL goes no further
    got = base64.b64encode(hashlib.sha256(blob).digest()).decode()
    if got != config["CodeSha256"]:
        out(f"FAIL: downloaded zip SHA-256 {got} is not the function's CodeSha256 {config['CodeSha256']}")
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dest / "code.zip").write_bytes(blob)
    try:
        names = safe_extract(dest / "code.zip", dest / "src")
    except ValueError as exc:
        out(f"FAIL: {exc}")
        return 1

    out(json.dumps(config, indent=2, sort_keys=True))
    out(f"{len(names)} file(s) in {dest / 'src'}:")
    for name in names:
        out(f"  {name}")
    total = 0
    for path, found in public_scan.per_file([dest / "src"], SCAN_KINDS).items():
        n = sum(found.values())
        total += n
        out(f"hits in {path.relative_to(dest / 'src')}: {n} ({', '.join(f'{k} {v}' for k, v in found.items() if v)})")
    out(f"scan total: {total}")
    if total:
        out("STOP: do not open these files. Gilly reviews them himself before anything is copied (SB-7 Task 7).")
        return 1
    out("OK: nothing secret-shaped; review each first-party file in full next (SB-7 Task 7 Step 3)")
    return 0


if __name__ == "__main__":
    sys.exit(run())

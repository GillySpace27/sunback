#!/usr/bin/env python3
"""Count anything that must not reach the public repository, without printing it (SB-7).

Scans every file under the given paths for token-, key-, account-id-, email- and
SecretString-shaped text and prints only counts: one line per file with a hit
(its path and the count of each kind), then `total: <n>`. It never prints a
matched value, a line or a line number, so it is safe to run on code nobody has
reviewed yet (the dispatcher zip, SB-7 Task 7 Step 2) and on a fresh snapshot
(infra/live, Task 8 Step 2).

    python infra/public_scan.py /tmp/sb7-dispatcher/src
    python infra/public_scan.py infra/live

Exit 0 when the total is 0, 1 when anything matched (do not open the files: Gilly
reviews them himself), 2 when a path does not exist.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KINDS = {
    "github-token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    "aws-access-key-id": re.compile(r"A[KS]IA[0-9A-Z]{16}"),
    "private-key": re.compile(r"-----BEGIN"),
    "aws-secret-key-name": re.compile(r"aws_secret_access_key", re.IGNORECASE),
    "secret-string": re.compile(r"SecretString"),
    "12-digit-number": re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}


def files(paths):
    for path in paths:
        path = Path(path)
        if path.is_file():
            yield path
        else:
            yield from sorted(p for p in path.rglob("*") if p.is_file())


def per_file(paths, kinds=None):
    """{file: {kind: number of matches}} for every file with at least one match (read as latin-1)."""
    kinds = KINDS if kinds is None else kinds
    hits = {}
    for path in files(paths):
        text = path.read_bytes().decode("latin-1")
        found = {kind: len(pattern.findall(text)) for kind, pattern in kinds.items()}
        if any(found.values()):
            hits[path] = found
    return hits


def scan(paths, kinds=None):
    """{kind: number of matches} over every file under paths."""
    kinds = KINDS if kinds is None else kinds
    counts = dict.fromkeys(kinds, 0)
    for found in per_file(paths, kinds).values():
        for kind, n in found.items():
            counts[kind] += n
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Count secret-shaped text without printing it (SB-7).")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--allow", action="append", default=[], choices=sorted(KINDS),
                        help="leave this kind out (for example secret-string in dispatcher code, which "
                             "names the field it reads); repeatable")
    args = parser.parse_args(argv)
    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"UNCHECKED: no such path: {', '.join(missing)}")
        return 2
    hits = per_file(args.paths, {k: v for k, v in KINDS.items() if k not in args.allow})
    total = 0
    for path, found in hits.items():
        n = sum(found.values())
        total += n
        print(f"{path}: {n} ({', '.join(f'{k} {v}' for k, v in found.items() if v)})")
    print(f"total: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

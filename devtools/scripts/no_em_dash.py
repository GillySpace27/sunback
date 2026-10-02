#!/usr/bin/env python3
# heliosoftware-vendored: Website:heliosoftware/spec/tools/no_em_dash.py sha256=3e81d98a79655d7bedbbe04202d2fc5267a703afafce2bb3673076e6e91aadcc
"""Fail when a change ADDS a line containing an em dash (U+2014).

    python3 no_em_dash.py [--base REF] [--exclude GLOB]...     # git diff --unified=0 REF...HEAD
    git diff --unified=0 main...HEAD | python3 no_em_dash.py --stdin

Only added lines are scanned, so older text that still carries em dashes is never
flagged and nothing has to be swept to turn the guard on. Prints one `path:line`
per hit (line numbers of the new file) and exits 1; exits 0 when clean; exits 2
when git cannot produce the diff (an unknown --base is an error, never a pass).
Committed changes only: commit before running it, as CI does.

Suite shared name 11 (SU-10). Standard library only. The dash is written as an
escape so this file never contains the character itself.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys

DASH = "\u2014"
HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def added_hits(diff: str) -> list[tuple[str, int]]:
    """(path, new line number) of every added line that contains DASH, in diff order."""
    hits: list[tuple[str, int]] = []
    path = None
    old_left = new_left = new_line = 0
    for raw in diff.splitlines():
        if old_left > 0 or new_left > 0:       # inside a hunk: count by the header's lengths
            if raw.startswith("\\"):           # "\ No newline at end of file"
                continue
            tag = raw[:1]
            if tag == "+":
                new_left -= 1
                if path is not None and DASH in raw:
                    hits.append((path, new_line))
                new_line += 1
            elif tag == "-":
                old_left -= 1
            else:                              # context line (absent with --unified=0)
                old_left -= 1
                new_left -= 1
                new_line += 1
            continue
        if raw.startswith("+++ "):
            name = raw[4:].split("\t", 1)[0].strip()
            if len(name) > 1 and name.startswith('"') and name.endswith('"'):
                name = name[1:-1]
            if name == "/dev/null":
                path = None
            else:
                path = name[2:] if name[:2] in ("a/", "b/") else name
            continue
        m = HUNK.match(raw)
        if m:
            old_left = int(m.group(1)) if m.group(1) is not None else 1
            new_line = int(m.group(2))
            new_left = int(m.group(3)) if m.group(3) is not None else 1
    return hits


def git_diff(base: str) -> str:
    cmd = ["git", "-c", "core.quotepath=off", "diff", "--unified=0", "--no-color", "--no-ext-diff",
           "--diff-filter=ACMR", f"{base}...HEAD", "--"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"git diff {base}...HEAD failed: {msg[0] if msg else 'no message'}")
    return r.stdout.decode("utf-8", "replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="origin/HEAD", help="compare REF...HEAD (default origin/HEAD)")
    ap.add_argument("--stdin", action="store_true", help="read a unified diff from stdin instead of running git")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="skip paths matching this fnmatch glob (repeatable)")
    args = ap.parse_args(argv)
    if args.stdin:
        diff = sys.stdin.buffer.read().decode("utf-8", "replace")
    else:
        try:
            diff = git_diff(args.base)
        except (RuntimeError, OSError) as exc:
            print(f"no_em_dash: {exc}", file=sys.stderr)
            return 2
    hits = [(p, n) for p, n in added_hits(diff) if not any(fnmatch.fnmatch(p, g) for g in args.exclude)]
    for p, n in hits:
        print(f"{p}:{n}")
    if hits:
        print(f"no_em_dash: {len(hits)} added line(s) contain U+2014; "
              "use a colon, semicolon, comma or parentheses", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Orrery tracker for the sunback live imagery chain (suite SU-5).

Wraps devtools.scripts.freshness_probe.run_checks (SU-2): one milestone per
probe, done only when that probe says PASS. UNCHECKED shows as not done with
its reason, never as PASS. Public URLs only; prints no secret.

  python3 -m devtools.scripts.nrt_status [--json] [--emit]
  python3 devtools/scripts/nrt_status.py [--json] [--emit]
"""
import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from devtools.scripts.freshness_probe import run_checks, urllib_fetch  # noqa: E402

NAME = "sunback-nrt"
TITLE = "Sunback live imagery"
HOW = ("shell", "python3 -m devtools.scripts.freshness_probe --json")
LABELS = {
    "s3-image-times": "S3 image_times.txt is fresh",
    "r2-index": "R2 manifest/index.json is fresh",
    "sun-html": "gilly.space/sun.html serves",
    "appcast": "Legacy /heliograph/appcast.xml carries sparkle:version",
    "appcast-heliogram": "/heliogram/appcast.xml carries sparkle:version",
    "store-build-info": "myheliograph.com/api/build-info answers",
    "aws-budget": "AWS budget whole-account-monthly exists",
}


def snapshot(checks, now):
    milestones = []
    for c in checks:
        note = c.detail if c.status == "PASS" else f"{c.status}: {c.detail}"
        milestones.append({"key": c.id, "label": LABELS.get(c.id, c.id),
                           "done": c.status == "PASS", "note": note, "gated": False,
                           "how_kind": HOW[0], "how": HOW[1]})
    statuses = {c.status for c in checks}
    verdict = "FAIL" if "FAIL" in statuses else ("UNCHECKED" if "UNCHECKED" in statuses else "PASS")
    return {
        "name": NAME,
        "title": TITLE,
        "checked_at": now.isoformat(timespec="seconds"),
        "complete": sum(1 for m in milestones if m["done"]),
        "total": len(milestones),
        "next": next((m["label"] for m in milestones if not m["done"]), None),
        "external_state": verdict,
        "external_label": "freshness probe verdict",
        "milestones": milestones,
    }


def render(snap):
    lines = [f"{snap['title']}: {snap['complete']}/{snap['total']} ({snap['external_state']})"]
    for m in snap["milestones"]:
        lines.append(f"{'[x]' if m['done'] else '[ ]'} {m['label']}  ({m['note']})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Orrery tracker for the sunback live imagery chain.")
    ap.add_argument("--json", action="store_true", help="print the snapshot as JSON")
    ap.add_argument("--emit", action="store_true",
                    help="also write ~/.claude/runbooks/state/sunback-nrt.json (a cache, never truth)")
    args = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    snap = snapshot(run_checks(now, urllib_fetch), now)
    print(json.dumps(snap, indent=2) if args.json else render(snap))
    if args.emit:
        d = os.path.expanduser("~/.claude/runbooks/state")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, NAME + ".json"), "w") as f:
            json.dump(snap, f, indent=2)
        print(f"(snapshot written to {d}/{NAME}.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

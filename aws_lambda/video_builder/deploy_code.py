"""Code-only deploy of the sun-video-builder Lambda (SB-3).

Builds a deterministic zip of exactly the files deploy.py packs, shows live
versus new CodeSha256 and a per-file diff, and changes nothing until the
operator types the function name. Then update_function_code(Publish=True),
wait, re-read, check the live sha, env and layers, and write a receipt.
It never touches the ffmpeg layer, the environment or the bucket notification.

  python aws_lambda/video_builder/deploy_code.py --plan [--json]
  python aws_lambda/video_builder/deploy_code.py
  python aws_lambda/video_builder/deploy_code.py --rollback aws_lambda/video_builder/receipts/<stamp>.json

Exit codes: 0 deployed or already live (with --plan: live equals the source);
1 refused, failed, or (with --plan) a deploy is pending; 2 declined at the
typed gate; 3 AWS unreachable or no credentials (UNCHECKED).
Agents run --plan only. A real deploy or rollback is Gilly's action.
"""
import argparse
import base64
import difflib
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REGION = "us-east-2"
FUNCTION = "sun-video-builder"
PACKED_FILES = ("__init__.py", "handler.py", "frame_queue.py", "manifest.py")
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
RECEIPTS = HERE / "receipts"
ENV_FILE = HERE / "lambda_env.json"
LOCK_FILE = HERE / "layer" / "ffmpeg.lock"


def build_zip(src_dir, files=PACKED_FILES):
    """Deterministic Lambda zip: sorted names, fixed timestamps, mode 0o644."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(f"video_builder/{name}", date_time=FIXED_DATE)
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (Path(src_dir) / name).read_bytes())
    return buf.getvalue()


def code_sha256(data):
    """Base64 SHA-256 of the zip bytes, the form Lambda reports as CodeSha256."""
    return base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")


def zip_members(data):
    """{name: bytes} for the regular files in a zip; directory entries skipped."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}


def file_diff(live, new):
    """Unified diff lines between two {name: bytes} maps; [] when equal."""
    out = []
    for name in sorted(set(live) | set(new)):
        a, b = live.get(name), new.get(name)
        if a == b:
            continue
        if a is None or b is None:
            out.append(f"only in {'new' if a is None else 'live'}: {name}\n")
        out += difflib.unified_diff(
            (a or b"").decode("utf-8", "replace").splitlines(keepends=True),
            (b or b"").decode("utf-8", "replace").splitlines(keepends=True),
            f"live/{name}", f"new/{name}")
    return out


def redact(arn):
    """Replace the 12-digit account id in an ARN with <ACCOUNT> (public repo)."""
    return re.sub(r":\d{12}:", ":<ACCOUNT>:", arn)


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=REPO, check=check, capture_output=True, text=True)


def preflight():
    """(sha, tag, refusals) for building from the working tree at HEAD."""
    refusals = []
    sha = git("rev-parse", "HEAD").stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        refusals.append("tracked files have uncommitted changes")
    if git("merge-base", "--is-ancestor", "HEAD", "origin/master", check=False).returncode != 0:
        refusals.append("HEAD is not on origin/master (git fetch origin; never deploy an unmerged branch)")
    tags = sorted(git("tag", "--points-at", "HEAD", "--list", "lambda-*").stdout.split())
    if not tags:
        refusals.append("HEAD carries no lambda-* tag")
    return sha, (tags[-1] if tags else ""), refusals


def rollback_source(receipt, dest):
    """(src_dir, sha, tag, refusals) rebuilt from the receipt's tag with git archive."""
    tag = receipt["tag"]
    sha = git("rev-list", "-n", "1", tag, check=False).stdout.strip()
    if not sha:
        return None, "", tag, [f"tag {tag} not found (git fetch --tags origin)"]
    refusals = []
    if git("merge-base", "--is-ancestor", sha, "origin/master", check=False).returncode != 0:
        refusals.append(f"tag {tag} is not on origin/master")
    paths = [f"aws_lambda/video_builder/{n}" for n in PACKED_FILES]
    tar = subprocess.run(["git", "archive", "--format=tar", tag, *paths], cwd=REPO,
                         check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        for member in tf.getmembers():
            if member.isfile():
                (Path(dest) / Path(member.name).name).write_bytes(tf.extractfile(member).read())
    return Path(dest), sha, tag, refusals


def live_state(lam, function):
    resp = lam.get_function(FunctionName=function)
    cfg = resp["Configuration"]
    return {
        "code_sha256": cfg["CodeSha256"],
        "revision_id": cfg.get("RevisionId", ""),
        "last_modified": cfg.get("LastModified", ""),
        "env": cfg.get("Environment", {}).get("Variables", {}),
        "layers": [layer["Arn"] for layer in cfg.get("Layers", [])],
        "location": resp.get("Code", {}).get("Location", ""),
    }


def fetch_url(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def env_drift(live_env, env_file=ENV_FILE):
    """Compare the live environment with lambda_env.json; None when the file is absent."""
    if not Path(env_file).exists():
        return None
    declared = json.loads(Path(env_file).read_text())
    return {
        "differs": sorted(k for k in declared if k in live_env and live_env[k] != declared[k]),
        "absent_live": sorted(k for k in declared if k not in live_env),
        "extra_live": sorted(k for k in live_env if k not in declared),
    }


def make_plan(function, live, live_zip, new_zip, sha, tag, refusals):
    live_files, new_files = zip_members(live_zip), zip_members(new_zip)
    new_sha = code_sha256(new_zip)
    return {
        "function": function,
        "git_sha": sha,
        "tag": tag,
        "live_code_sha256": live["code_sha256"],
        "new_code_sha256": new_sha,
        "live_zip_matches_reported_sha": code_sha256(live_zip) == live["code_sha256"],
        "already_live": live["code_sha256"] == new_sha or live_files == new_files,
        "changed_files": sorted(n for n in set(live_files) | set(new_files)
                                if live_files.get(n) != new_files.get(n)),
        "diff": file_diff(live_files, new_files),
        "layers": [redact(a) for a in live["layers"]],
        "env_drift": env_drift(live["env"]),
        "refusals": refusals,
    }


def print_plan(plan):
    print(f"function:          {plan['function']}")
    print(f"source:            {plan['git_sha']} ({plan['tag'] or 'no lambda-* tag'})")
    print(f"live CodeSha256:   {plan['live_code_sha256']}")
    print(f"new CodeSha256:    {plan['new_code_sha256']}")
    print("sha check:         base64 sha256 of the downloaded live zip "
          f"{'equals' if plan['live_zip_matches_reported_sha'] else 'DIFFERS FROM'} the reported CodeSha256")
    print(f"changed files:     {', '.join(plan['changed_files']) or 'none'}")
    sys.stdout.writelines(plan["diff"])
    print("layer, env, notification: unchanged (code-only deploy)")
    drift = plan["env_drift"]
    if drift is None:
        print("env vs lambda_env.json: lambda_env.json missing")
    else:
        print("env vs lambda_env.json: " + "; ".join(
            f"{k}: {', '.join(v) or 'none'}" for k, v in drift.items()))
    for r in plan["refusals"]:
        print(f"REFUSED: {r}")


def write_receipt(receipt, receipts_dir=RECEIPTS):
    Path(receipts_dir).mkdir(parents=True, exist_ok=True)
    stamp = receipt["deployed_at"].replace("-", "").replace(":", "")
    path = Path(receipts_dir) / f"{stamp}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


def ffmpeg_version(layers, lock_file=LOCK_FILE):
    lock = json.loads(Path(lock_file).read_text()) if Path(lock_file).exists() else {}
    return f"layers={','.join(layers) or 'none'}; lock_sha256={lock.get('sha256', 'unknown')}"


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true", help="read-only: print the plan and exit")
    ap.add_argument("--json", action="store_true", help="with --plan: print one JSON object")
    ap.add_argument("--rollback", metavar="RECEIPT", help="redeploy the tag recorded in RECEIPT")
    ap.add_argument("--function", default=FUNCTION)
    args = ap.parse_args(argv)
    if args.json and not args.plan:
        ap.error("--json needs --plan")
    return args


def run(argv=None, lam=None, input_fn=input, fetch_zip=fetch_url,
        preflight_fn=preflight, receipts_dir=RECEIPTS, now=None):
    args = parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        if args.rollback:
            src, sha, tag, refusals = rollback_source(json.loads(Path(args.rollback).read_text()), tmp)
            if src is None:
                print(f"REFUSED: {refusals[0]}")
                return 1
        else:
            src = HERE
            sha, tag, refusals = preflight_fn()
        new_zip = build_zip(src)
    try:
        if lam is None:
            import boto3
            lam = boto3.client("lambda", region_name=REGION)
        live = live_state(lam, args.function)
        live_zip = fetch_zip(live["location"])
    except Exception as exc:  # no boto3, no credentials, no network: nothing was changed
        print(f"UNCHECKED: cannot read {args.function}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    plan = make_plan(args.function, live, live_zip, new_zip, sha, tag, refusals)
    if args.plan:
        if args.json:
            print(json.dumps(plan))
        else:
            print_plan(plan)
        return 0 if plan["already_live"] else 1
    print_plan(plan)
    if plan["already_live"]:
        print("already live")
        return 0
    if refusals:
        return 1
    print(f"Type the function name ({args.function}) to deploy; anything else stops.")
    if input_fn("> ").strip() != args.function:
        print("declined: nothing was changed")
        return 2
    resp = lam.update_function_code(FunctionName=args.function, ZipFile=new_zip,
                                    Publish=True, RevisionId=live["revision_id"])
    lam.get_waiter("function_updated").wait(FunctionName=args.function)
    after = live_state(lam, args.function)
    now = now or datetime.now(timezone.utc)
    receipt = {
        "function": args.function,
        "git_sha": sha,
        "tag": tag,
        "code_sha256": plan["new_code_sha256"],
        "version": resp.get("Version", ""),
        "ffmpeg_version": ffmpeg_version([redact(a) for a in after["layers"]]),
        "deployed_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = write_receipt(receipt, receipts_dir)
    print(f"receipt: {path}")
    problems = []
    if after["code_sha256"] != plan["new_code_sha256"]:
        problems.append(f"live CodeSha256 {after['code_sha256']} != {plan['new_code_sha256']}")
    if after["env"] != live["env"]:
        problems.append("environment changed during the deploy")
    if after["layers"] != live["layers"]:
        problems.append("layers changed during the deploy")
    for p in problems:
        print(f"POST-CHECK FAILED: {p}")
    if problems:
        return 1
    print(f"deployed {args.function} version {receipt['version']}; commit {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

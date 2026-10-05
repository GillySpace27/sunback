#!/usr/bin/env python3
"""Probe every public key of the-sun-now over HTTPS (SB-6). Read-only GET and HEAD.

Per product: GET manifest/<id>.json (status, Content-Type, Cache-Control, schema,
key names, age of ``updated``), the 1k still (PNG signature), the thumbnail (PNG,
512 x 512), the fixed video (video/mp4) and, with ffprobe on PATH, the video's
colour tags (bt709 primaries, transfer and matrix, tv range). Then
manifest/index.json, one versioned v/ pair (immutable Cache-Control) and
image_times.txt. The contract is aws_lambda/video_builder/CONTRACT.md.

Exit 0 when nothing failed, 1 when any check failed, 3 when the base URL never
answered (UNCHECKED). A missing ffprobe prints ``UNCHECKED: bt709`` and is not a
failure. ``--json`` prints exactly one JSON object instead of the lines.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aws_lambda.video_builder.manifest import (  # noqa: E402
    IMMUTABLE,
    INDEX_KEY,
    INDEX_OPTIONAL,
    PRODUCTS,
    img1k_key,
    manifest_key,
    thumb_key,
    validate_fragment,
    video_key,
)

DEFAULT_BASE_URL = "https://the-sun-now.s3.us-east-2.amazonaws.com/"
TIMEOUT_S = 20
THUMB_PX = 512
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
BT709 = {"color_primaries": "bt709", "color_transfer": "bt709", "color_space": "bt709", "color_range": "tv"}
CACHE = {"fragment": "no-cache", "index": "public, max-age=300"}


class Unreachable(Exception):
    """No HTTP response at all (DNS, refused connection, timeout)."""


class Probe:
    def __init__(self, base_url, prefix=""):
        self.base = base_url if base_url.endswith("/") else base_url + "/"
        self.prefix = prefix
        self.answered = False

    def url(self, key):
        return self.base + self.prefix + key

    def request(self, key, method="GET", byte_range=None):
        """(status, headers, body) for one key; raises Unreachable on no response."""
        req = urllib.request.Request(self.url(key), method=method)
        if byte_range:
            req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                self.answered = True
                return resp.status, resp.headers, resp.read() if method == "GET" else b""
        except urllib.error.HTTPError as err:
            self.answered = True
            return err.code, err.headers, b""
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise Unreachable(f"{self.url(key)}: {getattr(err, 'reason', err)}") from err


def png_size(head):
    """(width, height) from the first 24 bytes of a PNG, or None if not a PNG."""
    if len(head) < 24 or not head.startswith(PNG_SIGNATURE) or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def iso_age_s(text, now):
    stamp = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int((now - stamp).total_seconds())


def check_type(problems, what, status, headers, content_type, cache_control=None):
    if status not in (200, 206):
        problems.append(f"{what} HTTP {status}")
        return False
    got = headers.get("Content-Type", "")
    if got != content_type:
        problems.append(f"{what} Content-Type {got!r}, expected {content_type!r}")
    if cache_control is not None and headers.get("Cache-Control") != cache_control:
        problems.append(f"{what} Cache-Control {headers.get('Cache-Control')!r}, expected {cache_control!r}")
    return True


def ffprobe_colour(ffprobe, url):
    """The four colour fields of the first video stream, as ffprobe reports them."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_primaries,color_transfer,color_space,color_range",
         "-of", "json", url],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return {"error": out.stderr.strip()[:200] or f"ffprobe exit {out.returncode}"}
    streams = json.loads(out.stdout or "{}").get("streams") or [{}]
    return {k: streams[0].get(k, "unknown") for k in BT709}


def check_product(probe, product, now, ffprobe):
    """(problems, fragment or None, detail) for one product."""
    pid = product["id"]
    problems = []
    status, headers, body = probe.request(manifest_key(pid))
    if not check_type(problems, "fragment", status, headers, "application/json", CACHE["fragment"]):
        return problems, None, ""
    try:
        frag = json.loads(body)
    except ValueError as err:
        return problems + [f"fragment is not JSON: {err}"], None, ""
    problems += [f"fragment {p}" for p in validate_fragment(frag)]
    if not isinstance(frag, dict):
        return problems, None, ""
    if frag.get("id") != pid:
        problems.append(f"fragment id {frag.get('id')!r}")
    if "kind" not in product:
        for field, expected in (("img1k", img1k_key(pid)), ("thumb", thumb_key(pid)), ("video", video_key(pid))):
            if frag.get(field) != expected:
                problems.append(f"fragment {field} {frag.get(field)!r}, expected {expected!r}")
    detail = ""
    try:
        detail = f"age {iso_age_s(frag.get('updated', ''), now)} s"
    except (TypeError, ValueError):
        problems.append(f"updated {frag.get('updated')!r} is not YYYY-MM-DDTHH:MM:SSZ")
    for field, square in (("img1k", None), ("thumb", THUMB_PX)):
        key = frag.get(field)
        if not isinstance(key, str):
            continue
        status, headers, head = probe.request(key, byte_range=(0, 23))
        if check_type(problems, field, status, headers, "image/png"):
            size = png_size(head)
            if size is None:
                problems.append(f"{field} is not a PNG")
            elif square and size != (square, square):
                problems.append(f"{field} is {size[0]}x{size[1]}, expected {square}x{square}")
    if isinstance(frag.get("video"), str):
        status, headers, _ = probe.request(frag["video"], method="HEAD")
        if check_type(problems, "video", status, headers, "video/mp4") and ffprobe:
            colour = ffprobe_colour(ffprobe, probe.url(frag["video"]))
            if colour != BT709:
                problems.append(f"video colour {colour}, expected {BT709}")
    return problems, frag, detail


def run(base_url=DEFAULT_BASE_URL, prefix="", ffprobe=None, now=None):
    """All checks as a list of {"name", "status", "detail"}; raises Unreachable."""
    now = now or datetime.now(timezone.utc)
    probe = Probe(base_url, prefix)
    checks = []
    fragments = []

    def add(name, problems, detail=""):
        checks.append({"name": name, "status": "FAIL" if problems else "PASS",
                       "detail": "; ".join(problems) if problems else detail})

    for product in PRODUCTS:
        try:
            problems, frag, detail = check_product(probe, product, now, ffprobe)
        except Unreachable:
            if not probe.answered:
                raise
            problems, frag, detail = ["no response"], None, ""
        add(product["id"], problems, detail)
        if frag is not None:
            fragments.append(frag)
    if not ffprobe:
        checks.append({"name": "bt709", "status": "UNCHECKED", "detail": "ffprobe not on PATH"})

    problems = []
    status, headers, body = probe.request(INDEX_KEY)
    if check_type(problems, "index", status, headers, "application/json", CACHE["index"]):
        try:
            index = json.loads(body)
            extra = sorted(set(index) - {"generated", "products"} - set(INDEX_OPTIONAL))
            if extra:
                problems.append(f"index has undocumented fields {extra}")
            for frag in index.get("products", []):
                problems += [f"index {frag.get('id')}: {p}" for p in validate_fragment(frag)]
        except (ValueError, AttributeError) as err:
            problems.append(f"index is not a JSON object: {err}")
    add("index", problems)

    problems = []
    versioned = next((f for f in fragments if f.get("video_v")), None)
    if versioned is None:
        problems.append("no fragment names a video_v")
    else:
        for field in ("video_v", "still_v"):
            if versioned.get(field):
                status, headers, _ = probe.request(versioned[field], method="HEAD")
                ctype = "video/mp4" if field == "video_v" else "image/png"
                check_type(problems, versioned[field], status, headers, ctype, IMMUTABLE)
    add("v/" + (f" ({versioned['id']})" if versioned else ""), problems)

    problems = []
    status, headers, body = probe.request("image_times.txt")
    if check_type(problems, "image_times.txt", status, headers, "text/plain"):
        try:
            datetime.fromisoformat(body.decode("utf-8").strip())
        except ValueError:
            problems.append(f"image_times.txt body {body[:40]!r} is not an ISO time")
    add("image_times.txt", problems)
    return checks


def main(argv=None, ffprobe=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default="", help="key prefix, for example staging/")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)
    ffprobe = shutil.which("ffprobe") if ffprobe is None else (ffprobe or None)
    try:
        checks = run(args.base_url, args.prefix, ffprobe)
    except Unreachable as err:
        checks = [{"name": "base-url", "status": "UNCHECKED", "detail": f"unreachable: {err}"}]
        code = 3
    else:
        code = 1 if any(c["status"] == "FAIL" for c in checks) else 0
    if args.json:
        print(json.dumps({"base_url": args.base_url, "prefix": args.prefix, "checks": checks,
                          "failed": sum(c["status"] == "FAIL" for c in checks),
                          "unchecked": sum(c["status"] == "UNCHECKED" for c in checks), "exit": code}))
    else:
        for c in checks:
            line = f"{c['status']}: {c['name']}"
            if c["detail"]:
                line += f": {c['detail']}" if c["status"] == "FAIL" else f" ({c['detail']})"
            print(line)
        failed = sum(c["status"] == "FAIL" for c in checks)
        unchecked = sum(c["status"] == "UNCHECKED" for c in checks)
        print(f"smoke_public: {len(checks)} checks, {failed} failed, {unchecked} unchecked, exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())

"""Manifest-driven fetch of the newest 1k still (SB-21), standard library only.

The client reads ``manifest/<id>.json`` (one small public JSON per product) and
downloads the ``img1k`` key it names. It never requests the bucket listing, which
returns only its first page and broke the old client when the bucket grew.
When a fragment is missing (404) the client falls back to the fixed key
``1k/rhef_<id>_1k.png``, still without a listing.

The image download is a conditional GET (If-None-Match with the cached ETag), so
an unchanged still costs one small 304 answer and no bytes.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

BASE_URL = "https://the-sun-now.s3.us-east-2.amazonaws.com/"
TIMEOUT_S = 30
USER_AGENT = "sunback-client"

# The product ids of aws_lambda/video_builder/manifest.py PRODUCTS (without "kind"
# entries), copied because the wheel does not ship aws_lambda. Bound to the
# canonical list by sunback/__tests__/test_client.py.
PRODUCT_IDS = ("rainbow", "171", "193", "211", "304", "335", "94", "131", "1600", "1700",
               "composite_uv", "dem")


def manifest_key(product_id):
    return f"manifest/{product_id}.json"


def img1k_key(product_id):
    return f"1k/rhef_{product_id}_1k.png"


def _url(base_url, key):
    return base_url.rstrip("/") + "/" + key


def _get(url, headers=None):
    req = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urlopen(req, timeout=TIMEOUT_S)


def read_fragment(product_id, base_url=BASE_URL):
    """The manifest fragment for one product, or None when it is not published (404)."""
    try:
        with _get(_url(base_url, manifest_key(product_id))) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            return None
        raise


def read_index(base_url=BASE_URL):
    """``manifest/index.json``: every product's fragment in one document."""
    with _get(_url(base_url, "manifest/index.json")) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_still(product_id, cache_dir, base_url=BASE_URL):
    """Download the newest 1k still of one product into ``cache_dir``.

    Returns ``(path, when, changed)``: the local PNG, the fragment's observation
    time (``obs_end``) or upload time (``updated``) or None, and whether a new
    image arrived. The PNG alternates between two file names (``.a.png`` and
    ``.b.png``) so a desktop that caches by path sees a new file each change,
    while the cache never holds more than two images per product.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frag = read_fragment(product_id, base_url)
    if frag is None:
        logger.warning("manifest/%s.json not found; using the fixed key %s", product_id, img1k_key(product_id))
        frag = {}
    key = frag.get("img1k") or img1k_key(product_id)
    when = frag.get("obs_end") or frag.get("updated")

    state_path = cache / f"{product_id}.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    current = cache / f"rhef_{product_id}_1k.{state.get('slot', 'b')}.png"
    headers = {}
    if state.get("etag") and current.exists():
        headers["If-None-Match"] = state["etag"]
    try:
        with _get(_url(base_url, key), headers) as resp:
            data = resp.read()
            etag = resp.headers.get("ETag", "")
    except HTTPError as e:
        if e.code == 304:
            return str(current), when, False
        raise
    slot = "b" if state.get("slot") == "a" else "a"
    target = cache / f"rhef_{product_id}_1k.{slot}.png"
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    state_path.write_text(json.dumps({"etag": etag, "slot": slot, "key": key, "when": when}), encoding="utf-8")
    return str(target), when, True


def parse_time(text):
    """'2026-10-13T03:43:03Z' -> an aware datetime, or None."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def describe(product_id, when, now=None):
    """'171 at 14:20 UTC (7 min ago)', or '171 (time unknown)'."""
    t = parse_time(when)
    if t is None:
        return f"{product_id} (time unknown)"
    now = now or datetime.now(timezone.utc)
    minutes = max(0, int((now - t).total_seconds() // 60))
    ago = f"{minutes} min ago" if minutes < 120 else f"{minutes // 60} h ago"
    return f"{product_id} at {t:%H:%M} UTC ({ago})"

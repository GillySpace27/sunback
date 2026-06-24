"""Per-product manifest fragments.

One fragment per product (``manifest/<id>.json``) so the 8 concurrent Lambda
invocations never write the same object — no read-modify-write race. The landing
page fetches all fragments (ids are fixed) and merges them client-side.

Product ids and S3 key conventions are the single source of truth shared by the
reducer (which writes ``1k/`` + ``thumb/``) and the Lambda (which writes ``video/``
and the fragment).
"""
import re

_1K_KEY_RE = re.compile(r"^1k/rhef_([A-Za-z0-9]+)_1k\.png$")

PRODUCTS = [
    {"id": "rainbow", "label": "Rainbow (RHEF composite)"},
    {"id": "171", "label": "AIA 171 Å"},
    {"id": "193", "label": "AIA 193 Å"},
    {"id": "211", "label": "AIA 211 Å"},
    {"id": "304", "label": "AIA 304 Å"},
    {"id": "335", "label": "AIA 335 Å"},
    {"id": "94", "label": "AIA 94 Å"},
    {"id": "131", "label": "AIA 131 Å"},
]

_LABELS = {p["id"]: p["label"] for p in PRODUCTS}


def img1k_key(product_id):
    return f"1k/rhef_{product_id}_1k.png"


def thumb_key(product_id):
    return f"thumb/rhef_{product_id}_thumb.png"


def video_key(product_id):
    return f"video/rhef_{product_id}_1k.mp4"


def manifest_key(product_id):
    return f"manifest/{product_id}.json"


def product_from_1k_key(key):
    """Extract the product id from a triggering ``1k/`` key, or None if not one."""
    m = _1K_KEY_RE.match(key)
    return m.group(1) if m else None


def build_manifest_fragment(product_id, updated, frame_count, integration):
    """Build the JSON fragment for one product. Raises KeyError on unknown id."""
    label = _LABELS[product_id]
    return {
        "id": product_id,
        "label": label,
        "thumb": thumb_key(product_id),
        "img1k": img1k_key(product_id),
        "video": video_key(product_id),
        "updated": updated,
        "frame_count": frame_count,
        "integration": integration,
    }

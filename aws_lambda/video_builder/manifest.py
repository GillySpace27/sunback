"""Per-product manifest fragments.

One fragment per product (``manifest/<id>.json``) so the 8 concurrent Lambda
invocations never write the same object — no read-modify-write race. The landing
page fetches all fragments (ids are fixed) and merges them client-side.

Product ids and S3 key conventions are the single source of truth shared by the
reducer (which writes ``1k/`` + ``thumb/``) and the Lambda (which writes ``video/``
and the fragment).
"""
import re

_1K_KEY_RE = re.compile(r"^1k/rhef_([A-Za-z0-9_]+)_1k\.png$")

PRODUCTS = [
    {"id": "rainbow", "label": "Rainbow (RHEF composite)"},
    {"id": "171", "label": "AIA 171 Å"},
    {"id": "193", "label": "AIA 193 Å"},
    {"id": "211", "label": "AIA 211 Å"},
    {"id": "304", "label": "AIA 304 Å"},
    {"id": "335", "label": "AIA 335 Å"},
    {"id": "94", "label": "AIA 94 Å"},
    {"id": "131", "label": "AIA 131 Å"},
    {"id": "1600", "label": "AIA 1600 Å (UV)"},
    {"id": "1700", "label": "AIA 1700 Å (UV)"},
    {"id": "composite_uv", "label": "Composite (1700/1600/304)"},
    {"id": "dem", "label": "Temperature map (DEM)"},
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


# Versioned copies live under v/ and are never rewritten once written, so
# anything in front of the bucket may cache them for a year. The fixed keys
# above keep their meaning for the landing page and for older Heliograph builds.
INDEX_KEY = "manifest/index.json"
IMMUTABLE = "public, max-age=31536000, immutable"
_STAMP_RE = re.compile(r"(\d{8}T\d{6})")


def compact_stamp(text):
    """The YYYYMMDDTHHMMSS stamp inside a frame key or an ISO timestamp."""
    m = _STAMP_RE.search(re.sub(r"[-:]", "", text))
    if not m:
        raise ValueError(f"no timestamp in {text!r}")
    return m.group(1)


def versioned_video_key(product_id, through):
    """``through`` is the compact stamp of the newest frame in the video."""
    return f"v/{product_id}/{through}.mp4"


def versioned_still_key(product_id, stamp):
    return f"v/{product_id}/{stamp}.png"


def product_from_1k_key(key):
    """Extract the product id from a triggering ``1k/`` key, or None if not one."""
    m = _1K_KEY_RE.match(key)
    return m.group(1) if m else None


def build_manifest_fragment(product_id, updated, frame_count, integration,
                            video_v=None, still_v=None, through=None):
    """Build the JSON fragment for one product. Raises KeyError on unknown id.

    ``video_v``/``still_v`` name the immutable copies and ``through`` is the time
    of the newest frame in the video, which can trail ``updated`` by up to the
    encode throttle. They appear only when known, so a reader that predates them
    sees exactly the fragment it always did.
    """
    label = _LABELS[product_id]
    frag = {
        "id": product_id,
        "label": label,
        "thumb": thumb_key(product_id),
        "img1k": img1k_key(product_id),
        "video": video_key(product_id),
        "updated": updated,
        "frame_count": frame_count,
        "integration": integration,
    }
    for key, value in (("video_v", video_v), ("still_v", still_v), ("through", through)):
        if value is not None:
            frag[key] = value
    return frag


def build_index(fragments, generated):
    """Every product's fragment in one document, in PRODUCTS order.

    A reader then learns whether anything moved with one request instead of
    twelve. Unknown ids are dropped rather than trusted; products that have not
    built yet are simply absent.
    """
    by_id = {f.get("id"): f for f in fragments}
    return {"generated": generated,
            "products": [by_id[p["id"]] for p in PRODUCTS if p["id"] in by_id]}

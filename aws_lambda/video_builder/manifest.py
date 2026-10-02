"""Per-product manifest fragments.

One fragment per product (``manifest/<id>.json``), so the up to 12 concurrent
Lambda invocations of one reducer run (one per ``1k/`` still) never write the
same fragment. ``manifest/index.json`` is the one shared object: every
invocation rebuilds it from all fragments (``handler._write_index``) and the
next trigger repairs a lost write. The landing page fetches the fragments (ids
are fixed); Heliogram 0.7+ and the R2 mirror read the index. The public schema
is written down in CONTRACT.md beside this file.

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

# The nine AIA channels served as single-wavelength cards (SB-6); the other ids
# in PRODUCTS are composites or derived products. Bound to
# serve_keys.SERVED_CHANNELS by sunback/__tests__/test_product_catalog.py.
AIA_IDS = ("171", "193", "211", "304", "335", "94", "131", "1600", "1700")


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


# Public schema (SB-6). CONTRACT.md beside this file documents every field below,
# and sunback/__tests__/test_manifest_contract.py binds code, fixtures and doc.
# Additive only: a new field goes into FRAGMENT_OPTIONAL (or INDEX_OPTIONAL) and
# into CONTRACT.md in the same commit. Removing or renaming a field needs Gilly
# and a Heliogram release first.
FRAGMENT_REQUIRED: dict[str, type] = {
    "id": str, "label": str, "thumb": str, "img1k": str, "video": str,
    "updated": str, "frame_count": int, "integration": dict,
}
FRAGMENT_OPTIONAL: dict[str, type] = {"video_v": str, "still_v": str, "through": str}
INDEX_OPTIONAL: dict[str, type] = {}
_INTEGRATION_FIELDS = {"frames": int, "method": str}


def _type_problem(name, value, expected):
    if expected is int and isinstance(value, bool):
        return f"{name}: expected int, got bool"
    if not isinstance(value, expected):
        return f"{name}: expected {expected.__name__}, got {type(value).__name__}"
    return None


def validate_fragment(frag):
    """Problems with one manifest fragment, one sentence each; [] means valid.

    Unknown keys are problems, so no field reaches readers without a row in
    CONTRACT.md. Stdlib only: the Lambda zip and the smoke check both use it.
    """
    if not isinstance(frag, dict):
        return [f"fragment: expected dict, got {type(frag).__name__}"]
    problems = [f"missing required key {name!r}" for name in FRAGMENT_REQUIRED if name not in frag]
    for name, value in frag.items():
        expected = FRAGMENT_REQUIRED.get(name) or FRAGMENT_OPTIONAL.get(name)
        if expected is None:
            problems.append(f"unknown key {name!r}")
            continue
        problem = _type_problem(name, value, expected)
        if problem:
            problems.append(problem)
    if isinstance(frag.get("id"), str) and frag["id"] not in _LABELS:
        problems.append(f"id: unknown product id {frag['id']!r}")
    integration = frag.get("integration")
    if isinstance(integration, dict):
        for name, expected in _INTEGRATION_FIELDS.items():
            if name not in integration:
                problems.append(f"integration: missing {name!r}")
                continue
            problem = _type_problem(f"integration.{name}", integration[name], expected)
            if problem:
                problems.append(problem)
    return problems

"""Map reducer-emitted PNG filenames to served product ids and S3 keys.

Ground truth from the production bucket (last hourly run):
  renders/DrGilly_<wave>_ups(rhef).png   -- one per wavelength
  renders/BGR_0171_0193_0211_ups(rhef).png   -- EUV coronal composite  (the "rainbow")
  renders/BGR_1700_1600_0304_ups(rhef).png   -- UV composite           ("composite_uv")
  renders/C_isothermal.png                    -- DEM temperature map    ("dem")

Twelve page cards are served, the ids of manifest.PRODUCTS: the rainbow
composite, the nine AIA singles (seven EUV plus 1600 and 1700), the UV
composite and the DEM map. Videos are not uploaded from here (the Lambda builds
them), except the DEM temperature scan (AwsPutter).

Key conventions mirror aws_lambda/video_builder/manifest.py.
"""

import os
import re

# reducer wave code -> page product id (EUV leading zeros stripped; UV kept as-is)
SERVED_CHANNELS = {
    "0171": "171",
    "0193": "193",
    "0211": "211",
    "0304": "304",
    "0335": "335",
    "0094": "94",
    "0131": "131",
    "1600": "1600",
    "1700": "1700",
}

# Which composite is the headline "rainbow" card. To use the 1700/1600/0304 blend
# as the headline instead, swap these two source strings.
RAINBOW_SOURCE = "BGR_0171_0193_0211"  # -> "rainbow"
UV_COMPOSITE_SOURCE = "BGR_1700_1600_0304"  # -> "composite_uv"
DEM_SOURCE = "C_isothermal"  # -> "dem" (isothermal temperature map)

_DRGILLY_RE = re.compile(r"DrGilly_(\d{4})_")

# SB-16: the log-stretched "before" view of a single AIA wave (processor/raw_view.py)
RAW_SUFFIX = "_raw.png"
_RAW_RE = re.compile(r"DrGilly_(\d{4})_raw\.png$")


def serve_id_for_local_png(path):
    """Return the served product id for a local PNG, or None if it isn't served."""
    name = os.path.basename(path)
    if name.endswith(RAW_SUFFIX):
        return None  # SB-16: a raw view is never uploaded as the RHEF still
    if name.startswith(RAINBOW_SOURCE):
        return "rainbow"
    if name.startswith(UV_COMPOSITE_SOURCE):
        return "composite_uv"
    if name.startswith(DEM_SOURCE):
        return "dem"
    m = _DRGILLY_RE.match(name)
    if m and m.group(1) in SERVED_CHANNELS:
        return SERVED_CHANNELS[m.group(1)]
    return None


def s3_img_key(product_id):
    return f"1k/rhef_{product_id}_1k.png"


def s3_thumb_key(product_id):
    return f"thumb/rhef_{product_id}_thumb.png"


def s3_meta_key(product_id):
    """Provenance sidecar for the newest still (SB-9); schema.org ImageObject JSON."""
    return f"meta/rhef_{product_id}.json"


def overlay_grid_key():
    """Grid and limb overlay for the AIA stills (SB-15). Not under 1k/, so the video Lambda never fires on it."""
    return "overlay/grid_1k.png"


def raw_id_for_local_png(path):
    """Product id for a local DrGilly_<wave>_raw.png, or None (composites have no raw view)."""
    m = _RAW_RE.match(os.path.basename(path))
    return SERVED_CHANNELS.get(m.group(1)) if m else None


def s3_raw_key(product_id):
    """Raw (log-stretched) view of an AIA single (SB-16). Not under 1k/, so the Lambda never fires on it."""
    return f"raw/rhef_{product_id}_raw_1k.png"

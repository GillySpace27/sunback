"""Map reducer-emitted PNG filenames to served product ids and S3 keys.

Ground truth from the production bucket (last hourly run):
  renders/DrGilly_<wave>_ups(rhef).png   -- one per wavelength
  renders/BGR_0171_0193_0211_ups(rhef).png   -- EUV coronal composite  (the "rainbow")
  renders/BGR_1700_1600_0304_ups(rhef).png   -- alternate composite     (not served)
  renders/C_isothermal.png                    -- DEM product            (not served)

Only the 8 page cards are served: the rainbow composite + 7 EUV singles. The
UV channels (1600/1700) are fetched only to build the composite, and the DEM /
alternate-composite / video outputs are skipped.

Key conventions mirror aws_lambda/video_builder/manifest.py.
"""
import os
import re

# zero-padded reducer wave code -> page product id (leading zeros stripped)
SERVED_EUV = {
    "0171": "171", "0193": "193", "0211": "211", "0304": "304",
    "0335": "335", "0094": "94", "0131": "131",
}

# Which composite is the headline "rainbow" card. The EUV coronal composite is the
# default; to use the 1700/1600/0304 blend instead, set this to "BGR_1700_1600_0304".
RAINBOW_SOURCE = "BGR_0171_0193_0211"

_DRGILLY_RE = re.compile(r"DrGilly_(\d{4})_")


def serve_id_for_local_png(path):
    """Return the served product id for a local PNG, or None if it isn't served."""
    name = os.path.basename(path)
    if name.startswith(RAINBOW_SOURCE):
        return "rainbow"
    m = _DRGILLY_RE.match(name)
    if m and m.group(1) in SERVED_EUV:
        return SERVED_EUV[m.group(1)]
    return None


def s3_img_key(product_id):
    return f"1k/rhef_{product_id}_1k.png"


def s3_thumb_key(product_id):
    return f"thumb/rhef_{product_id}_thumb.png"

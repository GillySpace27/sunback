"""Pure helpers for the 48h sliding-window video frame queue.

Frame objects live in S3 under ``frames/<product>/<YYYYMMDDTHHMMSS>_1k.png``.
These functions decide ordering and pruning; all S3 I/O lives in the handler.
"""
import re

# frames/<product>/<YYYYMMDDTHHMMSS>_1k.png
_FRAME_KEY_RE = re.compile(r"/(\d{8}T\d{6})_1k\.png$")


def frame_key_for(product, iso_timestamp):
    """Build the S3 frame key for a product + ISO-8601 timestamp.

    ``"2026-06-24T20:03:00Z"`` -> ``"frames/0171/20260624T200300_1k.png"``.
    """
    compact = re.sub(r"[-:]", "", iso_timestamp).replace("Z", "")
    compact = compact.split(".")[0]  # drop fractional seconds if present
    return f"frames/{product}/{compact}_1k.png"


def _stamp(key):
    m = _FRAME_KEY_RE.search(key)
    return m.group(1) if m else None


def sorted_newest_last(keys):
    """Frame keys sorted oldest->newest; unparseable keys are dropped."""
    parseable = [k for k in keys if _stamp(k) is not None]
    return sorted(parseable, key=_stamp)


def select_frames_to_delete(keys, max_frames):
    """Return the frame keys to delete so at most ``max_frames`` newest remain."""
    ordered = sorted_newest_last(keys)
    if len(ordered) <= max_frames:
        return []
    return ordered[: len(ordered) - max_frames]

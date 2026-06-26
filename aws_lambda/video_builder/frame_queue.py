"""Pure helpers for the 48h sliding-window video frame queue.

Frame objects live in S3 under ``frames/<product>/<YYYYMMDDTHHMMSS>_1k.png``.
These functions decide ordering and pruning; all S3 I/O lives in the handler.
"""
import re
from datetime import datetime, timedelta

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


def _stamp_dt(key):
    return datetime.strptime(_stamp(key), "%Y%m%dT%H%M%S")


def build_grid_sequence(frame_keys, cadence_s=1200, max_slots=144):
    """Snap frames onto a uniform time grid for a steady-cadence video.

    Builds slots every ``cadence_s`` seconds ending at the newest frame (at most
    ``max_slots`` of them), and assigns each slot the temporally-nearest frame.
    Empty slots therefore reuse a neighbouring frame — i.e. the previous frame is
    "held" through a gap — so playback advances at a constant rate regardless of
    when the reducer actually ran. Returns the per-slot key list, oldest→newest.
    On ties (a slot equidistant from two frames) the earlier frame wins (hold).
    """
    items = sorted((_stamp_dt(k), k) for k in frame_keys if _stamp(k) is not None)
    if not items:
        return []
    latest = items[-1][0]
    span = (latest - items[0][0]).total_seconds()
    n = min(max_slots, int(span // cadence_s) + 1)
    slots = [latest - timedelta(seconds=cadence_s * (n - 1 - i)) for i in range(n)]
    seq = []
    for t in slots:
        # nearest frame to this slot; min() keeps the first (earliest) on a tie
        seq.append(min(items, key=lambda it: abs((it[0] - t).total_seconds()))[1])
    return seq

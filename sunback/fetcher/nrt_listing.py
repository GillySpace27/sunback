"""Parse JSOC synoptic-NRT directory listings and pick recent frames per wavelength.

The NRT synoptic dirs (e.g. ``.../synoptic/nrt/YYYY/MM/DD/H2000/``) hold filenames
like ``AIA20260624_200300_0171.fits`` at ~3-min cadence. Unlike ``mostrecent/`` they
expose *many* recent frames, which is what makes time integration possible.
"""
import re
from datetime import datetime, timedelta

# AIA<YYYYMMDD>_<HHMMSS>_<wave>.fits
_FRAME_RE = re.compile(r"^AIA(\d{8})_(\d{6})_(\d{4})\.fits$")


def nrt_hour_dirs(when, base_url, lookback_hours=1):
    """Build synoptic-NRT hour-bucket directory URLs, newest first.

    The archive is laid out as ``<base>/YYYY/MM/DD/H<HH>00/``. We return the
    bucket for ``when`` plus ``lookback_hours`` earlier buckets (rolling across
    day boundaries) so frame selection has enough recent frames near the hour edge.
    """
    base = base_url.rstrip("/") + "/"
    dirs = []
    for delta in range(lookback_hours + 1):
        t = when - timedelta(hours=delta)
        dirs.append(base + t.strftime("%Y/%m/%d/H%H00/"))
    return dirs


def parse_frame_time(filename):
    """Return a ``datetime`` for an NRT frame filename, or ``None`` if malformed."""
    m = _FRAME_RE.match(filename.strip())
    if not m:
        return None
    date_str, time_str, _wave = m.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")


def _parse(filename):
    """Return (datetime, wave) or None for a frame filename."""
    m = _FRAME_RE.match(filename.strip())
    if not m:
        return None
    date_str, time_str, wave = m.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S"), wave


def select_recent_frames(filenames, waves, n):
    """Pick the ``n`` most-recent frames for each requested wavelength.

    Args:
        filenames: iterable of NRT frame filenames (and any junk, which is ignored).
        waves: wavelengths to keep, as zero-padded strings (e.g. ``"0171"``).
        n: max frames to return per wavelength.

    Returns:
        dict ``{wave: [filenames]}`` sorted oldest→newest (natural integration order),
        each list at most ``n`` long. Wavelengths with no frames are omitted.
    """
    wanted = set(waves)
    by_wave = {}
    for name in filenames:
        parsed = _parse(name)
        if parsed is None:
            continue
        when, wave = parsed
        if wave in wanted:
            by_wave.setdefault(wave, []).append((when, name))

    result = {}
    for wave, items in by_wave.items():
        items.sort(key=lambda pair: pair[0])  # oldest -> newest
        result[wave] = [name for _when, name in items[-n:]]
    return result

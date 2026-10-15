"""Daily A/B pair (SB-20): today's 171 under the current and the candidate integration.

    python -m sunback.run.run_ab_pair [--frames-dir DIR] [--out DIR] [--no-upload]

Both halves come from one NRT listing, so they end on the same newest frame:
"current" integrates the newest CURRENT[0] frames with CURRENT[1] (production
today: 5, median) and "candidate" the newest CANDIDATE[0] with CANDIDATE[1]
(15, sigclip). When the listing holds fewer frames than the candidate asks
for, the candidate uses what there is and the JSON records the real count;
the window is never widened silently.

Each half is rendered through RHEF exactly as the Filter Lab's rhef tile
(run_filter_lab._rhef) and uploaded under SUNBACK_PREFIX to
lab/ab_171_current.png and lab/ab_171_candidate.png, plus lab/ab_171.json.
The JSON's "std" values are computed here: numpy std of the integrated data
(before RHEF) inside "box", a fixed square at the image centre that is chosen,
not checked to be quiet Sun on a given day. Both PNGs are visualizations, not
calibrated radiances. Production changes N or method only by setting, after
Gilly's yes.
"""
import argparse
import glob
import json
import logging
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

WAVE = "0171"
CURRENT = (5, "median")       # production default (NRTFitsFetcher, NrtSettings)
CANDIDATE = (15, "sigclip")   # the owner wish: N 10 to 15, sigma-clipped mean
BOX_FRACTION = 8              # box side = image side / 8, centred (a chosen box)
JSON_KEY = "lab/ab_171.json"
NOTE = ("Visualization, not a calibrated radiance. RHEF: Gilly & Cranmer 2025. "
        "std: computed, numpy std of the integrated data (before RHEF) inside box.")


def png_key(half):
    return f"lab/ab_171_{half}.png"


def centre_box(shape, fraction=BOX_FRACTION):
    """[x0, y0, x1, y1] of the centred square whose side is the shorter image side / fraction."""
    ny, nx = shape
    half = max(1, min(nx, ny) // fraction // 2)
    cx, cy = nx // 2, ny // 2
    return [cx - half, cy - half, cx + half, cy + half]


def box_std(data, box):
    x0, y0, x1, y1 = box
    return float(np.nanstd(np.asarray(data, dtype=np.float64)[y0:y1, x0:x1]))


def list_171_frames(now=None):
    """(filename -> url) of the newest NRT 171 frames, from the reducer's own listing helpers."""
    from sunback.fetcher.NRTFitsFetcher import LOOKBACK_HOURS, NRTFitsFetcher
    from sunback.fetcher.nrt_listing import nrt_hour_dirs, select_recent_frames

    now = now or datetime.now(timezone.utc)
    listing = {}
    for dir_url in nrt_hour_dirs(now, NRTFitsFetcher.base_url, LOOKBACK_HOURS):
        for name in NRTFitsFetcher._list_dir_filenames(None, dir_url):
            listing.setdefault(name, dir_url + name)
    names = select_recent_frames(list(listing), [WAVE], CANDIDATE[0]).get(WAVE, [])
    return {name: listing[name] for name in names}


def download_frames(urls, frames_dir):
    """Download each url into frames_dir; returns the local paths oldest to newest."""
    from sunback.fetcher.NRTFitsFetcher import _HEADERS

    os.makedirs(frames_dir, exist_ok=True)
    paths = []
    for name in sorted(urls):
        local = os.path.join(frames_dir, name)
        req = urllib.request.Request(urls[name], headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp, open(local, "wb") as fh:
            fh.write(resp.read())
        paths.append(local)
    return paths


def rhef_render(fits_path, out_png):
    """RHEF tile of one integrated FITS, as the Filter Lab renders its rhef tile."""
    import sunpy.map

    from sunback.processor.raw_view import colour_png, first_image
    from sunback.run.run_filter_lab import _rhef, display_unit

    data, header = first_image(fits_path)
    out, _, _ = _rhef(sunpy.map.Map(data, header))
    return colour_png(display_unit(out), int(WAVE), out_png)


def ab_pair(frame_paths, out_dir, render=rhef_render, now=None):
    """Integrate, render and measure both halves; returns (doc, [(path, key, content type)])."""
    from sunback.fetcher.nrt_integrate import write_integrated_synoptic
    from sunback.processor.raw_view import first_image

    frame_paths = sorted(frame_paths, key=os.path.basename)  # NRT names sort oldest to newest
    if not frame_paths:
        raise ValueError("no 171 frames")
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "generated": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"wave": WAVE.lstrip("0"), "frames_available": len(frame_paths),
                   "newest": os.path.basename(frame_paths[-1])},
        "box": None,
        "note": NOTE,
    }
    files = []
    for half, (n, method) in (("current", CURRENT), ("candidate", CANDIDATE)):
        used = frame_paths[-n:]
        fits_path = write_integrated_synoptic(used, os.path.join(out_dir, f"ab_171_{half}.fits"), method=method)
        data, header = first_image(fits_path)
        if doc["box"] is None:
            doc["box"] = centre_box(data.shape)
        png = render(fits_path, os.path.join(out_dir, f"ab_171_{half}.png"))
        doc[half] = {"frames": len(used), "frames_asked": n, "method": method,
                     "oldest": os.path.basename(used[0]), "t_rec": str(header.get("T_REC", "")),
                     "std": box_std(data, doc["box"]), "key": png_key(half)}
        files.append((png, png_key(half), "image/png"))
    doc_path = os.path.join(out_dir, "ab_171.json")
    with open(doc_path, "w", encoding="utf-8") as fp:
        json.dump(doc, fp, indent=2)
    files.append((doc_path, JSON_KEY, "application/json"))
    return doc, files


def upload(files, settings=None):
    from sunback.putter.AwsPutter import upload_public
    from sunback.settings import NrtSettings

    settings = settings if settings is not None else NrtSettings.from_env()
    for path, key, content_type in files:
        upload_public(path, key, content_type, cache_control="no-cache", settings=settings)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render today's 171 A/B integration pair")
    ap.add_argument("--frames-dir", help="use the NRT 171 frames already here instead of downloading")
    ap.add_argument("--out", default="sunback_data/lab_ab", help="local output directory")
    ap.add_argument("--no-upload", action="store_true", help="write locally only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.frames_dir:
        frames = glob.glob(os.path.join(args.frames_dir, f"AIA*_*_{WAVE}.fits"))
    else:
        frames = download_frames(list_171_frames(), tempfile.mkdtemp(prefix="sunback_ab_"))
    if not frames:
        print("A/B pair: no 171 frames; nothing written")
        return 1
    doc, files = ab_pair(frames, args.out)
    if not args.no_upload:
        upload(files)
    print(f"A/B pair: current {doc['current']['frames']}x {doc['current']['method']} std {doc['current']['std']:.3g}; "
          f"candidate {doc['candidate']['frames']}x {doc['candidate']['method']} std {doc['candidate']['std']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

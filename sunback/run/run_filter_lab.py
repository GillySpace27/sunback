"""Daily Filter Lab (SB-16): today's integrated 171 through six filters, one tile each.

    python -m sunback.run.run_filter_lab [--fits PATH | --fits-dir DIR] [--out DIR] [--no-upload]

Each filter calls sunkit-image the way the matching class in
sunback/processor/SunPyProcessor.py does (named in the tile's "function"),
on the first 2-D image HDU of AIAsynoptic0171.fits (the integrated frame
before any filter). A filter that raises is listed under "skipped" with its
error and gets no tile; nothing is guessed. Tiles are written in the AIA
colour table, flipped like the served stills, and uploaded under
SUNBACK_PREFIX to lab/filter_<name>_171.png plus lab/filter_lab.json.
Every tile is a visualization, not a calibrated radiance.
"""
import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

WAVE = "0171"
LAB_JSON_KEY = "lab/filter_lab.json"
DISPLAY_PCT = (0.5, 99.5)  # chosen display percentiles for a tile, not measured values
NOTE = "Visualization, not a calibrated radiance. RHEF: Gilly & Cranmer 2025; other filters: see the sunkit-image documentation."


def lab_key(name):
    return f"lab/filter_{name}_171.png"


def _upsilon_171():
    """(aL, aH) for 171 from the reducer's own table (Processor.init_upsilon_array)."""
    from sunback.processor.Processor import Processor

    table = SimpleNamespace()
    Processor.init_upsilon_array(table)
    row = table.dictdict[WAVE]
    return [row["aL"], row["aH"]]


def _rhef(smap):
    import astropy.units as u
    import sunkit_image.radial as radial

    # RHEFProcessor.do_work with the reducer's settings: do_vignette True, upsilon not applied together
    params = {"upsilon": None, "vignette": "1.51 R_sun", "method": "scipy"}
    out = radial.rhef(smap, upsilon=None, vignette=1.51 * u.R_sun, method="scipy")
    return out.data, "sunkit_image.radial.rhef", params


def _upsilon(smap):
    import astropy.units as u
    import sunkit_image.radial as radial
    import sunpy.map

    # UpsilonProcessor.do_work on the RHEF frame: the served ups(rhef) look
    first, _, _ = _rhef(smap)
    ups = _upsilon_171()
    out = radial.rhef(sunpy.map.Map(first, smap.meta), upsilon=ups, vignette=None, method="scipy",
                      application_radius=0 * u.R_sun)
    return out.data, "sunkit_image.radial.rhef (twice)", {"upsilon": ups, "application_radius": "0 R_sun"}


def _nrgf(smap):
    import astropy.units as u
    import sunkit_image.radial as radial
    from sunkit_image.utils import equally_spaced_bins

    # NRGFProcessor.do_work
    edges = equally_spaced_bins(inner_value=0.0, nbins=300) * u.R_sun
    out = radial.nrgf(smap, radial_bin_edges=edges, application_radius=0.0 * u.R_sun)
    return out.data, "sunkit_image.radial.nrgf", {"nbins": 300, "inner_value": 0.0, "application_radius": "0 R_sun"}


def _fnrgf(smap):
    import sunkit_image.radial as radial

    # FNRGFProcessor uses order 20; its other arguments do not fit the installed signature,
    # so the installed defaults are used for them (listed under skipped if this raises).
    out = radial.fnrgf(smap, order=20)
    return out.data, "sunkit_image.radial.fnrgf", {"order": 20}


def _msgn(smap):
    import sunkit_image.enhance as enhance

    # MSGNProcessor.do_work: NaN set to -1, defaults otherwise
    data = np.where(np.isnan(smap.data), -1.0, smap.data)
    return enhance.mgn(data), "sunkit_image.enhance.mgn", {}


def _wow(smap):
    import sunkit_image.enhance as enhance

    # WOWProcessor.do_work: NaN set to -1, defaults otherwise
    data = np.where(np.isnan(smap.data), -1.0, smap.data)
    return enhance.wow(data), "sunkit_image.enhance.wow", {}


FILTERS = [("rhef", _rhef), ("upsilon", _upsilon), ("nrgf", _nrgf), ("fnrgf", _fnrgf),
           ("msgn", _msgn), ("wow", _wow)]


def display_unit(data, pct=DISPLAY_PCT):
    """Linear 0..1 between two percentiles of the finite pixels (NaN shows as 0)."""
    data = np.asarray(data, dtype=np.float64)
    good = data[np.isfinite(data)]
    if good.size == 0:
        raise ValueError("no finite pixels")
    lo, hi = np.percentile(good, pct)
    if hi <= lo:
        raise ValueError("flat output")
    return np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def find_171(fits_dir):
    hits = sorted(glob.glob(os.path.join(fits_dir, "**", f"AIAsynoptic{WAVE}.fits"), recursive=True),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def _version(dist):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


def run_filter_lab(fits_path, out_dir, upload=True, settings=None, filters=None):
    """Render every tile and the JSON; upload them unless upload is False. Returns the JSON document."""
    import sunpy.map

    from sunback.processor.raw_view import colour_png, first_image

    data, header = first_image(fits_path)
    smap = sunpy.map.Map(data, header)
    os.makedirs(out_dir, exist_ok=True)
    tiles, skipped, files = [], [], []
    for name, func in (filters or FILTERS):
        try:
            out, function, params = func(smap)
            png = colour_png(display_unit(out), int(WAVE), os.path.join(out_dir, f"filter_{name}_171.png"))
        except Exception as exc:  # a broken filter is listed, never fatal
            skipped.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            logger.warning(f"filter lab: {name} skipped ({type(exc).__name__}: {exc})")
            continue
        tiles.append({"name": name, "key": lab_key(name), "function": function, "params": params})
        files.append((png, lab_key(name), "image/png"))
    doc = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"wave": WAVE.lstrip("0"), "t_rec": str(header.get("T_REC", ""))},
        "sunkit_image_version": _version("sunkit-image"),
        "display": f"linear between the {DISPLAY_PCT[0]} and {DISPLAY_PCT[1]} percentiles, AIA colour table",
        "note": NOTE,
        "tiles": tiles,
        "skipped": skipped,
    }
    doc_path = os.path.join(out_dir, "filter_lab.json")
    with open(doc_path, "w", encoding="utf-8") as fp:
        json.dump(doc, fp, indent=2)
    files.append((doc_path, LAB_JSON_KEY, "application/json"))
    if upload:
        from sunback.putter.AwsPutter import upload_public
        from sunback.settings import NrtSettings

        settings = settings if settings is not None else NrtSettings.from_env()
        for path, key, content_type in files:
            upload_public(path, key, content_type, cache_control="no-cache", settings=settings)
    logger.info(f"filter lab: {len(tiles)} tiles, {len(skipped)} skipped")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the daily Filter Lab from today's integrated 171 FITS")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--fits", help="path to AIAsynoptic0171.fits")
    src.add_argument("--fits-dir", default="sunback_data", help="search here for the newest AIAsynoptic0171.fits")
    ap.add_argument("--out", default="sunback_data/lab", help="local output directory")
    ap.add_argument("--no-upload", action="store_true", help="write locally only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    fits_path = args.fits or find_171(args.fits_dir)
    if not fits_path or not os.path.exists(fits_path):
        print(f"filter lab: no AIAsynoptic{WAVE}.fits found; nothing written")
        return 1
    doc = run_filter_lab(fits_path, args.out, upload=not args.no_upload)
    print(f"filter lab: {len(doc['tiles'])} tiles, {len(doc['skipped'])} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

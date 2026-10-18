"""Stonyhurst grid and solar limb overlay for the served AIA stills (SB-15).

The overlay is drawn from the integrated synoptic FITS's own WCS, on that
FITS's pixel grid, so it lines up with the 1k still ImageProcessorCV renders
from the same file: that still is the FITS array flipped top to bottom
(``np.flipud`` in ``plot_aia_changed``), which is what matplotlib's
``origin="lower"`` display writes too. Nothing is cropped or scaled on that
path, so one overlay pixel is one still pixel.

The output is a transparent RGBA PNG: lines only, fully transparent elsewhere.
It is a drawing aid, not a measurement.
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

OVERLAY_KEY = "overlay/grid_1k.png"
OVERLAY_MANIFEST_KEY = "manifest/overlay.json"
OVERLAY_SOURCE_ID = "171"
OVERLAY_SOURCE_WAVE = "0171"
LINE_RGBA = (1.0, 1.0, 1.0, 0.85)
GRID_SPACING_DEG = 15


def _image_header(fits_path):
    """Header of the first 2-D image HDU (a JSOC synoptic file has an empty primary)."""
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.header.get("NAXIS", 0) == 2:
                return hdu.header.copy()
    raise ValueError(f"no 2-D image HDU in {fits_path}")


def header_map(header):
    """A sunpy map with the header's WCS and observer and an all-zero image.

    Only the geometry is needed, so the pixel data are never read.
    """
    import sunpy.map

    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    return sunpy.map.Map(np.zeros(shape, dtype=np.float32), header)


def render_grid_overlay(fits_path_or_map, out_png, size=1024, grid=True, limb=True):
    """Draw the grid and limb on a transparent ``size`` x ``size`` RGBA PNG.

    Returns True when the PNG was written, False when the map is not ``size``
    by ``size`` pixels (an overlay on another grid would not line up with the
    still) or the drawing failed. Never raises.
    """
    try:
        import astropy.units as u
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if isinstance(fits_path_or_map, (str, os.PathLike)):
            amap = header_map(_image_header(fits_path_or_map))
        else:
            amap = fits_path_or_map
        ny, nx = amap.data.shape
        if (ny, nx) != (size, size):
            logger.warning(f"grid overlay: map is {nx}x{ny} px, not {size}x{size}; no overlay written")
            return False

        dpi = 100
        fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
        FigureCanvasAgg(fig)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1], projection=amap.wcs)
        ax.patch.set_alpha(0.0)
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.coords.frame.set_linewidth(0)
        for coord in ax.coords:
            coord.set_ticks_visible(False)
            coord.set_ticklabel_visible(False)
            coord.set_axislabel("")
            coord.grid(False)
        if grid:
            overlay = amap.draw_grid(axes=ax, grid_spacing=GRID_SPACING_DEG * u.deg, annotate=False,
                                     color=LINE_RGBA, linewidth=0.6)
            for coord in overlay:
                coord.set_ticks_visible(False)
                coord.set_ticklabel_visible(False)
                coord.set_axislabel("")
        if limb:
            amap.draw_limb(axes=ax, color=LINE_RGBA, linewidth=1.2)
        fig.savefig(out_png, dpi=dpi, transparent=True, format="png")
        return True
    except Exception as exc:  # the overlay is optional: it never stops the stills
        logger.error(f"grid overlay: render failed ({type(exc).__name__}: {exc})")
        return False


def geometry_mismatches(headers, reference, tol_px=1.0, tol_scale=0.01):
    """Waves whose CRPIX or CDELT differ from the reference header beyond a tolerance.

    ``headers`` maps wave code to header. Returns {wave: reason}; empty means one
    overlay fits every given wave. ``tol_scale`` is a relative CDELT tolerance.
    The tolerances are choices, not measured values.
    """
    out = {}
    for wave, hdr in headers.items():
        reasons = []
        for axis in ("1", "2"):
            if abs(float(hdr["CRPIX" + axis]) - float(reference["CRPIX" + axis])) > tol_px:
                reasons.append(f"CRPIX{axis} {hdr['CRPIX' + axis]} vs {reference['CRPIX' + axis]}")
            ref = float(reference["CDELT" + axis])
            if abs(float(hdr["CDELT" + axis]) - ref) > tol_scale * abs(ref):
                reasons.append(f"CDELT{axis} {hdr['CDELT' + axis]} vs {reference['CDELT' + axis]}")
        if (hdr.get("NAXIS1"), hdr.get("NAXIS2")) != (reference.get("NAXIS1"), reference.get("NAXIS2")):
            reasons.append("image size differs")
        if reasons:
            out[wave] = "; ".join(reasons)
    return out


def overlay_manifest(updated):
    """Body of manifest/overlay.json, which the Lambda folds into index.json as overlay_grid."""
    return {"key": OVERLAY_KEY, "updated": updated, "source_id": OVERLAY_SOURCE_ID}


def publish_grid_overlay(fits_dir, work_dir, settings=None, size=1024):
    """Render the overlay from the integrated 171 FITS and upload it and its manifest.

    Writes ``overlay/grid_1k.png`` (not under ``1k/``, so the video Lambda never
    fires on it) and ``manifest/overlay.json``, both under ``settings.prefix``.
    Returns the list of keys written; [] when skipped. Never raises: the
    overlay must not stop a reducer run.
    """
    try:
        from sunback.putter.AwsPutter import _find_fits, upload_public
        from sunback.putter.serve_keys import SERVED_CHANNELS
        from sunback.settings import NrtSettings

        settings = settings if settings is not None else NrtSettings.from_env()
        source = _find_fits(fits_dir, OVERLAY_SOURCE_WAVE)
        if source is None:
            logger.info(f"grid overlay: no AIAsynoptic{OVERLAY_SOURCE_WAVE}.fits under {fits_dir}; skipped")
            return []
        reference = _image_header(source)
        headers = {}
        for wave in SERVED_CHANNELS:
            path = _find_fits(fits_dir, wave)
            if path is not None and wave != OVERLAY_SOURCE_WAVE:
                try:
                    headers[wave] = _image_header(path)
                except (OSError, ValueError, KeyError) as exc:
                    logger.info(f"grid overlay: could not read {path}: {exc}")
        for wave, reason in geometry_mismatches(headers, reference).items():
            logger.warning(f"grid overlay: {wave} geometry differs from {OVERLAY_SOURCE_WAVE} ({reason}); "
                           "the overlay may not line up on that still")

        os.makedirs(work_dir, exist_ok=True)
        png = os.path.join(work_dir, "grid_overlay_1k.png")
        if not render_grid_overlay(header_map(reference), png, size=size):
            return []
        written = [upload_public(png, OVERLAY_KEY, "image/png", cache_control="no-cache", settings=settings)]
        updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        doc = os.path.join(work_dir, "overlay.json")
        with open(doc, "w", encoding="utf-8") as fp:
            json.dump(overlay_manifest(updated), fp)
        written.append(upload_public(doc, OVERLAY_MANIFEST_KEY, "application/json", cache_control="no-cache",
                                     settings=settings))
        logger.info(f"grid overlay: wrote {written}")
        return written
    except Exception as exc:  # optional output: log and carry on
        logger.error(f"grid overlay: publish failed ({type(exc).__name__}: {exc})")
        return []

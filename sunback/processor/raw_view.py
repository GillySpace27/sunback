"""The "before" half of the swipe: each integrated FITS with a plain log stretch (SB-16).

The served still is RHEF processed. This module renders the same integrated
frame (the first 2-D image HDU of AIAsynoptic<wave>.fits, before any filter)
with a log10 stretch between two percentiles and the same AIA colour table,
flipped top to bottom exactly as ImageProcessorCV writes the still, so the two
halves of a swipe line up pixel for pixel. Both are visualizations, not
calibrated radiances.
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

LO_PCT = 1.0    # chosen display percentiles, not measured values
HI_PCT = 99.5


def first_image(fits_path):
    """(data, header) of the first 2-D image HDU: the integrated frame before any filter."""
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.header.get("NAXIS", 0) == 2 and hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"no 2-D image HDU in {fits_path}")


def log_stretch(data, lo_pct=LO_PCT, hi_pct=HI_PCT):
    """log10 of the data between two percentiles of its positive finite pixels, as uint8."""
    data = np.asarray(data, dtype=np.float64)
    good = data[np.isfinite(data) & (data > 0)]
    if good.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(good, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo * 10.0
    clipped = np.clip(np.nan_to_num(data, nan=lo), lo, hi)
    scaled = (np.log10(clipped) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return np.round(scaled * 255).astype(np.uint8)


def colour_png(unit, wave, out_png):
    """Write a 0..1 array as a PNG in the AIA colour table, flipped like the served still."""
    import astropy.units as u
    import cv2

    from sunback.science.color_tables import aia_color_table

    cmap = aia_color_table(int(wave) * u.angstrom)
    rgb = (cmap(np.flipud(unit))[:, :, :3] * 255).astype(np.uint8)
    if not cv2.imwrite(str(out_png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {out_png}")
    return str(out_png)


def raw_png_name(wave):
    """DrGilly_<wave>_raw.png, the local name serve_keys.raw_id_for_local_png maps."""
    return f"DrGilly_{wave}_raw.png"


def render_raw_png(fits_path, out_png, wave):
    """Log-stretched PNG of one integrated FITS; returns the path written."""
    data, _ = first_image(fits_path)
    return colour_png(log_stretch(data) / 255.0, wave, out_png)


def publish_raw_views(fits_dir, work_dir, settings=None):
    """Render and upload raw/rhef_<id>_raw_1k.png for every served AIA wave found.

    Composites (rainbow, composite_uv, dem) get no raw view in v1: they have
    no single source frame. Keys go under settings.prefix through
    upload_public and never under 1k/, so the video Lambda never fires on
    them. Returns the keys written. Never raises: a missing raw view must not
    stop a reducer run.
    """
    written = []
    try:
        from sunback.putter.AwsPutter import _find_fits, upload_public
        from sunback.putter.serve_keys import SERVED_CHANNELS, raw_id_for_local_png, s3_raw_key
        from sunback.settings import NrtSettings

        settings = settings if settings is not None else NrtSettings.from_env()
        os.makedirs(work_dir, exist_ok=True)
        for wave in SERVED_CHANNELS:
            source = _find_fits(fits_dir, wave)
            if source is None:
                continue
            try:
                png = render_raw_png(source, os.path.join(work_dir, raw_png_name(wave)), wave)
                written.append(upload_public(png, s3_raw_key(raw_id_for_local_png(png)), "image/png",
                                             cache_control="no-cache", settings=settings))
            except Exception as exc:  # one wave failing never stops the others
                logger.warning(f"raw view: {wave} failed ({type(exc).__name__}: {exc})")
        logger.info(f"raw view: wrote {len(written)} keys; no raw view for composites")
    except Exception as exc:
        logger.error(f"raw view: publish failed ({type(exc).__name__}: {exc})")
    return written

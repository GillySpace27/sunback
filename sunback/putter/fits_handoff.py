"""The FITS behind every AIA picture (SB-17).

After a reducer run, each served AIA wave's integrated synoptic FITS is
published as fits/aia_<id>_latest.fits.gz: the empty primary HDU plus the
integrated frame (the first 2-D image HDU, before any filter) with its full
header, gzipped. The RHEF frames the reducer appends to the local file are
left out; they are visualizations, not data. The key is latest-only and
rewritten in place each run, like 1k/; no versioned copy is kept.
Composites (rainbow, composite_uv, dem) have no single source file and get
no FITS.
"""
import logging
import os

logger = logging.getLogger(__name__)

FITS_CONTENT_TYPE = "application/gzip"


def write_integrated_gz(src_fits, out_gz):
    """Write primary + the first 2-D image HDU of src_fits as a gzipped FITS (astropy gzips a .gz name)."""
    from astropy.io import fits

    with fits.open(src_fits) as hdul:
        image = next((h for h in hdul if h.header.get("NAXIS", 0) == 2 and h.data is not None), None)
        if image is None:
            raise ValueError(f"no 2-D image HDU in {src_fits}")
        header = image.header.copy()
        for key in ("ZIMAGE", "ZBITPIX", "ZNAXIS", "ZNAXIS1", "ZNAXIS2", "ZTILE1", "ZTILE2", "ZCMPTYPE",
                    "ZQUANTIZ", "ZDITHER0"):
            header.remove(key, ignore_missing=True)
        out = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy()) if hdul[0].data is None
                            else fits.PrimaryHDU(), fits.ImageHDU(data=image.data, header=header)])
        out.writeto(out_gz, overwrite=True, output_verify="silentfix")
    return out_gz


def publish_integrated_fits(fits_dir, work_dir, settings=None):
    """Upload fits/aia_<id>_latest.fits.gz for every served AIA wave found. Returns the keys written.

    Never raises: a missing download must not stop a reducer run.
    """
    written = []
    try:
        from sunback.putter.AwsPutter import _find_fits, upload_public
        from sunback.putter.serve_keys import SERVED_CHANNELS, s3_fits_key
        from sunback.settings import NrtSettings

        settings = settings if settings is not None else NrtSettings.from_env()
        os.makedirs(work_dir, exist_ok=True)
        for wave, product_id in SERVED_CHANNELS.items():
            source = _find_fits(fits_dir, wave)
            if source is None:
                continue
            try:
                gz = write_integrated_gz(source, os.path.join(work_dir, f"aia_{product_id}_latest.fits.gz"))
                written.append(upload_public(gz, s3_fits_key(product_id), FITS_CONTENT_TYPE,
                                             cache_control="no-cache", settings=settings))
            except Exception as exc:  # one wave failing never stops the others
                logger.warning(f"fits handoff: {wave} failed ({type(exc).__name__}: {exc})")
        logger.info(f"fits handoff: wrote {len(written)} keys")
    except Exception as exc:
        logger.error(f"fits handoff: publish failed ({type(exc).__name__}: {exc})")
    return written

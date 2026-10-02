"""Write a small synthetic AIA-like FITS file for offline tests.

Layout matches a JSOC synoptic file: an empty primary HDU plus a compressed
2-D image HDU (see sunback/fetcher/nrt_integrate.py). The header carries the
keys the pipeline reads (T_REC, T_OBS, WAVELNTH, X0_MP, Y0_MP, R_SUN) and the
WCS and observer keys sunpy.map.Map needs. Every number here is synthetic:
the plate scale and solar radius are chosen so a 64-pixel disk fits the
frame. They are not AIA calibration values and must not be quoted as such.
"""
import pathlib

import numpy as np
from astropy.io import fits

SYNTHETIC_RSUN_ARCSEC = 960.0  # synthetic; sets the disk size only
SYNTHETIC_DSUN_M = 1.496e11    # synthetic observer distance, about 1 au


def make_synthetic_aia_fits(path, wave="0171", shape=(64, 64),
                            date_obs="2026-09-28T12:00:00.000", seed=0):
    """Write the file and return its path as a pathlib.Path.

    The image is a limb-darkened disk plus Gaussian noise from ``seed``, so two
    calls with the same arguments write identical data.
    """
    path = pathlib.Path(path)
    ny, nx = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    r_pix = 0.4 * min(nx, ny)
    rr = np.hypot(xx - cx, yy - cy) / r_pix
    disk = np.where(rr <= 1.0, 1000.0 * np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0)), 50.0)
    data = (disk + rng.normal(0.0, 5.0, size=shape)).astype(np.float32)

    arcsec_per_pix = SYNTHETIC_RSUN_ARCSEC / r_pix
    hdr = fits.Header()
    hdr["TELESCOP"] = "SDO/AIA"
    hdr["INSTRUME"] = "AIA_3"
    hdr["DETECTOR"] = "AIA"
    hdr["WAVELNTH"] = int(wave)
    hdr["WAVEUNIT"] = "angstrom"
    hdr["T_REC"] = date_obs[:19] + "Z"
    hdr["T_OBS"] = date_obs[:19] + "Z"
    hdr["DATE-OBS"] = date_obs
    hdr["EXPTIME"] = 2.0
    hdr["CTYPE1"] = "HPLN-TAN"
    hdr["CTYPE2"] = "HPLT-TAN"
    hdr["CUNIT1"] = "arcsec"
    hdr["CUNIT2"] = "arcsec"
    hdr["CDELT1"] = arcsec_per_pix
    hdr["CDELT2"] = arcsec_per_pix
    hdr["CRPIX1"] = cx + 1.0
    hdr["CRPIX2"] = cy + 1.0
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = 0.0
    hdr["CROTA2"] = 0.0
    hdr["RSUN_OBS"] = SYNTHETIC_RSUN_ARCSEC
    hdr["DSUN_OBS"] = SYNTHETIC_DSUN_M
    hdr["HGLN_OBS"] = 0.0
    hdr["HGLT_OBS"] = 0.0
    hdr["X0_MP"] = cx
    hdr["Y0_MP"] = cy
    hdr["R_SUN"] = r_pix
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.CompImageHDU(data=data, header=hdr),
    ]).writeto(path, overwrite=True)
    return path

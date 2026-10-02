from os.path import split
from os import makedirs
from time import time
from tqdm import tqdm
from sunback.putter.Putter import Putter
import boto3
import cv2
import os
from copy import copy
from time import sleep

from datetime import datetime, timezone

from sunback.utils.array_util import get_thumblinks, make_thumbs
from sunback.putter.serve_keys import serve_id_for_local_png, s3_img_key, s3_thumb_key
from sunback.putter.serve_keys import SERVED_CHANNELS, s3_meta_key
from sunback.settings import NrtSettings
import glob
import json
import re

THUMB_PX = 512

S3_UPLOAD_ARGS = {'ACL': 'public-read', "ContentDisposition": "inline"}

txt_args = copy(S3_UPLOAD_ARGS)
txt_args["ContentType"] = "text/plain"

png_args = copy(S3_UPLOAD_ARGS)
png_args["ContentType"] = "image/png"

video_args = copy(S3_UPLOAD_ARGS)
video_args["ContentType"] = "video/mp4"

# 2026-10-01 (SB-5): the module-level boto3 objects below were replaced by a
# client created on first use (get_s3_client) and a bucket and key prefix read
# from NrtSettings (SUNBACK_BUCKET, SUNBACK_PREFIX). Kept for the record:
# s3 = boto3.resource('s3')
# bucket_name = 'the-sun-now'
# bucket = s3.Bucket(bucket_name)
# s3_client = boto3.client('s3')

_S3_CLIENT = None


def get_s3_client():
    """Return the boto3 S3 client, created on first use (not at import)."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client('s3')
    return _S3_CLIENT


def upload_public(local_path, key, content_type, cache_control=None, metadata=None, settings=None):
    """Upload one file public-read to settings.bucket at settings.prefix + key.

    Returns the full key written. With default settings the key is unchanged,
    so production keys stay byte-identical.
    """
    settings = settings if settings is not None else NrtSettings.from_env()
    full_key = settings.prefixed(key)
    extra = copy(S3_UPLOAD_ARGS)
    extra["ContentType"] = content_type
    if cache_control is not None:
        extra["CacheControl"] = cache_control
    if metadata is not None:
        extra["Metadata"] = metadata
    get_s3_client().upload_file(local_path, settings.bucket, full_key, ExtraArgs=extra)
    return full_key


# --- SB-9: honest time and visible provenance -----------------------------------
# Which integrated FITS feed each served still. Singles map through
# SERVED_CHANNELS; composites per CompositeRainbowImageProcessor (rgb1, rgb3);
# DEM per ScienceProcessor.DEMReconstructionProcessor.channel_waves.
PRODUCT_INPUT_WAVES = {
    **{pid: (wave,) for wave, pid in SERVED_CHANNELS.items()},
    "rainbow": ("0171", "0193", "0211"),
    "composite_uv": ("1700", "1600", "0304"),
    "dem": ("0094", "0131", "0171", "0193", "0211", "0335"),
}

# Wording for Gilly to confirm before merge (overview Q18); WS-17 shows the same strings.
RHEF_CITATION = "Gilly and Cranmer 2025, Solar Physics, doi:10.1007/s11207-025-02578-x"
CREDIT = "Imagery courtesy of NASA/SDO and the AIA science team."
# Read from sunback_webapp api/solar-archive.js (CITATIONS.AIA_PAPER); matches the vault's instruments/AIA.md.
AIA_PAPER = "Lemen, J. R., et al. 2012, Sol. Phys., 275, 17."
RHEF_NOTE = "RHEF output is a visualization, not a calibrated radiance."

_ISO_RE = re.compile(r"(\d{4})[-.](\d{2})[-.](\d{2})[T_ ](\d{2}):(\d{2}):(\d{2})")


def _iso_z(value):
    """'2026-09-28T12:00:00.12' or '2026.09.28_12:00:00' -> '2026-09-28T12:00:00Z'; '' if no time.

    A _TAI suffix, if a header ever carries one, is not converted.
    """
    m = _ISO_RE.search(str(value or ""))
    return "{}-{}-{}T{}:{}:{}Z".format(*m.groups()) if m else ""


def header_provenance(fits_path):
    """Times and integration recorded in one integrated synoptic FITS (empty strings when absent)."""
    from astropy.io import fits
    from sunback.fetcher.nrt_integrate import frame_time

    with fits.open(fits_path) as hdul:
        header = next((h.header for h in hdul if h.header.get("NAXIS", 0) == 2), hdul[-1].header)
        _, newest = frame_time(header)
        return {
            "obs_start": _iso_z(header.get("TINT_T0", "")),
            "obs_end": _iso_z(newest),
            "tint_n": str(header.get("TINT_N", "")),
            "tint_m": str(header.get("TINT_M", "")),
        }


def _find_fits(fits_dir, wave):
    if not fits_dir:
        return None
    name = f"AIAsynoptic{wave}.fits"
    direct = os.path.join(fits_dir, name)
    if os.path.exists(direct):
        return direct
    hits = sorted(glob.glob(os.path.join(fits_dir, "**", name), recursive=True), key=os.path.getmtime)
    return hits[-1] if hits else None


def png_provenance(png_path, fits_dir, upload_time):
    """Provenance of one served still from the integrated FITS behind it.

    obstime is the newest input's newest frame (obs_end); obs_start is the
    oldest frame of any input. Without a readable header time the upload time
    is used and flagged obstime_source='upload'.
    """
    product_id = serve_id_for_local_png(png_path)
    waves = PRODUCT_INPUT_WAVES.get(product_id, ())
    found = []
    for wave in waves:
        path = _find_fits(fits_dir, wave)
        if path is None:
            continue
        try:
            found.append(header_provenance(path))
        except (OSError, ValueError) as exc:
            print(f"\t* provenance: could not read {path}: {exc}")
    ends = [f["obs_end"] for f in found if f["obs_end"]]
    starts = [f["obs_start"] or f["obs_end"] for f in found if f["obs_end"]]
    newest = max(found, key=lambda f: f["obs_end"]) if ends else {}
    if ends:
        return {"product_id": product_id, "obstime": max(ends), "obstime_source": "header",
                "obs_start": min(starts), "obs_end": max(ends),
                "tint_n": newest.get("tint_n", ""), "tint_m": newest.get("tint_m", ""),
                "inputs": list(waves)}
    return {"product_id": product_id, "obstime": upload_time, "obstime_source": "upload",
            "obs_start": "", "obs_end": "", "tint_n": "", "tint_m": "", "inputs": list(waves)}


def obstime_for_png(png_path, fits_dir):
    """(obstime_iso, source, obs_start_iso, obs_end_iso); source is 'header' or 'upload'."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prov = png_provenance(png_path, fits_dir, now)
    return prov["obstime"], prov["obstime_source"], prov["obs_start"], prov["obs_end"]


def _version(dist):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


def png_text_chunks(prov):
    return {"obs_start": prov["obs_start"], "obs_end": prov["obs_end"], "n_frames": prov["tint_n"],
            "method": prov["tint_m"], "sunkit_image_version": _version("sunkit-image"),
            "sunback_version": _version("sunback")}


def write_png_text(src_png, dst_png, chunks):
    """Lossless PNG re-save with tEXt chunks (the renderer writes with cv2, which has no hook)."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    for key, value in chunks.items():
        info.add_text(key, str(value))
    with Image.open(src_png) as im:
        im.save(dst_png, format="PNG", pnginfo=info)
    return dst_png


def sidecar_doc(prov):
    """schema.org ImageObject; field names follow the Solar Archive provenance JSON
    (sunback_webapp api/bundler.js _buildProvenanceJsonLd)."""
    pid = prov["product_id"]
    when = prov["obstime"]
    props = [
        ("instrument", "AIA"), ("spacecraft", "SDO"), ("productId", pid),
        ("inputChannels", ",".join(w.lstrip("0") for w in prov["inputs"])),
        ("observationStartUTC", prov["obs_start"]), ("observationDateUTC", prov["obs_end"] or when),
        ("obstimeSource", prov["obstime_source"]),
        ("integrationFrames", prov["tint_n"]), ("integrationMethod", prov["tint_m"]),
        ("pipeline", "sunback NRT reducer: SunPy + sunkit-image (RHEF)"),
        ("sunkitImageVersion", _version("sunkit-image")), ("sunbackVersion", _version("sunback")),
        ("note", RHEF_NOTE),
    ]
    return {
        "@context": "https://schema.org",
        "@type": "ImageObject",
        "name": f"The Sun, right now: {pid}",
        "dateCreated": when,
        "creator": {"@type": "Organization", "name": "The Sun, right now (gilly.space)"},
        "contentLocation": "NASA/SDO/AIA",
        "encodingFormat": "image/png",
        "license": "https://sdo.gsfc.nasa.gov/data/rules.php",
        "creditText": CREDIT,
        "citation": [CREDIT, AIA_PAPER, RHEF_CITATION],
        "isBasedOn": [{"@type": "Dataset", "name": f"AIA synoptic NRT {w.lstrip('0')} A",
                       "datePublished": prov["obs_end"] or when,
                       "distributor": "Joint Science Operations Center (JSOC), Stanford",
                       "via": "https://jsoc1.stanford.edu/data/aia/synoptic/nrt/"} for w in prov["inputs"]],
        "potentialAction": {"@type": "ViewAction",
                            "description": "RHEF (Radial Histogram Equalization Filter); " + RHEF_CITATION},
        "additionalProperty": [{"@type": "PropertyValue", "name": n, "value": str(v)} for n, v in props],
    }


class AwsPutter(Putter):
    filt_name = "AWSputter"
    description = "Upload Images to AWS S3 (bucket and prefix from NrtSettings)"
    progress_verb = "Uploaded"
    progress_unit = "Images"

    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        super().__init__(params, quick, rp, in_name)
        self.ii = 0
        self.pbar = None
        self.to_upload = None

    def put(self, params=None):
        if params is not None:
            self.__init__(params)
        self.settings = NrtSettings.from_env()
        print(" V Uploading PNGs to s3://{}/{}...".format(self.settings.bucket, self.settings.prefix), flush=True)
        # NOTE: do NOT empty the bucket. The Lambda video-builder maintains the
        # frames/ queue and video/ outputs there; wiping would destroy the 48h
        # sliding window every run. The reducer only overwrites its own keys.
        # self.empty_the_bucket()   # <-- intentionally disabled (see spec)
        # obstime stamps the 1k stills so the Lambda can order the video queue.
        # Upload time ~= observation time (reducer runs right after NRT publish).
        self.obstime = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.__upload_files()
        self.__upload_tscan_video()
        self.__save_times()

    def __upload_tscan_video(self):
        """Upload the DEM temperature-scan video to a stable key for the DEM card.

        This is a pre-rendered scan over temperature (not a 48h timelapse), so it
        bypasses the Lambda and is uploaded directly, overwriting each run.
        """
        import glob
        roots = [self.params.imgs_top_directory(), self.params.base_directory()]
        found = None
        for root in roots:
            if not root:
                continue
            for pat in ("a_temp_video_small.mp4", "a_temp_video.mp4"):
                hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
                if hits:
                    found = hits[0]
                    break
            if found:
                break
        if not found:
            print("\t* No temperature-scan video found; skipping.")
            return
        key = upload_public(found, "video/rhef_tscan.mp4", "video/mp4", settings=self._settings())
        print(f"\t* Uploaded temperature-scan video -> {key}")

    def _settings(self):
        settings = getattr(self, "settings", None)
        return settings if settings is not None else NrtSettings.from_env()

    def empty_the_bucket(self):
        raise RuntimeError(
            "empty_the_bucket is fenced: agent code never deletes S3 objects; "
            "staging cleanup is a Gilly-approved lifecycle rule (SB-7)"
        )
        # 2026-10-01 (SB-5): original body, kept for the record. It deleted every
        # object in the bucket, including the Lambda's frames/ queue.
        # print("\t* Emptying Bucket...", end='')
        # bucket.objects.all().delete()
        # print("Done!")

    def get_file_list(self, force=False):
        if self.to_upload is None or force:
            # Only the served PNG stills. Videos are built by the Lambda, not here,
            # so the .mp4 is no longer collected/uploaded.
            self.to_upload = [
                f for f in self.params.local_imgs_paths()
                if "_orig" not in f and serve_id_for_local_png(f) is not None
            ]

        self.pbar = tqdm(self.to_upload, desc="\r\t* Uploading Files", ncols=120)
        return self.to_upload, self.pbar

    def __upload_files(self):
        to_upload, pbar = self.get_file_list()
        if self.params.multi_pool is not None:
            results = self.params.multi_pool.imap(self.do_upload, to_upload)
            for res in results:
                pbar.update()
                self.ii += 1
        else:
            self.upload_serial(to_upload, pbar)
        pbar.close()
        print(" ^ Success! Uploaded {} PNGs\n".format(len(self.params.local_imgs_paths())))

    def upload_serial(self, to_upload=None, pbar=None):
        if to_upload is None:
            to_upload, pbar = self.get_file_list()
        for upload in to_upload:
            self.do_upload(upload)
            pbar.update()
            self.ii += 1

    def do_upload(self, root_path):
        """Upload one served still as 1k/rhef_<id>_1k.png + a THUMB_PX (512) square thumb.

        The 1k still upload is what fires the Lambda video-builder; obstime
        metadata lets the Lambda order the 48h frame queue.
        """
        product_id = serve_id_for_local_png(root_path)
        if product_id is None:
            return  # not a served product (see serve_keys.serve_id_for_local_png)

        settings = self._settings()
        # SB-9: observation time from the integrated FITS header, per product;
        # the upload time stamped in put() is only the flagged fallback.
        prov = png_provenance(root_path, self.params.fits_directory(), getattr(self, "obstime", ""))
        meta = {"obstime": prov["obstime"]}
        for key in ("obstime_source", "obs_start", "obs_end", "tint_n", "tint_m"):
            if prov[key]:
                meta[key] = prov[key]
        if prov["obstime_source"] == "upload":
            print(f"\t* {product_id}: no header time found; obstime falls back to upload time")
        tagged = write_png_text(root_path, os.path.join(os.path.dirname(root_path), f".meta_{product_id}.png"),
                                png_text_chunks(prov))

        # full-res 1k still (pixels identical to root_path; tEXt chunks added)
        upload_public(tagged, s3_img_key(product_id), "image/png", metadata=meta, settings=settings)

        # THUMB_PX (512) square thumbnail (square 1024 source -> direct resize)
        img = cv2.imread(root_path, cv2.IMREAD_UNCHANGED)
        thumb_path = os.path.join(os.path.dirname(root_path),
                                  f".thumb_{product_id}.png")
        cv2.imwrite(thumb_path, cv2.resize(img, (THUMB_PX, THUMB_PX),
                                           interpolation=cv2.INTER_AREA))
        upload_public(thumb_path, s3_thumb_key(product_id), "image/png", settings=settings)

        # SB-9: provenance sidecar, rewritten each run beside the still
        sidecar = os.path.join(os.path.dirname(root_path), f".meta_{product_id}.json")
        with open(sidecar, "w", encoding="utf-8") as fp:
            json.dump(sidecar_doc(prov), fp, indent=2)
        upload_public(sidecar, s3_meta_key(product_id), "application/json", cache_control="no-cache",
                      settings=settings)

    def __save_times(self):
        print("\t* Uploading Time File...", end='', flush=True)
        path = self.params.time_path()
        path2 = path.replace(".txt", "_readable.txt")

        frame, wave, t_rec, center, int_time, nm = self.load_this_fits_frame(self.params.local_fits_paths()[0], -1)

        with open(path, "w") as fp:
            fp.write(t_rec)

        tz_list = []
        nzt = self.clean_time_string(t_rec, "NZ").replace("NZDT, ", "NZDT,").replace("NZST, ", "NZST,")
        tz_list.append(nzt)
        tz_list.append(self.clean_time_string(t_rec, 'Japan'))
        tz_list.append(self.clean_time_string(t_rec, "EET").replace("EEST, ", "EEST,"))
        tz_list.append("       ~*~")
        tz_list.append(self.clean_time_string(t_rec, None))
        tz_list.append("       ~*~")
        tz_list.append(self.clean_time_string(t_rec, "US/Eastern"))
        tz_list.append(self.clean_time_string(t_rec, "US/Central"))
        tz_list.append(self.clean_time_string(t_rec, "US/Mountain"))
        tz_list.append(self.clean_time_string(t_rec, "US/Pacific"))
        tz_list.append(self.clean_time_string(t_rec, "US/Hawaii"))

        with open(path2, "w") as fp:
            for item in tz_list:
                fp.write(item + "\n")

        settings = self._settings()
        upload_public(path, os.path.basename(path), "text/plain", settings=settings)
        if settings.write_readable_times:
            upload_public(path2, os.path.basename(path2), "text/plain", settings=settings)
        print("Done! ", flush=True)
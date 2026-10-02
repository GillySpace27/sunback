"""SB-9: honest observation time and provenance for every uploaded still.

Every S3 call goes to a recorder; nothing here reaches AWS. Integrated FITS are
made with the real write_integrated_synoptic, so TINT_T0 is exercised too.
"""
import importlib
import json
from unittest.mock import MagicMock

import numpy as np
from astropy.io import fits

from sunback.fetcher.nrt_integrate import write_integrated_synoptic

_sb9_aws = importlib.import_module("sunback.putter.AwsPutter")


def _sb9_frame(path, t_rec):
    hdr = fits.Header()
    hdr["WAVELNTH"] = 171
    hdr["T_REC"] = t_rec
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data=np.ones((4, 4), np.float32), header=hdr)]
                 ).writeto(str(path), overwrite=True)


def _sb9_integrated(fits_dir, wave, times):
    fits_dir.mkdir(exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        p = fits_dir / f"nrt_{wave}_{i}.fits"
        _sb9_frame(p, t)
        paths.append(str(p))
    out = fits_dir / f"AIAsynoptic{wave}.fits"
    write_integrated_synoptic(paths, str(out), method="median")
    return out


def _sb9_png(path):
    from PIL import Image
    Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3)).save(str(path))
    return str(path)


SB9_UPLOAD = "2026-09-28T12:30:00Z"   # what put() would stamp: half an hour after the observation


def test_single_channel_uses_header_time(tmp_path):
    fits_dir = tmp_path / "fits"
    _sb9_integrated(fits_dir, "0171", ["2026-09-28T11:52:00Z", "2026-09-28T11:56:00Z", "2026-09-28T12:00:00Z"])
    assert _sb9_aws.obstime_for_png(str(tmp_path / "DrGilly_0171_ups(rhef).png"), str(fits_dir)) == (
        "2026-09-28T12:00:00Z", "header", "2026-09-28T11:52:00Z", "2026-09-28T12:00:00Z")


def test_composite_spans_its_inputs(tmp_path):
    fits_dir = tmp_path / "fits"
    _sb9_integrated(fits_dir, "0171", ["2026-09-28T11:50:00Z", "2026-09-28T11:58:00Z"])
    _sb9_integrated(fits_dir, "0193", ["2026-09-28T11:48:00Z", "2026-09-28T11:59:00Z"])
    _sb9_integrated(fits_dir, "0211", ["2026-09-28T11:51:00Z", "2026-09-28T12:01:00Z"])
    prov = _sb9_aws.png_provenance(str(tmp_path / "BGR_0171_0193_0211_ups(rhef).png"), str(fits_dir), SB9_UPLOAD)
    assert prov["obstime"] == "2026-09-28T12:01:00Z"       # newest input
    assert prov["obs_start"] == "2026-09-28T11:48:00Z"     # oldest frame of any input
    assert prov["obs_end"] == "2026-09-28T12:01:00Z"
    assert prov["obstime_source"] == "header"
    assert prov["inputs"] == ["0171", "0193", "0211"]
    assert (prov["tint_n"], prov["tint_m"]) == ("2", "median")


def test_missing_fits_falls_back_to_upload_time(tmp_path):
    prov = _sb9_aws.png_provenance(str(tmp_path / "DrGilly_0304_ups(rhef).png"), str(tmp_path / "nowhere"), SB9_UPLOAD)
    assert (prov["obstime"], prov["obstime_source"], prov["obs_start"], prov["obs_end"]) == (SB9_UPLOAD, "upload", "", "")


def test_png_text_chunks_round_trip_and_pixels_unchanged(tmp_path):
    from PIL import Image
    src = _sb9_png(tmp_path / "DrGilly_0171_ups(rhef).png")
    chunks = {"obs_start": "2026-09-28T11:52:00Z", "obs_end": "2026-09-28T12:00:00Z", "n_frames": "3",
              "method": "median", "sunkit_image_version": "x", "sunback_version": "y"}
    out = _sb9_aws.write_png_text(src, str(tmp_path / ".meta_171.png"), chunks)
    with Image.open(out) as im:
        assert {k: im.text[k] for k in chunks} == chunks
        assert np.array_equal(np.asarray(im), np.asarray(Image.open(src)))


def test_do_upload_stamps_header_time_and_writes_sidecar(tmp_path, monkeypatch):
    uploads = []

    class Recorder:
        def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
            body = open(Filename, "rb").read()
            uploads.append((Key, dict(ExtraArgs or {}), body))

    monkeypatch.setattr(_sb9_aws, "_S3_CLIENT", Recorder())
    monkeypatch.delenv("SUNBACK_PREFIX", raising=False)
    fits_dir = tmp_path / "fits"
    _sb9_integrated(fits_dir, "0171", ["2026-09-28T11:52:00Z", "2026-09-28T11:56:00Z", "2026-09-28T12:00:00Z"])
    png = _sb9_png(tmp_path / "DrGilly_0171_ups(rhef).png")

    putter = _sb9_aws.AwsPutter.__new__(_sb9_aws.AwsPutter)  # skip Processor.__init__ (no pipeline needed)
    putter.params = MagicMock()
    putter.params.fits_directory.return_value = str(fits_dir)
    putter.obstime = SB9_UPLOAD
    putter.do_upload(png)

    keys = [k for k, _, _ in uploads]
    assert keys == ["1k/rhef_171_1k.png", "thumb/rhef_171_thumb.png", "meta/rhef_171.json"]
    meta = uploads[0][1]["Metadata"]
    assert meta == {"obstime": "2026-09-28T12:00:00Z", "obstime_source": "header",
                    "obs_start": "2026-09-28T11:52:00Z", "obs_end": "2026-09-28T12:00:00Z",
                    "tint_n": "3", "tint_m": "median"}
    assert b"obs_start" in uploads[0][2]                     # tEXt chunk rides in the uploaded still
    side = json.loads(uploads[2][2])
    assert side["@type"] == "ImageObject" and side["dateCreated"] == "2026-09-28T12:00:00Z"
    props = {p["name"]: p["value"] for p in side["additionalProperty"]}
    assert props["observationStartUTC"] == "2026-09-28T11:52:00Z"
    assert props["observationDateUTC"] == "2026-09-28T12:00:00Z"
    assert props["obstimeSource"] == "header"
    assert uploads[2][1]["ContentType"] == "application/json"

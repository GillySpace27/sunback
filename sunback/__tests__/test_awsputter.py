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
    # obstime is the upload time, exactly as before SB-9 (the Lambda keys frames and `updated` on it);
    # the header observation window rides in obs_start / obs_end.
    assert meta == {"obstime": SB9_UPLOAD,
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


# --- SB-11: AwsPutter through botocore's Stubber ---------------------------------------------
# A real boto3 S3 client with a Stubber attached replaces AwsPutter's lazy client
# (``_S3_CLIENT``, SB-5), so every PutObject request is checked against the S3 API model
# (parameter names, types, the ACL enum) as well as against the keys, content types, ACL and
# metadata production writes today. Nothing leaves the process: the Stubber answers before a
# request is signed or sent. (SB-9's tests above use a hand-written recorder instead.)
import boto3  # noqa: E402
import pytest  # noqa: E402
from botocore.config import Config  # noqa: E402
from botocore.stub import ANY, Stubber  # noqa: E402

aws = _sb9_aws

BUCKET = "the-sun-now"
OBSTIME = "2026-09-28T12:00:00Z"
T_REC = "2026-09-28T12:00:00.00"


def expected_put(key, content_type, metadata=None, cache_control=None):
    params = {"Bucket": BUCKET, "Key": key, "Body": ANY, "ACL": "public-read",
              "ContentDisposition": "inline", "ContentType": content_type}
    if metadata is not None:
        params["Metadata"] = metadata
    if cache_control is not None:
        params["CacheControl"] = cache_control
    return params


def still_metadata():
    """Metadata of a 1k still when no FITS header time is found: upload time only, as before SB-9."""
    return {"obstime": OBSTIME}


@pytest.fixture
def stubbed(monkeypatch):
    for name in ("SUNBACK_BUCKET", "SUNBACK_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUNBACK_WRITE_READABLE_TIMES", "1")  # independent of SB-6's default flip (Q25)
    try:  # botocore >= 1.36: keep s3transfer from adding ChecksumAlgorithm, so params compare across versions
        config = Config(request_checksum_calculation="when_required")
    except TypeError:  # older botocore never adds it
        config = Config()
    client = boto3.client("s3", region_name="us-east-2", aws_access_key_id="testing",
                          aws_secret_access_key="testing", config=config)
    stubber = Stubber(client)
    monkeypatch.setattr(aws, "_S3_CLIENT", client)
    with stubber:
        yield stubber
        stubber.assert_no_pending_responses()


def make_putter(tmp_path, png_names):
    import cv2

    pngs = []
    for name in png_names:
        path = tmp_path / name
        cv2.imwrite(str(path), np.zeros((16, 16), np.uint8))
        pngs.append(str(path))
    params = type("P", (), {})()
    params.multi_pool = None
    params.local_imgs_paths = lambda: pngs
    params.imgs_top_directory = lambda: str(tmp_path)
    params.base_directory = lambda: str(tmp_path)
    params.fits_directory = lambda: str(tmp_path / "no_fits_here")
    params.time_path = lambda: str(tmp_path / "image_times.txt")
    params.local_fits_paths = lambda: [str(tmp_path / "AIAsynoptic0171.fits")]
    putter = aws.AwsPutter.__new__(aws.AwsPutter)  # skip Processor.__init__ (no FITS on disk)
    putter.params = params
    putter.ii = 0
    putter.pbar = None
    putter.to_upload = None
    putter.load_this_fits_frame = lambda *a, **k: (None, "0171", T_REC, None, None, None)
    putter.clean_time_string = lambda t, zone=None, out_fmt=None: f"{zone}|{t}"
    return putter


def queue_still(stubbed, product_id):
    """The three PutObject calls one served still makes, in order (SB-9 added the sidecar)."""
    stubbed.add_response("put_object", {},
                         expected_put(f"1k/rhef_{product_id}_1k.png", "image/png", still_metadata()))
    stubbed.add_response("put_object", {}, expected_put(f"thumb/rhef_{product_id}_thumb.png", "image/png"))
    stubbed.add_response("put_object", {}, expected_put(f"meta/rhef_{product_id}.json", "application/json",
                                                        cache_control="no-cache"))


def test_do_upload_writes_1k_still_thumb_and_sidecar(stubbed, tmp_path):
    putter = make_putter(tmp_path, ["DrGilly_0171_ups(rhef).png"])
    putter.obstime = OBSTIME
    queue_still(stubbed, "171")
    putter.do_upload(str(tmp_path / "DrGilly_0171_ups(rhef).png"))


def test_thumb_is_512_square(stubbed, tmp_path):
    import cv2

    putter = make_putter(tmp_path, ["DrGilly_0193_ups(rhef).png"])
    putter.obstime = OBSTIME
    queue_still(stubbed, "193")
    putter.do_upload(str(tmp_path / "DrGilly_0193_ups(rhef).png"))
    thumb = cv2.imread(str(tmp_path / ".thumb_193.png"), cv2.IMREAD_UNCHANGED)
    assert thumb.shape[:2] == (aws.THUMB_PX, aws.THUMB_PX) == (512, 512)


def test_unserved_png_is_not_uploaded(stubbed, tmp_path):
    putter = make_putter(tmp_path, ["DrGilly_1234_ups(rhef).png"])
    putter.obstime = OBSTIME
    putter.do_upload(str(tmp_path / "DrGilly_1234_ups(rhef).png"))  # no response queued: any call would fail


def test_put_writes_stills_then_time_files(stubbed, tmp_path, monkeypatch):
    putter = make_putter(tmp_path, ["DrGilly_0171_ups(rhef).png", "C_isothermal.png"])
    monkeypatch.setattr(aws, "datetime", type("D", (), {"now": staticmethod(
        lambda tz=None: __import__("datetime").datetime(2026, 9, 28, 12, 0, 0, tzinfo=tz))}))
    for key in ("171", "dem"):
        queue_still(stubbed, key)
    stubbed.add_response("put_object", {}, expected_put("image_times.txt", "text/plain"))
    stubbed.add_response("put_object", {}, expected_put("image_times_readable.txt", "text/plain"))
    putter.put()
    assert (tmp_path / "image_times.txt").read_text() == T_REC
    assert (tmp_path / "image_times_readable.txt").read_text().splitlines()[4] == f"None|{T_REC}"


# --- review fix: provenance must never block the 1k still ---------------------------------------
# Each SB-9 step (read FITS headers, re-save the PNG with tEXt chunks, write and put the JSON
# sidecar) may raise anything. The plain 1k still, its thumb and image_times.txt still go out,
# with the pre-SB-9 behaviour: upload time as obstime and no provenance.


def _recording_putter(monkeypatch, tmp_path, fail_keys=()):
    """AwsPutter whose S3 client is a recorder; keys in fail_keys raise RuntimeError on upload."""
    uploads = []

    class Recorder:
        def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
            if Key in fail_keys:
                raise RuntimeError(f"simulated S3 failure for {Key}")
            uploads.append((Key, dict(ExtraArgs or {}), open(Filename, "rb").read()))

    monkeypatch.setattr(aws, "_S3_CLIENT", Recorder())
    monkeypatch.delenv("SUNBACK_PREFIX", raising=False)
    fits_dir = tmp_path / "fits"
    _sb9_integrated(fits_dir, "0171", ["2026-09-28T11:52:00Z", "2026-09-28T12:00:00Z"])
    png = _sb9_png(tmp_path / "DrGilly_0171_ups(rhef).png")
    putter = aws.AwsPutter.__new__(aws.AwsPutter)
    putter.params = MagicMock()
    putter.params.fits_directory.return_value = str(fits_dir)
    putter.obstime = SB9_UPLOAD
    return putter, png, uploads


def _assert_plain_1k_went_out(uploads, png):
    by_key = {k: (extra, body) for k, extra, body in uploads}
    assert "1k/rhef_171_1k.png" in by_key
    assert by_key["1k/rhef_171_1k.png"][0]["Metadata"]["obstime"] == SB9_UPLOAD
    assert "thumb/rhef_171_thumb.png" in by_key
    return by_key


def _boom(*args, **kwargs):
    raise RuntimeError("simulated provenance failure")


def test_header_read_failure_still_uploads_the_1k(monkeypatch, tmp_path, caplog):
    putter, png, uploads = _recording_putter(monkeypatch, tmp_path)
    monkeypatch.setattr(aws, "header_provenance", _boom)
    with caplog.at_level("WARNING"):
        putter.do_upload(png)
    by_key = _assert_plain_1k_went_out(uploads, png)
    assert "obs_end" not in by_key["1k/rhef_171_1k.png"][0]["Metadata"]
    assert any("read FITS header" in r.getMessage() for r in caplog.records)


def test_provenance_lookup_failure_uploads_pre_sb9_still(monkeypatch, tmp_path, caplog):
    putter, png, uploads = _recording_putter(monkeypatch, tmp_path)
    monkeypatch.setattr(aws, "png_provenance", _boom)
    with caplog.at_level("WARNING"):
        putter.do_upload(png)
    by_key = _assert_plain_1k_went_out(uploads, png)
    assert by_key["1k/rhef_171_1k.png"][0]["Metadata"] == {"obstime": SB9_UPLOAD}
    assert by_key["1k/rhef_171_1k.png"][1] == open(png, "rb").read()   # untouched file
    assert "meta/rhef_171.json" not in by_key                          # no sidecar
    assert any("provenance" in r.getMessage() for r in caplog.records)


def test_png_resave_failure_uploads_the_original_png(monkeypatch, tmp_path, caplog):
    putter, png, uploads = _recording_putter(monkeypatch, tmp_path)
    monkeypatch.setattr(aws, "write_png_text", _boom)
    with caplog.at_level("WARNING"):
        putter.do_upload(png)
    by_key = _assert_plain_1k_went_out(uploads, png)
    assert by_key["1k/rhef_171_1k.png"][1] == open(png, "rb").read()
    assert any("tEXt" in r.getMessage() for r in caplog.records)


def test_sidecar_put_failure_does_not_raise(monkeypatch, tmp_path, caplog):
    putter, png, uploads = _recording_putter(monkeypatch, tmp_path, fail_keys=("meta/rhef_171.json",))
    with caplog.at_level("WARNING"):
        putter.do_upload(png)
    _assert_plain_1k_went_out(uploads, png)
    assert all(k != "meta/rhef_171.json" for k, _, _ in uploads)
    assert any("sidecar" in r.getMessage() for r in caplog.records)


def test_sidecar_build_failure_does_not_raise(monkeypatch, tmp_path, caplog):
    putter, png, uploads = _recording_putter(monkeypatch, tmp_path)
    monkeypatch.setattr(aws, "sidecar_doc", _boom)
    with caplog.at_level("WARNING"):
        putter.do_upload(png)
    _assert_plain_1k_went_out(uploads, png)
    assert any("sidecar" in r.getMessage() for r in caplog.records)


def test_put_still_writes_image_times_when_provenance_fails(stubbed, tmp_path, monkeypatch):
    putter = make_putter(tmp_path, ["DrGilly_0171_ups(rhef).png"])
    monkeypatch.setattr(aws, "datetime", type("D", (), {"now": staticmethod(
        lambda tz=None: __import__("datetime").datetime(2026, 9, 28, 12, 0, 0, tzinfo=tz))}))
    monkeypatch.setattr(aws, "png_provenance", _boom)
    stubbed.add_response("put_object", {}, expected_put("1k/rhef_171_1k.png", "image/png", {"obstime": OBSTIME}))
    stubbed.add_response("put_object", {}, expected_put("thumb/rhef_171_thumb.png", "image/png"))
    stubbed.add_response("put_object", {}, expected_put("image_times.txt", "text/plain"))
    stubbed.add_response("put_object", {}, expected_put("image_times_readable.txt", "text/plain"))
    putter.put()
    assert (tmp_path / "image_times.txt").read_text() == T_REC


def test_obstime_stays_upload_time_and_header_time_goes_to_obs_end(monkeypatch, tmp_path):
    """Two runs on the same observation must not share an obstime (the Lambda frame key)."""
    putter, png, uploads = _recording_putter(monkeypatch, tmp_path)
    putter.do_upload(png)
    putter.obstime = "2026-09-28T12:50:00Z"
    putter.do_upload(png)
    firsts = [extra["Metadata"] for k, extra, _ in uploads if k == "1k/rhef_171_1k.png"]
    assert [m["obstime"] for m in firsts] == [SB9_UPLOAD, "2026-09-28T12:50:00Z"]
    assert {m["obs_end"] for m in firsts} == {"2026-09-28T12:00:00Z"}
    assert "obstime_source" not in firsts[0]

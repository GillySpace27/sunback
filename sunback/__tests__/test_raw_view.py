"""SB-16: the raw (log-stretched) half of the swipe, offline."""
import numpy as np
import pytest
from PIL import Image

from aws_lambda.video_builder import manifest
from sunback.__tests__.fixtures.make_fits import make_synthetic_aia_fits
from sunback.__tests__.test_aws_putter_guards import fake_s3  # noqa: F401 (pytest fixture)
from sunback.processor import raw_view
from sunback.putter.serve_keys import raw_id_for_local_png, s3_raw_key, serve_id_for_local_png


def test_log_stretch_is_uint8_monotonic_and_spans_the_range():
    data = np.array([[100.0, 1000.0], [10000.0, 100000.0]], dtype=np.float32)
    out = raw_view.log_stretch(data, lo_pct=0, hi_pct=100)
    assert out.dtype == np.uint8 and out.shape == data.shape
    assert out.tolist() == [[0, 85], [170, 255]]   # computed: equal log steps


def test_log_stretch_survives_nan_and_non_positive():
    out = raw_view.log_stretch(np.array([[np.nan, -1.0], [0.0, 10.0]]))
    assert out.dtype == np.uint8
    assert raw_view.log_stretch(np.zeros((2, 2))).tolist() == [[0, 0], [0, 0]]


def test_raw_png_is_the_frame_flipped_like_the_still(tmp_path):
    fits_path = make_synthetic_aia_fits(tmp_path / "AIAsynoptic0171.fits", shape=(64, 64))
    from astropy.io import fits
    with fits.open(fits_path, mode="update") as hdul:   # FITS rows 0-7 (the bottom) nearly zero
        hdul[1].data[:8, :] = 1e-3
    out = raw_view.render_raw_png(str(fits_path), str(tmp_path / "x_raw.png"), "0171")
    img = np.asarray(Image.open(out).convert("L"), dtype=float)
    assert img.shape == (64, 64)
    # np.flipud, as ImageProcessorCV: FITS row y is PNG row 63 - y, so the dark band is at the PNG bottom
    assert img[-8:].max() < img[:8].min()


def test_raw_names_map_to_their_own_keys_and_never_to_the_still():
    assert raw_id_for_local_png("/x/DrGilly_0171_raw.png") == "171"
    assert raw_id_for_local_png("DrGilly_0094_raw.png") == "94"
    assert raw_id_for_local_png("DrGilly_0171_ups(rhef).png") is None
    assert raw_id_for_local_png("BGR_0171_0193_0211_raw.png") is None
    assert serve_id_for_local_png("DrGilly_0171_raw.png") is None
    assert serve_id_for_local_png("DrGilly_0171_ups(rhef).png") == "171"   # unchanged
    assert s3_raw_key("171") == manifest.raw1k_key("171") == "raw/rhef_171_raw_1k.png"
    assert manifest.product_from_1k_key(s3_raw_key("171")) is None   # never fires the Lambda


def test_fragment_carries_raw1k_only_when_given():
    base = manifest.build_manifest_fragment("171", updated="t", frame_count=1, integration={"frames": 5, "method": "m"})
    assert "raw1k" not in base
    full = manifest.build_manifest_fragment("171", updated="t", frame_count=1,
                                            integration={"frames": 5, "method": "m"}, raw1k=manifest.raw1k_key("171"))
    assert full["raw1k"] == "raw/rhef_171_raw_1k.png" and manifest.validate_fragment(full) == []


@pytest.mark.parametrize("prefix", ["", "staging/"])
def test_publish_writes_one_raw_key_per_aia_single_found(fake_s3, tmp_path, monkeypatch, prefix):  # noqa: F811
    monkeypatch.setenv("SUNBACK_PREFIX", prefix)
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    for wave in ("0171", "0094"):
        make_synthetic_aia_fits(fits_dir / f"AIAsynoptic{wave}.fits", wave=wave)
    keys = raw_view.publish_raw_views(str(fits_dir), str(tmp_path / "work"))
    assert keys == [prefix + "raw/rhef_171_raw_1k.png", prefix + "raw/rhef_94_raw_1k.png"]
    assert {x["ContentType"] for _, _, x in fake_s3.uploads} == {"image/png"}


def test_publish_with_nothing_to_do_writes_nothing(fake_s3, tmp_path):  # noqa: F811
    assert raw_view.publish_raw_views(str(tmp_path), str(tmp_path / "w")) == []
    assert raw_view.publish_raw_views(None, str(tmp_path / "w")) == []
    assert fake_s3.uploads == []


def test_lambda_names_raw1k_only_when_the_reducer_wrote_it(monkeypatch, tmp_path):
    from sunback.__tests__ import lambda_harness as lh

    fake = lh.FakeS3()
    handler = lh.load_handler(monkeypatch, fake, lh.fake_ffmpeg(tmp_path))
    assert handler._raw_view("171") is None
    fake._store("raw/rhef_171_raw_1k.png", b"png")
    assert handler._raw_view("171") == "raw/rhef_171_raw_1k.png"
    assert handler._raw_view("staging-only", "staging/") is None
    fake._store("raw/rhef_rainbow_raw_1k.png", b"png")
    assert handler._raw_view("rainbow") is None   # composites have no raw view


def test_reducer_publishes_raw_views_after_the_runner(monkeypatch):
    import importlib

    rsg = importlib.import_module("sunback.run.run_server_github")
    calls = []

    class Runner:
        def __init__(self, params):
            pass

        def start(self):
            calls.append("runner")

    monkeypatch.setattr(rsg, "SingleRunner", Runner)
    monkeypatch.setattr(rsg, "publish_grid_overlay", lambda *a, **k: calls.append("overlay") or [])
    monkeypatch.setattr(rsg, "publish_raw_views", lambda d, w, settings=None: calls.append(("raw", settings.prefix)) or [])
    monkeypatch.setenv("SUNBACK_PREFIX", "staging/")
    rsg.run_server_github()
    assert calls == ["runner", "overlay", ("raw", "staging/")]

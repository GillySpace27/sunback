"""Tests for multi-frame time integration (median/mean/sum) of NRT solar frames."""

import numpy as np
import pytest

from sunback.utils.time_integration import integrate_frames


def test_median_removes_cosmic_ray_spike():
    """A transient spike in one frame is rejected by the median."""
    base = np.full((4, 4), 10.0, dtype=np.float32)
    spike = base.copy()
    spike[2, 2] = 9999.0  # cosmic ray hit in a single frame
    result = integrate_frames([base, spike, base], method="median")
    assert result[2, 2] == pytest.approx(10.0)
    assert result.shape == (4, 4)


def test_mean_averages_frames():
    a = np.full((2, 2), 2.0, dtype=np.float32)
    b = np.full((2, 2), 4.0, dtype=np.float32)
    result = integrate_frames([a, b], method="mean")
    assert np.allclose(result, 3.0)


def test_sum_adds_frames():
    a = np.full((2, 2), 2.0, dtype=np.float32)
    b = np.full((2, 2), 4.0, dtype=np.float32)
    result = integrate_frames([a, b], method="sum")
    assert np.allclose(result, 6.0)


def test_single_frame_returns_unchanged_for_all_methods():
    """N=1 must reduce to current single-frame behavior (regression safety)."""
    frame = np.arange(9, dtype=np.float32).reshape(3, 3)
    for method in ("median", "mean", "sum"):
        result = integrate_frames([frame], method=method)
        assert np.allclose(result, frame), method


def test_nan_is_ignored():
    """NaNs in one frame don't poison the integrated result."""
    a = np.array([[1.0, np.nan]], dtype=np.float32)
    b = np.array([[3.0, 5.0]], dtype=np.float32)
    result = integrate_frames([a, b], method="mean")
    assert result[0, 0] == pytest.approx(2.0)
    assert result[0, 1] == pytest.approx(5.0)


def test_output_is_float32():
    frames = [np.ones((2, 2), dtype=np.int16) for _ in range(3)]
    result = integrate_frames(frames, method="median")
    assert result.dtype == np.float32


def test_empty_frame_list_raises():
    with pytest.raises(ValueError):
        integrate_frames([], method="median")


def test_invalid_method_raises():
    frame = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        integrate_frames([frame], method="bogus")


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        integrate_frames([np.ones((2, 2)), np.ones((3, 3))], method="median")


def _outlier_stack():
    """Four noisy frames of a flat 100 DN field with two cosmic-ray hits in different frames."""
    rng = np.random.default_rng(20)
    frames = [100.0 + rng.normal(0.0, 1.0, (4, 4)).astype(np.float32) for _ in range(4)]
    frames[1][2, 2] = 9999.0
    frames[2][1, 1] = 10000.0
    return frames


def test_sigclip_rejects_cosmic_ray_spikes_that_a_plain_mean_keeps():
    """SB-20: the sigma-clipped mean must not move on an outlier; a plain mean does."""
    frames = _outlier_stack()
    plain = integrate_frames(frames, method="mean")
    clipped = integrate_frames(frames, method="sigclip")
    assert plain[2, 2] > 2000.0  # the hit is averaged in
    assert clipped[2, 2] == pytest.approx(100.0, abs=3.0)
    assert clipped[1, 1] == pytest.approx(100.0, abs=3.0)
    # pixels without a hit keep the ordinary mean
    assert clipped[0, 0] == pytest.approx(plain[0, 0], abs=1e-4)


def test_sigclip_single_frame_and_nan():
    frame = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert np.allclose(integrate_frames([frame], method="sigclip"), frame)
    a = np.array([[1.0, np.nan]], dtype=np.float32)
    b = np.array([[3.0, 5.0]], dtype=np.float32)
    result = integrate_frames([a, b], method="sigclip")
    assert result.dtype == np.float32
    assert result[0, 0] == pytest.approx(2.0) and result[0, 1] == pytest.approx(5.0)

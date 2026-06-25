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
        integrate_frames(
            [np.ones((2, 2)), np.ones((3, 3))], method="median"
        )

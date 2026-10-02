"""NrtSettings: defaults reproduce production; the environment selects staging (SB-5)."""
import os

import pytest

from sunback.settings import NrtSettings


def test_defaults_are_production():
    s = NrtSettings.from_env({})
    assert s == NrtSettings()
    assert (s.bucket, s.prefix, s.data_dir) == ("the-sun-now", "", None)
    assert (s.integration_frames, s.integration_method) == (5, "median")
    assert s.write_readable_times is True
    assert s.prefixed("1k/rhef_171_1k.png") == "1k/rhef_171_1k.png"


def test_environment_selects_staging():
    s = NrtSettings.from_env({
        "SUNBACK_BUCKET": "other-bucket",
        "SUNBACK_PREFIX": "staging/",
        "SUNBACK_DATA_DIR": "/tmp/sunback-data",
        "SUNBACK_INTEGRATION_FRAMES": "10",
        "SUNBACK_INTEGRATION_METHOD": "mean",
        "SUNBACK_WRITE_READABLE_TIMES": "False",
    })
    assert s.bucket == "other-bucket"
    assert s.prefixed("1k/rhef_171_1k.png") == "staging/1k/rhef_171_1k.png"
    assert s.data_dir == "/tmp/sunback-data"
    assert (s.integration_frames, s.integration_method) == (10, "mean")
    assert s.write_readable_times is False


@pytest.mark.parametrize("bad", ["/staging/", "staging", "staging//", "../", "a/../", " staging/"])
def test_bad_prefix_raises(bad):
    with pytest.raises(ValueError, match="SUNBACK_PREFIX"):
        NrtSettings.from_env({"SUNBACK_PREFIX": bad})


@pytest.mark.parametrize("good", ["", "staging/", "staging/run-42/", "sb_5.test/"])
def test_good_prefix_accepted(good):
    assert NrtSettings.from_env({"SUNBACK_PREFIX": good}).prefix == good


def test_bad_method_and_frames_raise():
    with pytest.raises(ValueError, match="SUNBACK_INTEGRATION_METHOD"):
        NrtSettings.from_env({"SUNBACK_INTEGRATION_METHOD": "mode"})
    with pytest.raises(ValueError, match="SUNBACK_INTEGRATION_FRAMES"):
        NrtSettings.from_env({"SUNBACK_INTEGRATION_FRAMES": "0"})


def test_settings_are_frozen():
    with pytest.raises(AttributeError):
        NrtSettings().prefix = "staging/"


def test_from_env_reads_os_environ(monkeypatch):
    monkeypatch.setenv("SUNBACK_PREFIX", "staging/")
    assert NrtSettings.from_env().prefix == "staging/"

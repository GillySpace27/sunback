"""Packaging invariants (SB-12).

One metadata source (pyproject.toml), one version read at run time through
importlib.metadata, an explicit package list, and the IDL colour table loaded
from inside the package so an installed wheel needs no data/ directory.
"""

import importlib.metadata
import pathlib
import subprocess
import sys
import tomllib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_version_lives_only_in_pyproject():
    assert "version=" not in (ROOT / "setup.py").read_text().replace(" ", ""), "setup.py still carries a version"
    for moved in ("setup.cfg", "versioneer.py", "sunback/versioneer.py", "sunback/_version.py", "sunback/setup.cfg"):
        assert not (ROOT / moved).exists(), f"{moved} should be in attic/packaging/"


def test_runtime_version_comes_from_metadata():
    import sunback

    try:
        expected = importlib.metadata.version("sunback")
    except importlib.metadata.PackageNotFoundError:
        expected = "0+unknown"
    assert sunback.__version__ == expected


def test_pyproject_declares_what_the_client_imports():
    project = _pyproject()["project"]
    names = {d.split(";")[0].split("=")[0].split(">")[0].split("<")[0].strip() for d in project["dependencies"]}
    assert {"numpy", "tqdm"} <= names, sorted(names)
    assert project["requires-python"] == ">=3.11"


def test_wheel_package_list_is_explicit():
    st = _pyproject()["tool"]["setuptools"]
    assert st.get("include-package-data") is False
    assert st["packages"]["find"]["include"] == ["sunback", "sunback.*"]
    assert st["package-data"] == {"sunback.science": ["idl_3.csv"]}


def test_colour_table_has_no_home_directory_fallback():
    live = [
        ln
        for ln in (ROOT / "sunback/science/color_tables.py").read_text().splitlines()
        if "/Users/" in ln and not ln.lstrip().startswith("#")
    ]
    assert live == []


def test_colour_table_loads_without_the_repository(tmp_path):
    """Import color_tables from a copy of the package with no data/ directory two
    levels up: the layout of an installed wheel."""
    pkg = tmp_path / "sunback"
    (pkg / "science").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    for name in ("__init__.py", "color_tables.py", "idl_3.csv"):
        (pkg / "science" / name).write_bytes((ROOT / "sunback" / "science" / name).read_bytes())
    code = "import numpy, sunback.science.color_tables as c; numpy.save('idl_3.npy', c.idl_3)"
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    expected = np.loadtxt(ROOT / "data" / "idl_3.csv", delimiter=",")
    assert np.array_equal(np.load(tmp_path / "idl_3.npy"), expected)


def test_sunpy_is_declared_with_the_map_extra():
    """sunpy 7 imports reproject and mpl-animators from sunpy.map; the client path imports sunpy.map
    (Processor.py), so a bare "sunpy" fails in a clean venv with ModuleNotFoundError: reproject."""
    deps = _pyproject()["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("sunpy[map]") for d in deps), deps

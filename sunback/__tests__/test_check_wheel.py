"""devtools/scripts/check_wheel.py rejects what the 0.6.17 wheel shipped (SB-12)."""

import json
import pathlib
import subprocess
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "devtools" / "scripts" / "check_wheel.py"
CLEAN = {
    "sunback/__init__.py": b"",
    "sunback/science/idl_3.csv": b"0,0,0\n",
    "sunback-0.0.0.dist-info/METADATA": b"Name: sunback\n",
}


def _wheel(tmp_path, files):
    path = tmp_path / "sunback-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_clean_wheel_passes(tmp_path):
    r = _run(_wheel(tmp_path, CLEAN), "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads(r.stdout)
    assert report["status"] == "PASS"
    assert report["wheels"][0]["entries"] == 3


@pytest.mark.parametrize(
    "name,size,why",
    [
        ("data/idl_3.csv", 6, "top-level entry outside sunback/"),
        ("sunback/openh264-1.8.0-win64.dll", 2, "banned file type"),
        ("sunback/run/sunback_netcdf_notebook.ipynb", 2, "banned file type"),
        ("sunback/run/run_server_lingon.timestamp", 1, "banned file type"),
        ("sunback/big.bin", 1_000_001, "file over 1 MB"),
    ],
    ids=["top-level", "dll", "ipynb", "timestamp", "over-1MB"],
)
def test_each_rule_fails(tmp_path, name, size, why):
    r = _run(_wheel(tmp_path, {**CLEAN, name: b"x" * size}), "--json")
    assert r.returncode == 1, r.stdout + r.stderr
    problems = json.loads(r.stdout)["wheels"][0]["problems"]
    assert any(p.startswith(why) and name in p for p in problems), problems


def test_unexpanded_glob_is_unchecked(tmp_path):
    r = _run(tmp_path / "dist" / "*.whl")
    assert r.returncode == 3, r.stdout + r.stderr
    assert r.stdout.startswith("UNCHECKED: wheel")

"""SU-13: one Python RHEF build across sunback, the store and the reducer.

The canonical sunkit-image build is commit 41e89767fc8dbd0c3799d2e3aaaf87419c67fdfd of
https://github.com/GillySpace27/sunkit-image (fork branch solar-archive-fast-rhef). The store's
vendored wheel is that commit with the bundled test-data FITS stripped.
No tag or release marks the build (Gilly, 2026-10-12, option c): the commit sha and the
version string are its identity.

Always runs: the constants are well formed, pyproject.toml declares the build in its
`rhef-canonical` dependency group, requirements-exact.txt carries the SU-13 comment.
Skips, with the reason in the output, unless the environment can answer:
  installed build   needs sunkit-image installed. SUNBACK_REQUIRE_RHEF_BUILD=1 turns a
                    different installed build into a failure (use it in the reducer image job).
  wheel sha256      needs SUNKIT_WHEEL_PATH=<path to the wheel>.
  store constants   needs SUNBACK_WEBAPP_DIR=<store checkout>; compares scripts/smoke_rhef.py.
  golden vectors    needs the RH-6 runner and bundle in sunback/processor/tests/, and
                    SUNBACK_RHEF_GOLDEN_ENFORCE=1: SU-8's enforce mode is withheld (Gilly,
                    2026-10-12). On 2026-10-13 the canonical build gave pass=3 fail=6 against
                    the sunkit-0.7 bundle (computed, enforce mode), so this stays opt-in.
RHEF output is a visualization, not a calibrated radiance.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

CANONICAL_FORK = "https://github.com/GillySpace27/sunkit-image.git"
CANONICAL_COMMIT = "41e89767fc8dbd0c3799d2e3aaaf87419c67fdfd"
EXPECTED_SUNKIT_VERSION = "0.1.dev308+g41e89767f"
# Copied from sunback_webapp scripts/smoke_rhef.py WHEEL_SHA256 (RH-9: the deterministic rebuild).
EXPECTED_WHEEL_SHA256 = "df3123a4e0a0214324045febac239c28003f1bd6ab202e6ea77baf841fd35a41"
# The earlier hand-built wheel, still in the store's vendor/ as RH-9's rollback. Same .py files.
PRE_RH9_WHEEL_SHA256 = "8f8bec0b17dda8fc5d2d27f1a4522d83307de968cc9f9a26553a5dcfd0b6981b"

MARKER = "# SU-13 canonical RHEF build: "


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_sunkit_version():
    try:
        import sunkit_image
    except ImportError:
        return None
    return sunkit_image.__version__


def test_constants_are_well_formed():
    assert re.fullmatch(r"[0-9a-f]{40}", CANONICAL_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{64}", EXPECTED_WHEEL_SHA256)
    assert re.fullmatch(r"[0-9a-f]{64}", PRE_RH9_WHEEL_SHA256)
    assert EXPECTED_SUNKIT_VERSION.endswith("+g" + CANONICAL_COMMIT[:9])


def test_sha256_of_matches_a_known_digest(tmp_path):
    f = tmp_path / "abc.bin"
    f.write_bytes(b"abc")
    assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_pyproject_declares_the_canonical_build():
    tomllib = pytest.importorskip("tomllib", reason="needs Python 3.11 or later")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = data.get("dependency-groups", {})
    assert "rhef-canonical" in groups, "pyproject.toml has no [dependency-groups] rhef-canonical (SU-13)"
    assert groups["rhef-canonical"] == [f"sunkit-image @ git+{CANONICAL_FORK}@{CANONICAL_COMMIT}"]
    # PyPI rejects an upload whose metadata carries a direct reference, so none may sit in
    # the dependencies or any extra that the wheel publishes.
    published = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        published += extra
    direct = [d for d in published if "@" in d or "git+" in d]
    assert not direct, f"direct reference in published metadata: {direct}"


def test_requirements_exact_marks_the_canonical_build():
    lines = (ROOT / "requirements-exact.txt").read_text(encoding="utf-8").splitlines()
    marks = [i for i, line in enumerate(lines) if line.startswith(MARKER)]
    assert len(marks) == 1, f"expected one line starting {MARKER!r}, found {len(marks)}"
    assert CANONICAL_COMMIT in lines[marks[0]]
    assert "sunkit-image" in lines[marks[0] + 1], "the line below the marker must be the sunkit-image pin"
    # The rhe-bugfix pin is kept (live before RH-9, a comment after it): nothing is deleted.
    assert any("sunkit-image.git@rhe-bugfix#egg=sunkit-image" in line for line in lines)


def test_installed_build_is_canonical():
    got = installed_sunkit_version()
    if got is None:
        pytest.skip("sunkit-image is not installed in this environment")
    if got != EXPECTED_SUNKIT_VERSION:
        msg = f"installed sunkit-image {got} is not the canonical {EXPECTED_SUNKIT_VERSION}"
        if os.environ.get("SUNBACK_REQUIRE_RHEF_BUILD") == "1":
            pytest.fail(msg)
        pytest.skip(msg + " (set SUNBACK_REQUIRE_RHEF_BUILD=1 to make this a failure)")


def test_wheel_sha256_matches_when_a_wheel_is_given():
    path = os.environ.get("SUNKIT_WHEEL_PATH")
    if not path:
        pytest.skip("set SUNKIT_WHEEL_PATH to a copy of the canonical wheel to check its sha256")
    got = sha256_of(path)
    assert got in (EXPECTED_WHEEL_SHA256, PRE_RH9_WHEEL_SHA256), (
        f"{path} has sha256 {got}; expected {EXPECTED_WHEEL_SHA256} (RH-9 rebuild) "
        f"or {PRE_RH9_WHEEL_SHA256} (earlier hand-built wheel)")


def test_store_smoke_check_agrees_when_a_checkout_is_given():
    root = os.environ.get("SUNBACK_WEBAPP_DIR")
    if not root:
        pytest.skip("set SUNBACK_WEBAPP_DIR to a store checkout to compare scripts/smoke_rhef.py")
    smoke = pathlib.Path(root) / "scripts" / "smoke_rhef.py"
    assert smoke.is_file(), f"{smoke} not found; RH-9 has not landed in that checkout"
    text = smoke.read_text(encoding="utf-8")
    sha = re.search(r'^WHEEL_SHA256 = "([0-9a-f]{64})"', text, re.M)
    ver = re.search(r'^SUNKIT_VERSION = "([^"]+)"', text, re.M)
    assert sha and ver, "smoke_rhef.py has no WHEEL_SHA256 / SUNKIT_VERSION constants"
    assert (ver.group(1), sha.group(1)) == (EXPECTED_SUNKIT_VERSION, EXPECTED_WHEEL_SHA256), (
        f"store smoke check says {ver.group(1)} / {sha.group(1)}; this test says "
        f"{EXPECTED_SUNKIT_VERSION} / {EXPECTED_WHEEL_SHA256}; change both together")


def test_golden_vectors_on_the_installed_build():
    if os.environ.get("SUNBACK_RHEF_GOLDEN_ENFORCE") != "1":
        pytest.skip("SU-8 enforce mode is withheld; set SUNBACK_RHEF_GOLDEN_ENFORCE=1 to run the golden vectors")
    if installed_sunkit_version() is None:
        pytest.skip("sunkit-image is not installed in this environment")
    runner = ROOT / "sunback" / "processor" / "tests" / "test_rhef_conformance.py"
    golden = pathlib.Path(os.environ.get("RHEF_GOLDEN_DIR") or runner.parent / "golden")
    if not runner.is_file() or not (golden / "manifest.json").is_file():
        pytest.skip("RH-6 runner or golden bundle not present (sunback/processor/tests/golden)")
    env = dict(os.environ, RHEF_CONFORMANCE_MODE="enforce")
    run = subprocess.run(
        [sys.executable, "-m", "unittest", "sunback.processor.tests.test_rhef_conformance"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    m = re.search(r"summary pass=(\d+) report=(\d+) fail=(\d+) mode=enforce", run.stdout)
    assert m, f"no summary line (exit {run.returncode})\n{run.stdout[-800:]}\n{run.stderr[-800:]}"
    assert int(m.group(1)) > 0 and int(m.group(3)) == 0 and run.returncode == 0, (
        f"golden vectors on sunkit-image {installed_sunkit_version()}: {m.group(0)} (exit {run.returncode})")

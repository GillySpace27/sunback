"""The reducer image is defined by requirements-reducer.txt and pinned by digest (SB-13).

test_repository_invariants checks this checkout and is skipped until the lock
exists (it is frozen from the production image, which waits on Gilly's dispatch
of container.yml). The other tests run devtools/scripts/reducer_lock.py on a
synthetic freeze in a copy of the files it edits, so the chain freeze -> lock ->
Dockerfile -> pins is tested now.
"""

import shutil

import pytest

from devtools.scripts import reducer_lock as rl

ROOT = rl.ROOT
OLD_DIGEST = "ghcr.io/gillyspace27/sunback-ci@sha256:" + "a" * 64
NEW_DIGEST = "ghcr.io/gillyspace27/sunback-ci@sha256:" + "b" * 64
BASE = "sha256:" + "c" * 64
FREEZE = """\
astropy==7.1.0
boto3==1.40.0
matplotlib==3.10.0
numpy==2.3.0
opencv-python==4.12.0.88
pip==25.2
requests==2.33.0
scipy==1.16.0
setuptools==80.9.0
sunback==0.9.0
sunkit-image @ git+https://github.com/sunpy/sunkit-image.git@{sha}
sunpy==7.0.5
tqdm==4.67.1
xarray==2025.9.0
""".format(sha="d" * 40)


def _freeze(tmp_path, pip_freeze=FREEZE, pip_check=rl.CLEAN_PIP_CHECK):
    d = tmp_path / "freeze"
    d.mkdir()
    (d / "image-digest.txt").write_text(OLD_DIGEST + "\n")
    (d / "python-version.txt").write_text("Python 3.12.11\n")
    (d / "pip-freeze.txt").write_text(pip_freeze)
    (d / "ffmpeg-version.txt").write_text("ffmpeg version 5.1.7-0+deb12u1 Copyright (c) 2000-2025\n")
    (d / "ffmpeg-deb-version.txt").write_text("7:5.1.7-0+deb12u1\n")
    (d / "pip-check.txt").write_text(pip_check + "\n")
    return d


def _repo(tmp_path):
    """A copy of the files reducer_lock.py reads and edits, as they are on this branch."""
    root = tmp_path / "repo"
    (root / ".github" / "workflows").mkdir(parents=True)
    for name in ("Dockerfile", "pyproject.toml", *rl.INSTALLS):
        shutil.copy(ROOT / name, root / name)
    return root


def _run_chain(tmp_path, **freeze_kw):
    root, freeze = _repo(tmp_path), _freeze(tmp_path, **freeze_kw)
    assert rl.main(["--root", str(root), "make-lock", str(freeze)]) == 0
    assert rl.main(["--root", str(root), "dockerfile", str(freeze), "--base-digest", BASE]) == 0
    assert rl.main(["--root", str(root), "pin", OLD_DIGEST]) == 0
    return root


def test_repository_invariants():
    if not (ROOT / rl.LOCK_NAME).exists():
        pytest.skip("requirements-reducer.txt not frozen yet (SB-13 waits on Gilly's freeze dispatch)")
    assert rl.check(ROOT) == []


def test_check_is_unchecked_without_a_lock(tmp_path, capsys):
    assert rl.main(["--root", str(_repo(tmp_path)), "check"]) == 3
    assert capsys.readouterr().out.startswith("UNCHECKED: ")


def test_lock_alone_fails_the_check(tmp_path):
    root = _repo(tmp_path)
    assert rl.main(["--root", str(root), "make-lock", str(_freeze(tmp_path))]) == 0
    problems = rl.check(root)
    assert any(p.startswith("Dockerfile: FROM") for p in problems), problems
    assert any("GitCloudRunHourly.yml: expected one image line pinned by digest" in p for p in problems), problems
    assert any("tests.yml: an image line still follows :latest" in p for p in problems), problems
    assert not any(p.startswith(rl.LOCK_NAME) for p in problems), problems


def test_full_chain_passes_the_check(tmp_path, capsys):
    root = _run_chain(tmp_path)
    assert rl.check(root) == []
    assert rl.main(["--root", str(root), "check"]) == 0
    lock = (root / rl.LOCK_NAME).read_text()
    assert f"# source-image: {OLD_DIGEST}\n" in lock
    assert "sunback==" not in lock
    docker = (root / "Dockerfile").read_text()
    assert f"FROM python:3.12.11-slim@{BASE}\n" in docker
    assert 'ARG FFMPEG_DEB_VERSION=7:5.1.7-0+deb12u1\n' in docker
    assert "\nRUN pip check\n" in docker
    assert "# FROM python:3.12-slim\n" in docker
    for name in rl.INSTALLS:
        text = (root / name).read_text()
        assert text.count(f"      image: {OLD_DIGEST}\n") == 1, name
        assert "was ghcr.io/gillyspace27/sunback-ci:latest" in text, name


def test_digest_switch_moves_both_workflows(tmp_path):
    root = _run_chain(tmp_path)
    assert rl.pin(root, NEW_DIGEST, today="2026-10-05") == [
        f"{name}: {OLD_DIGEST} -> {NEW_DIGEST}" for name in rl.INSTALLS
    ]
    assert rl.check(root) == []
    with pytest.raises(rl.LockError, match="already uses"):
        rl.pin(root, NEW_DIGEST)


def test_pin_refuses_a_tag_and_writes_nothing_on_a_bad_file(tmp_path):
    root = _repo(tmp_path)
    with pytest.raises(rl.LockError):
        rl.pin(root, "ghcr.io/gillyspace27/sunback-ci:latest")
    tests_yml = root / ".github/workflows/tests.yml"
    tests_yml.write_text(tests_yml.read_text().replace("      image: ghcr.io", "      image:  ghcr.io"))
    before = (root / ".github/workflows/GitCloudRunHourly.yml").read_text()
    with pytest.raises(rl.LockError, match="tests.yml: expected one sunback-ci image line"):
        rl.pin(root, OLD_DIGEST)
    assert (root / ".github/workflows/GitCloudRunHourly.yml").read_text() == before


@pytest.mark.parametrize("bad", [
    "-e git+https://github.com/o/r.git@" + "e" * 40 + "#egg=thing",
    "thing @ file:///tmp/thing-1.0-py3-none-any.whl",
    "thing @ git+https://github.com/o/r.git@main",
])
def test_make_lock_refuses_what_an_index_cannot_reproduce(tmp_path, bad):
    with pytest.raises(rl.LockError, match="cannot lock"):
        rl.make_lock(_freeze(tmp_path, pip_freeze=FREEZE + bad + "\n"))


def test_a_runtime_dependency_missing_from_the_lock_is_reported(tmp_path):
    root = _run_chain(tmp_path, pip_freeze=FREEZE.replace("tqdm==4.67.1\n", ""))
    assert rl.check(root) == [f"{rl.LOCK_NAME}: pip install --no-deps . would leave these out: ['tqdm']"]


def test_a_broken_frozen_image_keeps_pip_check_as_a_comment(tmp_path):
    root = _run_chain(tmp_path, pip_check="astropy 7.1.0 has requirement numpy<2, but you have numpy 2.3.0.")
    docker = (root / "Dockerfile").read_text()
    assert "\nRUN pip check\n" not in docker
    assert "# RUN pip check  (SB-13: the frozen image reports: astropy 7.1.0 has requirement" in docker

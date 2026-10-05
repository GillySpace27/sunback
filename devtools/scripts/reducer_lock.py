#!/usr/bin/env python3
"""The reducer image's lock, Dockerfile and digest pins, written from a freeze (SB-13).

The freeze is the artifact "freeze" of .github/workflows/container.yml (job
`freeze`, dispatched by Gilly): image-digest.txt, python-version.txt,
pip-freeze.txt, ffmpeg-version.txt, ffmpeg-deb-version.txt and pip-check.txt.
These subcommands turn it into the three changes of SB-13 Tasks 3 to 5. They
only edit files in the checkout; nothing here pulls, pushes or builds an image.

  make-lock FREEZE_DIR        write requirements-reducer.txt (Task 3)
  dockerfile FREEZE_DIR --base-digest sha256:<64 hex>
                              rewrite the Dockerfile to install the lock with
                              --no-deps on python:3.12.<n>-slim@<digest>, the old
                              lines kept as comments (Task 4)
  pin IMAGE@sha256:<64 hex>   point GitCloudRunHourly.yml and tests.yml at one
                              image digest and install sunback with --no-deps
                              (Task 5; also the later digest switch, Task 11)
  check                       the invariants test_reducer_lock.py asserts; exit 0
                              when all hold, 1 when one fails, 3 (UNCHECKED) when
                              requirements-reducer.txt does not exist yet

Every subcommand takes --root (default: this checkout). Edits are exact-match
patches that stop with an error naming the file when a match count is not 1.
Runbook: docs/CONTAINER.md.
"""

import argparse
import datetime
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCK_NAME = "requirements-reducer.txt"
IMAGE_REPO = "ghcr.io/gillyspace27/sunback-ci"
DIGEST_RE = r"ghcr\.io/gillyspace27/sunback-ci@sha256:[0-9a-f]{64}"
PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==\S+$")
VCS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]* @ git\+https://\S+@[0-9a-f]{40}$")
CLEAN_PIP_CHECK = "No broken requirements found."
# Install lines as the two workflows carry them today (exact text, patched by pin).
INSTALLS = {
    ".github/workflows/GitCloudRunHourly.yml": "        run: pip install .\n",
    ".github/workflows/tests.yml": "          pip install .\n",
}
IMAGE_LINE = re.compile(r"^      image: (ghcr\.io/gillyspace27/sunback-ci(?::latest|@sha256:[0-9a-f]{64}))$", re.M)


class LockError(Exception):
    """A freeze or a file this script cannot use as it is."""


def normalize(name):
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _name(line):
    return normalize(re.split(r"[ =@\[]", line, maxsplit=1)[0])


def lock_body(text):
    """Requirement lines of a lock (comments and blank lines skipped)."""
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def _read(freeze, name):
    path = pathlib.Path(freeze) / name
    if not path.exists():
        raise LockError(f"the freeze has no {name} ({path})")
    return path.read_text(encoding="utf-8").strip()


def make_lock(freeze, today=None):
    """Return the text of requirements-reducer.txt for a freeze directory."""
    digest = _read(freeze, "image-digest.txt")
    python = _read(freeze, "python-version.txt")
    ffmpeg = _read(freeze, "ffmpeg-version.txt")
    ffmpeg_deb = _read(freeze, "ffmpeg-deb-version.txt")
    if not re.fullmatch(DIGEST_RE, digest):
        raise LockError(f"image-digest.txt is not {IMAGE_REPO}@sha256:<64 hex>: {digest!r}")
    if not re.fullmatch(r"Python 3\.12\.\d+", python):
        raise LockError(f"python-version.txt is not Python 3.12.<n>: {python!r}")
    if not ffmpeg.startswith("ffmpeg version "):
        raise LockError(f"ffmpeg-version.txt does not start with 'ffmpeg version ': {ffmpeg!r}")
    if not ffmpeg_deb or "\n" in ffmpeg_deb:
        raise LockError(f"ffmpeg-deb-version.txt is not one Debian version: {ffmpeg_deb!r}")

    body, sunkit = [], None
    for line in _read(freeze, "pip-freeze.txt").splitlines():
        line = line.strip()
        if not line:
            continue
        if _name(line) == "sunback":
            continue  # installed from the checkout at run time with --no-deps
        if not (PIN.match(line) or VCS.match(line)):
            raise LockError(f"cannot lock this freeze line (editable, local or unpinned): {line!r}")
        if _name(line) == "sunkit-image":
            sunkit = line
        body.append(line)
    if sunkit is None:
        raise LockError("sunkit-image is not in the freeze")

    today = today or datetime.date.today().isoformat()
    lineage = (f"{sunkit} (as frozen; installed by name from requirements-server.txt unless the line is a "
               "git pin). Which RHEF lineage is canonical awaits Gilly (open question Q6, RH-9).")
    header = [
        "# requirements-reducer.txt: the reducer image's complete Python environment (SB-13). Canonical.",
        "# Frozen once from the hand-built image below; since then changed only by reviewed PRs that",
        "# pass .github/workflows/container.yml. The Dockerfile installs it with",
        "# pip install --no-deps -r, so every package is listed, pip itself included. docs/CONTAINER.md.",
        f"# source-image: {digest}",
        f"# frozen: {today}",
        f"# ffmpeg: {ffmpeg}",
        f"# ffmpeg-deb: {ffmpeg_deb}",
        f"# python: {python}",
        f"# sunkit-image: {lineage}",
    ]
    return "\n".join(header + body) + "\n"


def make_dockerfile(old, freeze, base_digest, today=None):
    """Return the new Dockerfile text; the old file is kept below it as comments."""
    python = _read(freeze, "python-version.txt")
    ffmpeg_deb = _read(freeze, "ffmpeg-deb-version.txt")
    pip_check = _read(freeze, "pip-check.txt")
    match = re.fullmatch(r"Python (3\.12\.\d+)", python)
    if not match:
        raise LockError(f"python-version.txt is not Python 3.12.<n>: {python!r}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_digest):
        raise LockError(f"--base-digest is not sha256:<64 hex>: {base_digest!r}")
    if not old.startswith("FROM python:3.12-slim\n") or "pip install -r requirements-server.txt" not in old:
        raise LockError("Dockerfile moved on since SB-13 was written; patch it by hand")
    today = today or datetime.date.today().isoformat()
    if pip_check == CLEAN_PIP_CHECK:
        check = "RUN pip check"
    else:
        first = pip_check.splitlines()[0]
        check = f"# RUN pip check  (SB-13: the frozen image reports: {first})"
    kept = "\n".join("# " + ln if ln else "#" for ln in old.rstrip("\n").split("\n"))
    return (
        "# Reducer image (SB-13): built only by .github/workflows/container.yml, never by hand.\n"
        "# Base pinned by digest; Python packages exactly as requirements-reducer.txt (a freeze of\n"
        "# the hand-built image, see its header); apt ffmpeg pinned to the Debian version recorded\n"
        "# there. How to bump or roll back: docs/CONTAINER.md.\n"
        f"FROM python:{match.group(1)}-slim@{base_digest}\n"
        "\n"
        f"ARG FFMPEG_DEB_VERSION={ffmpeg_deb}\n"
        'RUN apt-get update && apt-get install -y "ffmpeg=${FFMPEG_DEB_VERSION}"\n'
        "\n"
        "COPY requirements-reducer.txt requirements-reducer.txt\n"
        "RUN pip install --no-cache-dir --no-deps -r requirements-reducer.txt\n"
        f"{check}\n"
        "\n"
        "WORKDIR /app\n"
        'CMD ["python3"]\n'
        "LABEL org.opencontainers.image.source=https://github.com/GillySpace27/sunback\n"
        "\n"
        f"# Until {today} this file read (kept for reference, SB-13):\n"
        f"{kept}\n"
    )


def pin(root, ref, today=None):
    """Point both workflows at one image digest and install sunback with --no-deps.

    Returns a list of "<file>: <old> -> <new>" lines.
    """
    if not re.fullmatch(DIGEST_RE, ref):
        raise LockError(f"not {IMAGE_REPO}@sha256:<64 hex>: {ref!r}")
    today = today or datetime.date.today().isoformat()
    new_texts, report = {}, []
    for name, install in INSTALLS.items():
        path = pathlib.Path(root) / name
        text = path.read_text(encoding="utf-8")
        found = IMAGE_LINE.findall(text)
        if len(found) != 1:
            raise LockError(f"{name}: expected one sunback-ci image line, found {len(found)}")
        old = found[0]
        if old == ref:
            raise LockError(f"{name} already uses {ref}")
        text = IMAGE_LINE.sub(lambda _m: f"      # SB-13, {today}: image pinned by digest; was {old}. "
                                         f"docs/CONTAINER.md\n      image: {ref}", text)
        count = text.count(install)
        if count > 1:
            raise LockError(f"{name}: expected one {install.strip()!r} line, found {count}")
        if count == 1:
            text = text.replace(install, install.replace("pip install .", "pip install --no-deps ."))
        if "pip install --no-deps .\n" not in text:
            raise LockError(f"{name}: no 'pip install .' line to change and no --no-deps line")
        new_texts[path] = text
        report.append(f"{name}: {old} -> {ref}")
    for path, text in new_texts.items():  # write only after both files patched cleanly
        path.write_text(text, encoding="utf-8")
    return report


def check(root):
    """Invariants of a finished SB-13 change. Returns a list of problems (empty when all hold).

    Raises FileNotFoundError when the lock does not exist yet.
    """
    root = pathlib.Path(root)
    lock = root / LOCK_NAME
    text = lock.read_text(encoding="utf-8")
    body = lock_body(text)
    problems = []

    for pattern, label in ((rf"^# source-image: {DIGEST_RE}$", "source-image"),
                           (r"^# frozen: \d{4}-\d{2}-\d{2}$", "frozen"),
                           (r"^# ffmpeg: ffmpeg version \S+", "ffmpeg"),
                           (r"^# sunkit-image: \S+", "sunkit-image")):
        if not re.search(pattern, text, re.M):
            problems.append(f"{LOCK_NAME}: header line '# {label}: ...' missing or malformed")
    for line in body:
        if not (PIN.match(line) or VCS.match(line)):
            problems.append(f"{LOCK_NAME}: not an exact pin: {line!r}")

    names = {_name(ln) for ln in body}
    deps = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    linux = [d for d in deps if "sys_platform == 'darwin'" not in d]
    wanted = {normalize(re.match(r"[A-Za-z0-9._-]+", d).group(0)) for d in linux}
    missing = sorted(wanted - names)
    if missing:
        problems.append(f"{LOCK_NAME}: pip install --no-deps . would leave these out: {missing}")
    if "sunback" in names:
        problems.append(f"{LOCK_NAME}: lists sunback itself")

    docker = (root / "Dockerfile").read_text(encoding="utf-8")
    if not re.search(r"^FROM python:3\.12\.\d+-slim@sha256:[0-9a-f]{64}$", docker, re.M):
        problems.append("Dockerfile: FROM is not python:3.12.<n>-slim@sha256:<64 hex>")
    if not re.search(r"^RUN pip install --no-cache-dir --no-deps -r requirements-reducer\.txt$", docker, re.M):
        problems.append("Dockerfile: no 'RUN pip install --no-cache-dir --no-deps -r requirements-reducer.txt'")
    if re.search(r"^RUN .*requirements(-server)?\.txt", docker, re.M):
        problems.append("Dockerfile: still installs requirements.txt or requirements-server.txt")

    digests = {}
    for name in INSTALLS:
        wf = (root / name).read_text(encoding="utf-8")
        found = re.findall(rf"^      image: ({DIGEST_RE})$", wf, re.M)
        if len(found) != 1:
            problems.append(f"{name}: expected one image line pinned by digest, found {len(found)}")
        else:
            digests[name] = found[0]
        if re.search(r"^\s*image: .*:latest", wf, re.M):
            problems.append(f"{name}: an image line still follows :latest")
        if not re.search(r"pip install --no-deps \.$", wf, re.M):
            problems.append(f"{name}: sunback is not installed with 'pip install --no-deps .'")
    if len(set(digests.values())) > 1:
        problems.append(f"the workflows pin different digests: {digests}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reducer lock, Dockerfile and digest pins from a freeze (SB-13).")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT, help="repository root (default: this checkout)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-lock", help="write requirements-reducer.txt from a freeze directory")
    p.add_argument("freeze", type=pathlib.Path)
    p = sub.add_parser("dockerfile", help="rewrite the Dockerfile to install the lock on a pinned base")
    p.add_argument("freeze", type=pathlib.Path)
    p.add_argument("--base-digest", required=True, help="sha256:<64 hex> of python:3.12.<n>-slim")
    p = sub.add_parser("pin", help="pin both workflows to one image digest; install sunback with --no-deps")
    p.add_argument("ref", help=f"{IMAGE_REPO}@sha256:<64 hex>")
    sub.add_parser("check", help="the SB-13 invariants; exit 0 ok, 1 a problem, 3 no lock yet")
    args = parser.parse_args(argv)
    root = args.root

    try:
        if args.cmd == "make-lock":
            text = make_lock(args.freeze)
            (root / LOCK_NAME).write_text(text, encoding="utf-8")
            print(f"{LOCK_NAME}: {len(lock_body(text))} pinned lines")
        elif args.cmd == "dockerfile":
            path = root / "Dockerfile"
            path.write_text(make_dockerfile(path.read_text(encoding="utf-8"), args.freeze, args.base_digest),
                            encoding="utf-8")
            print("Dockerfile rewritten")
        elif args.cmd == "pin":
            for line in pin(root, args.ref):
                print(line)
        else:
            if not (root / LOCK_NAME).exists():
                print(f"UNCHECKED: {LOCK_NAME} does not exist yet (SB-13 waits on the freeze dispatch)")
                return 3
            problems = check(root)
            for line in problems:
                print(f"FAIL: {line}")
            if problems:
                return 1
            print("PASS: lock, Dockerfile and workflow pins agree")
    except LockError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

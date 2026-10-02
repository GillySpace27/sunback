"""Nothing in sunback/ or aws_lambda/ imports quarantined code (SB-10).

Quarantined means: anything under ``attic``, a ``dep`` or ``depricated``
package, or a module whose name ends in ``_bk``, ``_bkk``, ``_Bkk``,
``_broken`` or ``_working``. The scan reads tracked files (``git ls-files``);
where git is unavailable (a CI checkout without ``.git``) it reads every
``.py`` file on disk under the two directories. It parses, never imports.
Files that are themselves quarantined are not scanned.

GRANDFATHERED lists the import statements that already existed when SB-10
landed, as exact (file, imported module) pairs: research runners that Gilly
runs by hand and a legacy test that pytest does not collect. Any other match
fails the test. Do not add to the list to make a new import pass; move the
code out of quarantine instead (``attic/README.md`` has the restore command).
"""
import ast
import pathlib
import subprocess
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
SCANNED = ("sunback", "aws_lambda")
FORBIDDEN_PARTS = {"attic", "dep", "depricated"}
FORBIDDEN_SUFFIXES = ("_bk", "_bkk", "_Bkk", "_broken", "_working")
GRANDFATHERED = {
    ("sunback/__tests__/test_sunback.py", "movie.dep"),
    ("sunback/run/run_range_multishot_movie.py", "sunback.fetcher.FidoSynopticFetcher_working"),
    ("sunback/run/run_single_PUNCH.py", "sunback.fetcher.FidoSynopticFetcher_working"),
}


def is_quarantined(name):
    return any(p in FORBIDDEN_PARTS or p.endswith(FORBIDDEN_SUFFIXES) for p in name.split("."))


def scanned_files(root):
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", *SCANNED], cwd=root,
                             capture_output=True, check=True).stdout.decode("utf-8")
        paths = [p for p in out.split("\0") if p.endswith(".py")]
    except (OSError, subprocess.CalledProcessError):
        paths = [p.relative_to(root).as_posix() for d in SCANNED for p in (root / d).rglob("*.py")]
    return sorted(paths)


def violations(root, paths):
    """(path, module) for every import statement that names quarantined code."""
    found = []
    for path in paths:
        if is_quarantined(path[:-3].replace("/", ".")):
            continue  # quarantined code importing quarantined code is not the concern here
        with warnings.catch_warnings():  # old escape sequences in legacy files warn on parse
            warnings.simplefilter("ignore")
            tree = ast.parse((root / path).read_text(encoding="utf-8"), filename=path)
        package = ".".join(pathlib.PurePosixPath(path).parent.parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)] if node.level else []
                mod = ".".join(base + ([node.module] if node.module else []))
                names = [mod] + [f"{mod}.{a.name}" for a in node.names if a.name != "*"]
            else:
                continue
            bad = [n for n in names if n and is_quarantined(n)]
            if bad:
                found.append((path, bad[0]))
    return found


def test_no_new_imports_of_quarantined_code():
    new = [v for v in violations(REPO, scanned_files(REPO)) if v not in GRANDFATHERED]
    assert new == [], "quarantined code imported: " + "; ".join(f"{p} imports {m}" for p, m in new)


def test_detector_flags_an_added_attic_import(tmp_path):
    pkg = tmp_path / "sunback" / "putter"
    pkg.mkdir(parents=True)
    (pkg / "probe.py").write_text("import os\nfrom attic.sunback.science import parameters_bkk\n")
    (pkg / "relative.py").write_text("from .dep import old\n")
    assert violations(tmp_path, ["sunback/putter/probe.py", "sunback/putter/relative.py"]) == [
        ("sunback/putter/probe.py", "attic.sunback.science"),
        ("sunback/putter/relative.py", "sunback.putter.dep"),
    ]

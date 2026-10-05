"""Which tracked Python modules can production, the client or research code reach? (SB-10)

Stdlib only. Reads tracked files (``git ls-files``), never imports the code it
scans, writes nothing. Run from anywhere inside the repository:

    python devtools/reachability.py            # three sections with LOC totals
    python devtools/reachability.py --json     # one JSON object on stdout

Scope: tracked ``.py`` files under ``sunback/`` and ``aws_lambda/`` (``attic/``
is never scanned). Roots:

- production: PRODUCTION_ROOTS below (the CI reducer, the wallpaper client, the
  legacy lingon server, the video Lambda and its deploy script);
- research: every other tracked ``.py`` file under ``sunback/run/`` and
  ``sunback/__tests__/``, every tracked ``.py`` outside ``sunback/`` and
  ``aws_lambda/`` (``resources/``, ``docs/``, ``devtools/``, ``setup.py``), and
  the modules imported by code cells of tracked notebooks. A file whose name
  marks it as a backup or deprecated copy (BACKUP_RE) is never a root.

An import is followed when it names a tracked module: absolute, relative, the
same name inside the importing file's package (Python 2 style), or the name
with a ``sunback.`` prefix (pre-2025 layout). Importing ``a.b.c`` also reaches
``a`` and ``a.b``. ``importlib.import_module("x")`` and ``__import__("x")``
with a literal string are followed. Every rule errs towards "reachable", so a
file reported unreachable is a safe candidate for the attic, never proof that
nobody runs it.

Output sets, each a list of ``{"path": str, "loc": int}`` sorted by path:
``reachable`` (from a production root), ``research_only`` (from a research
root only) and ``unreachable``. Exit 0, or 3 when git is unavailable.
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

PRODUCTION_ROOTS = (
    "sunback/run/run_server_github.py",
    "sunback/run/run_client_background.py",
    "sunback/run/run_server_lingon.py",
    "aws_lambda/video_builder/handler.py",
    "aws_lambda/video_builder/deploy.py",
)
SCANNED_TOPS = ("sunback/", "aws_lambda/")
RESEARCH_DIRS = ("sunback/run/", "sunback/__tests__/")
BACKUP_RE = re.compile(
    r"(/dep/|/depricated/|bkk|_bk\.|_broken|_working|_complete|_refactored|Old\.py$|depdep|Scratch|_good\.py$)",
    re.IGNORECASE,
)
SKIP_OUTSIDE = ("attic/", "devtools/scripts/git-filter-repo.py")


def repo_root():
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def module_name(path):
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def loc(root, path):
    with open(root / path, "rb") as fh:
        return sum(1 for _ in fh)


def parse(source, name):
    try:
        return ast.parse(source, filename=name)
    except (SyntaxError, ValueError):
        return None


def notebook_sources(root, path):
    """Code cells of a notebook, with IPython magics and shell lines dropped."""
    try:
        nb = json.loads((root / path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!", "?"))]
        cells.append("\n".join(lines))
    return cells


IMPORT_LINE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w*, ]+)|import\s+([\w., ]+))", re.M)


def imported_names(tree, source, package):
    """Absolute dotted names a file imports (relative imports resolved against ``package``)."""
    names = []
    if tree is None:  # unparsable file (for example Python 2 syntax): fall back to a line regex
        for frm, what, imp in IMPORT_LINE.findall(source):
            if frm:
                names.append(frm)
                names.extend(f"{frm}.{w.strip()}" for w in what.split(",") if w.strip() not in ("", "*"))
            else:
                names.extend(n.strip().split(" as ")[0] for n in imp.split(",") if n.strip())
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".") if package else []
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod:
                names.append(mod)
            names.extend(f"{mod}.{a.name}" if mod else a.name for a in node.names if a.name != "*")
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.append(arg.value)
    return names


def resolve(name, package, modules):
    """Tracked modules reached by importing ``name`` from ``package``."""
    hits = set()
    for candidate in (name, f"{package}.{name}" if package else None, f"sunback.{name}"):
        if not candidate:
            continue
        parts = candidate.split(".")
        found = [".".join(parts[:i]) for i in range(1, len(parts) + 1) if ".".join(parts[:i]) in modules]
        if candidate in modules or (found and found[-1] == ".".join(parts[:-1])):
            hits.update(found)
            break
    return hits


def walk(starts, graph):
    seen, todo = set(), list(starts)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo.extend(graph.get(m, ()))
    return seen


def analyse(root):
    files = tracked_files(root)
    scanned = [p for p in files if p.endswith(".py") and p.startswith(SCANNED_TOPS)]
    modules = {module_name(p): p for p in scanned}
    graph = {}
    for path in scanned:
        mod = module_name(path)
        package = mod if path.endswith("__init__.py") else mod.rpartition(".")[0]
        source = (root / path).read_text(encoding="utf-8", errors="replace")
        deps = set()
        for name in imported_names(parse(source, path), source, package):
            deps |= resolve(name, package, modules)
        graph[mod] = deps

    prod_starts = {module_name(p) for p in PRODUCTION_ROOTS if module_name(p) in modules}
    research_starts = {module_name(p) for p in scanned if p.startswith(RESEARCH_DIRS) and not BACKUP_RE.search(p)}
    for path in files:
        if path.startswith(SKIP_OUTSIDE) or BACKUP_RE.search(path):
            continue
        if path.endswith(".py") and not path.startswith(SCANNED_TOPS):
            source = (root / path).read_text(encoding="utf-8", errors="replace")
            for name in imported_names(parse(source, path), source, ""):
                research_starts |= resolve(name, "", modules)
        elif path.endswith(".ipynb"):
            for cell in notebook_sources(root, path):
                for name in imported_names(parse(cell, path), cell, ""):
                    research_starts |= resolve(name, "", modules)

    prod = walk(prod_starts, graph)
    research = walk(research_starts, graph) - prod

    def rows(mods):
        return [{"path": modules[m], "loc": loc(root, modules[m])} for m in sorted(mods, key=lambda m: modules[m])]

    return {
        "reachable": rows(prod),
        "research_only": rows(research),
        "unreachable": rows(set(modules) - prod - research),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    args = ap.parse_args(argv)
    try:
        root = repo_root()
        result = analyse(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        if args.json:
            print(json.dumps({"status": "UNCHECKED", "error": str(exc)}))
        else:
            print(f"UNCHECKED: reachability ({exc})")
        return 3
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    for key in ("reachable", "research_only", "unreachable"):
        rows = result[key]
        print(f"## {key}: {len(rows)} files, {sum(r['loc'] for r in rows)} LOC")
        for r in rows:
            print(f"{r['loc']:6d}  {r['path']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# attic

Files moved out of the live tree instead of being deleted. Nothing here is
imported, linted or shipped: `pyproject.toml` excludes `attic` from the wheel
(`[tool.setuptools.packages.find]`) and from ruff (`extend-exclude`). To restore
an entry, run its Restore command from the repository root.

| Original path | Moved to | LOC | Reason | Date | Restore |
|---|---|---|---|---|---|
| `__init__.py` | `attic/repo-root__init__.py` | 0 | Empty file at the repository root. It made pytest import the checkout itself as the package `sunback` whenever the checkout directory is named `sunback` (CI `/__w/sunback/sunback`, Gilly's `~/vscode/sunback`), so test collection failed (SB-2). | 2026-10-02 | `git mv attic/repo-root__init__.py __init__.py` |

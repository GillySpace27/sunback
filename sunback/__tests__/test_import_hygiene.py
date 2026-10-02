"""Importing the production modules has no side effects (SB-11).

Each import runs in a fresh interpreter with sockets disabled, no AWS
credentials and HOME pointed at a temporary directory, so a module that
prints, downloads, configures logging or needs AWS at import time fails here.
Warnings are silenced (PYTHONWARNINGS=ignore): compile-time SyntaxWarnings
from legacy string escapes are not output of this code.
"""
import ast
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MODULES = ["sunback.processor.SunPyProcessor", "sunback.putter.AwsPutter", "sunback.putter.DesktopPutter"]
NO_NETWORK = (
    "import socket\n"
    "def _refuse(*args, **kwargs):\n"
    "    raise OSError('network disabled by test_import_hygiene')\n"
    "socket.socket.connect = _refuse\n"
    "socket.socket.connect_ex = _refuse\n"
    "socket.create_connection = _refuse\n"
    "socket.getaddrinfo = _refuse\n"
)


def run_isolated(code, home):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env.update(HOME=str(home), AWS_SHARED_CREDENTIALS_FILE=os.devnull, AWS_CONFIG_FILE=os.devnull,
               AWS_DEFAULT_REGION="us-east-2", MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    return subprocess.run([sys.executable, "-c", NO_NETWORK + code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("module", MODULES)
def test_import_is_silent_offline_without_credentials(module, tmp_path):
    result = run_isolated(f"import {module}\n", tmp_path)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_sunpy_sample_data_is_not_imported(tmp_path):
    result = run_isolated(
        "import sys\nimport sunback.processor.SunPyProcessor\n"
        "print(sorted(m for m in sys.modules if m in ('sunpy.data.sample', 'sunpy.data._sample')))\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_desktop_putter_leaves_root_logging_alone(tmp_path):
    result = run_isolated(
        "import logging\nimport sunback.putter.DesktopPutter\nprint(len(logging.getLogger().handlers))\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


def test_touchup_imports_the_lowercase_rht_package():
    tree = ast.parse((REPO / "sunback" / "processor" / "TouchupProcessor.py").read_text(encoding="utf-8"))
    names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert [n for n in names if "RHT" in n.split(".")] == []
    assert "sunback.utils.rht.rht" in names

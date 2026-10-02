"""lambda_env.json names exactly the environment variables handler.py reads (SB-3).

The scan walks the whole module, so reads inside functions (INTEGRATION_FRAMES and
INTEGRATION_METHOD in _process_one) count, not only top-level assignments.
"""
import ast
import json
import pathlib

VB = pathlib.Path(__file__).resolve().parents[2] / "aws_lambda" / "video_builder"
EXPECTED = {"SUN_BUCKET", "VIDEO_FPS", "FRAME_WINDOW", "GRID_CADENCE_S", "PRUNE_WINDOW_S",
            "BUILD_THROTTLE_S", "FFMPEG_PATH", "X264_PRESET", "INTEGRATION_FRAMES",
            "INTEGRATION_METHOD"}


def env_names_read(source):
    """Names read via os.environ.get("X"), os.getenv("X") or os.environ["X"] anywhere in source."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            f = node.func
            is_environ_get = (isinstance(f, ast.Attribute) and f.attr == "get"
                              and isinstance(f.value, ast.Attribute) and f.value.attr == "environ")
            is_getenv = isinstance(f, ast.Attribute) and f.attr == "getenv"
            if is_environ_get or is_getenv:
                names.add(node.args[0].value)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ" and isinstance(node.slice, ast.Constant)):
            names.add(node.slice.value)
    return names


def _declared():
    return json.loads((VB / "lambda_env.json").read_text())


def test_scanner_sees_reads_inside_functions():
    src = ("import os\n"
           "TOP = os.environ.get('TOP', '1')\n"
           "def f():\n"
           "    return os.environ.get('INSIDE', '1') + os.environ['SUBSCRIPT'] + os.getenv('GETENV', '')\n")
    assert env_names_read(src) == {"TOP", "INSIDE", "SUBSCRIPT", "GETENV"}


def test_handler_reads_the_known_names():
    assert env_names_read((VB / "handler.py").read_text()) == EXPECTED


def test_lambda_env_json_matches_handler_reads():
    declared = set(_declared())
    read = env_names_read((VB / "handler.py").read_text())
    assert declared == read, (f"missing from lambda_env.json: {sorted(read - declared)}; "
                              f"not read by handler.py: {sorted(declared - read)}")


def test_lambda_env_values_are_strings():
    assert all(isinstance(v, str) for v in _declared().values())

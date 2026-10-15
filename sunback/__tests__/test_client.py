"""Wallpaper client 2.0 and sunback-dome (SB-21), offline.

A local http.server stands in for the public bucket. It serves the recaptured
manifest fixture and a small PNG, answers If-None-Match with 304, and records
every path asked for, so a test fails the moment the client asks for the
bucket listing. The platform setters are exercised with fake AppKit modules and
a recorded subprocess.run; that proves the calls they make, not that a real
Windows or Linux desktop changes (UNCHECKED on those platforms).
"""
import ast
import json
import shutil
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from aws_lambda.video_builder import manifest
from sunback.client import __main__ as cli
from sunback.client import dome, fetch, wallpaper

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class Bucket:
    """Objects by key, each with an ETag, plus the log of requested paths."""

    def __init__(self):
        self.objects = {}
        self.paths = []

    def put(self, key, body, etag):
        self.objects[key] = (body, etag)


def _handler(bucket):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            bucket.paths.append(self.path)
            key = self.path.lstrip("/").split("?")[0]
            if key not in bucket.objects:
                body = b"<ListBucketResult></ListBucketResult>" if key == "" else b"no such key"
                self.send_response(200 if key == "" else 404)
                self.end_headers()
                self.wfile.write(body)
                return
            body, etag = bucket.objects[key]
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


@pytest.fixture
def bucket():
    b = Bucket()
    b.put("manifest/171.json", (FIXTURES / "manifest_171.json").read_bytes(), '"m171"')
    b.put("1k/rhef_171_1k.png", PNG, '"p1"')
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(b))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    b.url = f"http://127.0.0.1:{server.server_port}/"
    yield b
    server.shutdown()
    server.server_close()


def is_listing(path):
    return path in ("/", "") or "list-type" in path or path.startswith("/?")


# Catalog and wording bound to their canonical copies


def test_product_ids_are_the_manifest_products():
    assert list(fetch.PRODUCT_IDS) == [p["id"] for p in manifest.PRODUCTS if "kind" not in p]


@pytest.mark.parametrize("pid", fetch.PRODUCT_IDS)
def test_key_builders_agree_with_the_manifest(pid):
    assert fetch.img1k_key(pid) == manifest.img1k_key(pid)
    assert fetch.manifest_key(pid) == manifest.manifest_key(pid)


def test_dome_credit_is_the_putter_credit():
    tree = ast.parse((ROOT / "sunback" / "putter" / "AwsPutter.py").read_text(encoding="utf-8"))
    values = [ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CREDIT"]
    assert values == [dome.CREDIT]


def test_client_imports_only_the_standard_library():
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "import sunback.client.__main__, sunback.client.dome;"
            "print(sorted(m for m in ('numpy', 'requests', 'boto3', 'astropy', 'sunpy', 'matplotlib', 'cv2')"
            " if m in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# Fetch


def test_fetch_reads_the_manifest_and_never_lists(bucket, tmp_path):
    path, when, changed = fetch.fetch_still("171", tmp_path, bucket.url)
    assert changed and Path(path).read_bytes() == PNG
    assert when == json.loads((FIXTURES / "manifest_171.json").read_text())["updated"]
    assert bucket.paths == ["/manifest/171.json", "/1k/rhef_171_1k.png"]
    assert not any(is_listing(p) for p in bucket.paths)


def test_second_fetch_skips_an_unchanged_image(bucket, tmp_path):
    first, _, _ = fetch.fetch_still("171", tmp_path, bucket.url)
    second, _, changed = fetch.fetch_still("171", tmp_path, bucket.url)
    assert (second, changed) == (first, False)


def test_a_new_image_lands_under_the_other_name(bucket, tmp_path):
    first, _, _ = fetch.fetch_still("171", tmp_path, bucket.url)
    bucket.put("1k/rhef_171_1k.png", PNG + b"new", '"p2"')
    second, _, changed = fetch.fetch_still("171", tmp_path, bucket.url)
    assert changed and second != first and Path(second).read_bytes() == PNG + b"new"
    bucket.put("1k/rhef_171_1k.png", PNG + b"newer", '"p3"')
    third, _, _ = fetch.fetch_still("171", tmp_path, bucket.url)
    assert third == first and len(list(tmp_path.glob("*.png"))) == 2


def test_missing_fragment_falls_back_to_the_fixed_key_not_the_listing(bucket, tmp_path):
    bucket.put("1k/rhef_193_1k.png", PNG, '"p193"')
    path, when, changed = fetch.fetch_still("193", tmp_path, bucket.url)
    assert changed and when is None and Path(path).exists()
    assert bucket.paths == ["/manifest/193.json", "/1k/rhef_193_1k.png"]


def test_describe_prints_the_time_and_age():
    now = fetch.parse_time("2026-10-13T03:50:03Z")
    assert fetch.describe("171", "2026-10-13T03:43:03Z", now) == "171 at 03:43 UTC (7 min ago)"
    assert fetch.describe("171", None, now) == "171 (time unknown)"


# Command line


def test_cli_once_downloads_and_a_second_once_skips(bucket, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    set_calls = []
    monkeypatch.setattr(wallpaper, "set_wallpaper", lambda path, screen=None: set_calls.append(path))
    args = ["--product", "171", "--once", "--base-url", bucket.url, "--cache-dir", str(tmp_path / "cache")]
    assert cli.main(args) == 0
    assert cli.main(args) == 0
    assert bucket.paths.count("/1k/rhef_171_1k.png") == 2
    assert len(set_calls) == 2 and set_calls[0] == set_calls[1]  # first pass always sets
    assert "171 at " in capsys.readouterr().out
    assert not (tmp_path / "cfg" / "sunback" / "client.json").exists()  # no chooser when --product is given


def test_download_only_leaves_the_desktop_alone(bucket, tmp_path, monkeypatch):
    monkeypatch.setattr(wallpaper, "set_wallpaper", lambda *a, **k: pytest.fail("desktop touched"))
    assert cli.main(["--product", "171", "--once", "--download-only", "--base-url", bucket.url,
                     "--cache-dir", str(tmp_path)]) == 0


def test_chooser_saves_the_pick(tmp_path):
    answers = iter(["3", "120"])
    cfg = cli.choose(tmp_path / "client.json", ask=lambda prompt: next(answers), out=lambda *a: None)
    assert cfg == {"product": fetch.PRODUCT_IDS[2], "interval": 120}
    assert json.loads((tmp_path / "client.json").read_text()) == cfg


def test_per_screen_plan_cycles_the_channels(monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(wallpaper, "screens", lambda: ["s0", "s1", "s2"])
    args = cli.parse_args(["--per-screen", "171,193"])
    assert cli.plan(args, {}) == [("171", 0), ("193", 1), ("171", 2)]


# Platform setters


@pytest.fixture
def fake_appkit(monkeypatch):
    calls = []

    class Workspace:
        def setDesktopImageURL_forScreen_options_error_(self, url, screen, options, error):
            calls.append((url, screen))
            return True, None

    appkit = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: Workspace()),
        NSScreen=types.SimpleNamespace(screens=lambda: ["left", "right"]),
    )
    foundation = types.SimpleNamespace(NSURL=types.SimpleNamespace(fileURLWithPath_=lambda p: f"file://{p}"))
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)
    return calls


def test_macos_sets_every_screen_or_one(fake_appkit):
    wallpaper.set_wallpaper_macos("/tmp/sun.png")
    assert fake_appkit == [("file:///tmp/sun.png", "left"), ("file:///tmp/sun.png", "right")]
    fake_appkit.clear()
    wallpaper.set_wallpaper_macos("/tmp/sun.png", "right")
    assert fake_appkit == [("file:///tmp/sun.png", "right")]


@pytest.fixture
def recorded_run(monkeypatch):
    calls = []
    monkeypatch.setattr(wallpaper.subprocess, "run", lambda cmd, check: calls.append(cmd))
    return calls


def test_windows_writes_the_registry_then_refreshes(recorded_run):
    wallpaper.set_wallpaper_windows("C:\\sun.png")
    assert recorded_run[0][:3] == ["REG", "ADD", r"HKCU\Control Panel\Desktop"]
    assert recorded_run[0][-1] == "C:\\sun.png"
    assert recorded_run[1] == ["RUNDLL32.EXE", "user32.dll,UpdatePerUserSystemParameters"]


def test_linux_gnome_kde_and_unknown(recorded_run, monkeypatch):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
    wallpaper.set_wallpaper_linux("/tmp/sun.png")
    assert recorded_run == [["gsettings", "set", "org.gnome.desktop.background", "picture-uri",
                             "file:///tmp/sun.png"]]
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    with pytest.raises(NotImplementedError):
        wallpaper.set_wallpaper_linux("/tmp/sun.png")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "")
    with pytest.raises(OSError):
        wallpaper.set_wallpaper_linux("/tmp/sun.png")


def test_desktop_putter_uses_the_shared_setter(monkeypatch, tmp_path):
    from sunback.putter.DesktopPutter import DesktopPutter
    png = tmp_path / "sun.png"
    png.write_bytes(PNG)
    seen = []
    monkeypatch.setattr(wallpaper, "set_wallpaper", lambda path: seen.append(path))
    DesktopPutter.update_background(object(), png)
    assert seen == [str(png.resolve())]


# Dome


def test_dome_filter_without_v360_pads_to_a_square():
    vf = dome.dome_filter(1000, 90, filters={"scale", "pad"})
    assert vf == "scale=500:500,pad=1000:1000:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p"


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")
def test_dome_once_writes_a_square_movie(bucket, tmp_path):
    flat = tmp_path / "flat.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=128x128:rate=4:duration=1", "-pix_fmt", "yuv420p", str(flat)], check=True)
    bucket.put("v/rainbow/20261013T032246.mp4", flat.read_bytes(), '"v1"')
    bucket.put("manifest/index.json", (FIXTURES / "manifest_index.json").read_bytes(), '"i1"')
    out = tmp_path / "dome"
    assert dome.main(["--once", "--out", str(out), "--size", "256", "--base-url", bucket.url]) == 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(out / "dome_rainbow.mp4")],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "256,256"
    assert (out / "dome.m3u").read_text() == "dome_rainbow.mp4\n"
    assert not any(is_listing(p) for p in bucket.paths)

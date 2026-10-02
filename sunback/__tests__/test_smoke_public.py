"""smoke_public.py against a local fake of the public bucket (SB-6). No network."""
import importlib.util
import json
import struct
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from aws_lambda.video_builder import manifest

SCRIPT = Path(__file__).resolve().parents[2] / "devtools" / "scripts" / "smoke_public.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_public", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png(size):
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)


def fake_bucket(thumb_px=512):
    """Key -> (Content-Type, Cache-Control or None, body), shaped like the live bucket."""
    updated = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    objects, fragments = {}, []
    for p in manifest.PRODUCTS:
        pid = p["id"]
        frag = manifest.build_manifest_fragment(
            pid, updated=updated, frame_count=144, integration={"frames": 5, "method": "median"},
            video_v=f"v/{pid}/20261001T120000.mp4", still_v=f"v/{pid}/20261001T122000.png",
            through="2026-10-01T12:00:00Z")
        fragments.append(frag)
        objects[manifest.manifest_key(pid)] = ("application/json", "no-cache", json.dumps(frag).encode())
        objects[frag["img1k"]] = ("image/png", None, _png(1024))
        objects[frag["thumb"]] = ("image/png", None, _png(thumb_px))
        objects[frag["video"]] = ("video/mp4", None, b"mp4")
        objects[frag["video_v"]] = ("video/mp4", manifest.IMMUTABLE, b"mp4")
        objects[frag["still_v"]] = ("image/png", manifest.IMMUTABLE, _png(1024))
    index = manifest.build_index(fragments, updated)
    objects[manifest.INDEX_KEY] = ("application/json", "public, max-age=300", json.dumps(index).encode())
    objects["image_times.txt"] = ("text/plain", None, b"2026-10-01T12:20:00.570")
    return objects


@pytest.fixture
def serve():
    servers = []

    def start(objects):
        class Handler(BaseHTTPRequestHandler):
            def _reply(self, with_body):
                obj = objects.get(self.path.lstrip("/"))
                if obj is None:
                    self.send_response(403)
                    self.end_headers()
                    return
                ctype, cache, body = obj
                rng = self.headers.get("Range")
                if rng and with_body:
                    lo, hi = (int(x) for x in rng.split("=")[1].split("-"))
                    body = body[lo:hi + 1]
                self.send_response(206 if rng and with_body else 200)
                self.send_header("Content-Type", ctype)
                if cache:
                    self.send_header("Cache-Control", cache)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if with_body:
                    self.wfile.write(body)

            def do_GET(self):
                self._reply(True)

            def do_HEAD(self):
                self._reply(False)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/"

    yield start
    for server in servers:
        server.shutdown()


def test_healthy_bucket_exits_0(serve, capsys):
    base = serve(fake_bucket())
    assert _load().main(["--base-url", base], ffprobe="") == 0
    out = capsys.readouterr().out
    assert "PASS: 171 (age " in out
    assert "UNCHECKED: bt709 (ffprobe not on PATH)" in out
    assert out.rstrip().endswith("0 failed, 1 unchecked, exit 0")


def test_missing_prefix_exits_1(serve, capsys):
    base = serve(fake_bucket())
    assert _load().main(["--base-url", base, "--prefix", "no-such-prefix/"], ffprobe="") == 1
    assert "FAIL: 171: fragment HTTP 403" in capsys.readouterr().out


def test_wrong_thumb_size_is_named(serve, capsys):
    base = serve(fake_bucket(thumb_px=256))
    assert _load().main(["--base-url", base], ffprobe="") == 1
    assert "FAIL: 171: thumb is 256x256, expected 512x512" in capsys.readouterr().out


def test_mutable_versioned_copy_is_caught(serve, capsys):
    objects = fake_bucket()
    ctype, _, body = objects["v/rainbow/20261001T120000.mp4"]
    objects["v/rainbow/20261001T120000.mp4"] = (ctype, "no-cache", body)
    assert _load().main(["--base-url", serve(objects), "--json"], ffprobe="") == 1
    result = json.loads(capsys.readouterr().out)
    assert [c["name"] for c in result["checks"] if c["status"] == "FAIL"] == ["v/ (rainbow)"]


def test_unreachable_base_url_exits_3(capsys):
    assert _load().main(["--base-url", "http://127.0.0.1:9/"], ffprobe="") == 3
    assert capsys.readouterr().out.startswith("UNCHECKED: base-url (unreachable: ")

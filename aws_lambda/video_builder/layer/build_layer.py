"""Verify, rebuild and (Gilly only) publish the ffmpeg Lambda layer from ffmpeg.lock (SB-3).

ffmpeg.lock is JSON {"url": str, "sha256": str, "source": "live-layer" | "archive"}.
sha256 is the hex SHA-256 of the ffmpeg binary itself, so a live layer and an
upstream archive compare directly.
  source "archive":    url is a pinned static-build .tar.xz; the binary is the
                       member whose name ends in "/ffmpeg".
  source "live-layer": url is the layer version ARN with the account id written
                       as <ACCOUNT>; the binary is bin/ffmpeg in the layer zip.

  python aws_lambda/video_builder/layer/build_layer.py --verify-only
  python aws_lambda/video_builder/layer/build_layer.py --capture-live
  python aws_lambda/video_builder/layer/build_layer.py --out /tmp/ffmpeg-layer.zip
  python aws_lambda/video_builder/layer/build_layer.py --out /tmp/ffmpeg-layer.zip --publish

--capture-live reads the layer attached to sun-video-builder and writes the lock
(read-only on AWS). --publish publishes a new layer version after the operator
types the layer name; it never attaches the layer to the function.
Exit codes: 0 ok; 1 sha mismatch or refused; 2 declined at the typed gate;
3 source unreachable or no credentials (UNCHECKED).
"""
import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "ffmpeg.lock"
REGION = "us-east-2"
FUNCTION = "sun-video-builder"
LAYER_NAME = "ffmpeg-static"
RUNTIME = "python3.12"
BUCKET = "the-sun-now"
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
DIRECT_UPLOAD_LIMIT = 49_000_000  # deploy.py:67 uses the same cut-over to an S3 upload


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def binary_from_archive(data):
    """The ffmpeg binary from a static-build .tar.xz (member name ends in /ffmpeg)."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith("/ffmpeg"):
                return tf.extractfile(member).read()
    raise ValueError("no */ffmpeg member in the archive")


def binary_from_layer_zip(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read("bin/ffmpeg")


def layer_zip(binary):
    """Deterministic layer zip holding bin/ffmpeg at mode 0o755 (/opt/bin/ffmpeg at runtime)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("bin/ffmpeg", date_time=FIXED_DATE)
        info.external_attr = 0o100755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, binary)
    return buf.getvalue()


def fetch_url(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def _clients():
    import boto3
    return boto3.client("lambda", region_name=REGION), boto3.client("sts")


def fetch_binary(lock, fetch=fetch_url, clients=_clients):
    if lock["source"] == "archive":
        return binary_from_archive(fetch(lock["url"]))
    if lock["source"] == "live-layer":
        lam, sts = clients()
        arn = lock["url"].replace("<ACCOUNT>", sts.get_caller_identity()["Account"])
        location = lam.get_layer_version_by_arn(Arn=arn)["Content"]["Location"]
        return binary_from_layer_zip(fetch(location))
    raise ValueError(f"unknown lock source {lock['source']!r}")


def capture_live(lock_path, fetch=fetch_url, clients=_clients):
    lam, _ = clients()
    layers = lam.get_function_configuration(FunctionName=FUNCTION).get("Layers", [])
    arns = [layer["Arn"] for layer in layers if f":layer:{LAYER_NAME}:" in layer["Arn"]]
    if len(arns) != 1:
        print(f"REFUSED: expected one {LAYER_NAME} layer on {FUNCTION}, found {len(arns)}")
        return 1
    location = lam.get_layer_version_by_arn(Arn=arns[0])["Content"]["Location"]
    binary = binary_from_layer_zip(fetch(location))
    lock = {"url": re.sub(r":\d{12}:", ":<ACCOUNT>:", arns[0]),
            "sha256": sha256_hex(binary), "source": "live-layer"}
    Path(lock_path).write_text(json.dumps(lock, indent=2) + "\n")
    print(f"wrote {lock_path}: {lock['url']} sha256 {lock['sha256']} ({len(binary)} bytes)")
    return 0


def publish(data, sha, input_fn=input, clients=_clients):
    print(f"Type the layer name ({LAYER_NAME}) to publish a new layer version; anything else stops.")
    if input_fn("> ").strip() != LAYER_NAME:
        print("declined: nothing was published")
        return 2
    lam, _ = clients()
    desc = f"static ffmpeg at /opt/bin/ffmpeg, binary sha256 {sha[:12]}"
    if len(data) < DIRECT_UPLOAD_LIMIT:
        content = {"ZipFile": data}
    else:
        import boto3
        key = f"deploy/ffmpeg-layer-{sha[:12]}.zip"  # new key per binary; nothing is overwritten
        boto3.client("s3", region_name=REGION).put_object(Bucket=BUCKET, Key=key, Body=data)
        content = {"S3Bucket": BUCKET, "S3Key": key}
    lv = lam.publish_layer_version(LayerName=LAYER_NAME, Description=desc,
                                   Content=content, CompatibleRuntimes=[RUNTIME])
    print(f"published {re.sub(r':[0-9]{12}:', ':<ACCOUNT>:', lv['LayerVersionArn'])} (not attached to {FUNCTION})")
    return 0


def run(argv=None, fetch=fetch_url, clients=_clients, input_fn=input):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", default=str(LOCK))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--capture-live", action="store_true")
    ap.add_argument("--out", help="write the deterministic layer zip here")
    ap.add_argument("--publish", action="store_true", help="with --out: publish after a typed yes")
    args = ap.parse_args(argv)
    try:
        if args.capture_live:
            return capture_live(args.lock, fetch, clients)
        lock = json.loads(Path(args.lock).read_text())
        binary = fetch_binary(lock, fetch, clients)
    except Exception as exc:  # no network, no credentials, no boto3: nothing was changed
        print(f"UNCHECKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    got = sha256_hex(binary)
    if got != lock["sha256"]:
        print(f"MISMATCH: binary sha256 {got} != lock {lock['sha256']}")
        return 1
    print(f"OK: binary sha256 {got} matches the lock ({lock['source']})")
    if args.verify_only or not args.out:
        return 0
    data = layer_zip(binary)
    Path(args.out).write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes)")
    if args.publish:
        return publish(data, got, input_fn, clients)
    return 0


if __name__ == "__main__":
    sys.exit(run())

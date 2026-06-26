"""AWS Lambda: build the 48h 1k timelapse video when a new 1k still lands.

Trigger: S3 ``ObjectCreated`` on prefix ``1k/`` suffix ``.png`` in the
``the-sun-now`` bucket (same region as this Lambda, so S3<->Lambda transfer is free).

Per invocation (one product):
  1. read the new 1k still's observation time (object metadata ``obstime``, else
     the event time) and copy it into the frame queue at frames/<prod>/<ts>_1k.png
  2. prune the queue by age (keep ~49h; robust to double-firing triggers)
  3. download the queue, ffmpeg -> video/rhef_<prod>_1k.mp4 at FPS
  4. upload the video and write manifest/<prod>.json

Pure logic (queue/manifest/keys) is unit-tested in sunback/__tests__; this module
is the I/O shell and is verified by deploying against a staging prefix.
"""
import json
import os
import re
import subprocess
import tempfile

import boto3

from .frame_queue import (
    build_grid_sequence,
    frame_key_for,
    select_stale_frames,
)
from .manifest import (
    build_manifest_fragment,
    manifest_key,
    product_from_1k_key,
    video_key,
)

# --- Tunables (env-overridable) ---------------------------------------------
BUCKET = os.environ.get("SUN_BUCKET", "the-sun-now")
FPS = int(os.environ.get("VIDEO_FPS", "18"))
FRAME_WINDOW = int(os.environ.get("FRAME_WINDOW", "144"))  # 48h * 3/hr (grid slots)
GRID_CADENCE_S = int(os.environ.get("GRID_CADENCE_S", "1200"))  # 20 min grid
PRUNE_WINDOW_S = int(os.environ.get("PRUNE_WINDOW_S", str(49 * 3600)))  # keep ~49h
FFMPEG = os.environ.get("FFMPEG_PATH", "/opt/bin/ffmpeg")  # from the ffmpeg layer
# ----------------------------------------------------------------------------

s3 = boto3.client("s3")

_OBSTIME_FALLBACK_RE = re.compile(r"(\d{8}T\d{6})")


def _obstime_for(bucket, key, event_time):
    """Observation timestamp for the new still: object metadata, else event time."""
    head = s3.head_object(Bucket=bucket, Key=key)
    meta = head.get("Metadata", {})
    if "obstime" in meta:
        return meta["obstime"]
    # event_time like 2026-06-24T20:20:31.123Z -> compact
    return re.sub(r"[-:]", "", event_time).split(".")[0]


def _list_queue(product):
    prefix = f"frames/{product}/"
    keys = []
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return keys


def _build_video(product, frame_keys, workdir):
    """Snap frames to a uniform 20-min grid (holding through gaps), then ffmpeg.

    Returns (mp4_path, n_unique_real_frames). The video has one slot per grid step
    so playback advances at a constant rate regardless of when the reducer ran.
    """
    seq = build_grid_sequence(frame_keys, cadence_s=GRID_CADENCE_S, max_slots=FRAME_WINDOW)
    if not seq:
        return None, 0
    # download each distinct real frame once; held slots reuse the same local file
    local_of = {}
    for key in dict.fromkeys(seq):
        local = os.path.join(workdir, key.replace("/", "_"))
        s3.download_file(BUCKET, key, local)
        local_of[key] = local
    list_path = os.path.join(workdir, "frames.txt")
    with open(list_path, "w") as fp:
        for key in seq:  # one line per grid slot (repeats = held frames)
            fp.write(f"file '{local_of[key]}'\n")
    out_path = os.path.join(workdir, f"{product}.mp4")
    subprocess.run(
        [
            FFMPEG, "-y", "-r", str(FPS), "-f", "concat", "-safe", "0",
            "-i", list_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path,
        ],
        check=True,
        capture_output=True,
    )
    return out_path, len(local_of)


def _process_one(product, trigger_key, obstime):
    # 1. add the new still to the queue
    new_frame_key = frame_key_for(product, obstime)
    s3.copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": trigger_key},
        Key=new_frame_key,
        MetadataDirective="COPY",
    )

    # 2. prune by AGE (keep a fixed 48h+margin window, robust to double-firing)
    queue = _list_queue(product)
    stale = set(select_stale_frames(queue, PRUNE_WINDOW_S))
    for key in stale:
        s3.delete_object(Bucket=BUCKET, Key=key)
    queue = [k for k in queue if k not in stale]
    if new_frame_key not in queue:
        queue.append(new_frame_key)

    # 3. ffmpeg the window into a video
    with tempfile.TemporaryDirectory() as workdir:
        video_path, frame_count = _build_video(product, queue, workdir)
        # 4a. upload video
        s3.upload_file(
            video_path, BUCKET, video_key(product),
            ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4",
                       "ContentDisposition": "inline"},
        )

    # 4b. write the manifest fragment
    fragment = build_manifest_fragment(
        product,
        updated=_iso(obstime),
        frame_count=frame_count,
        integration={
            "frames": int(os.environ.get("INTEGRATION_FRAMES", "5")),
            "method": os.environ.get("INTEGRATION_METHOD", "median"),
        },
    )
    s3.put_object(
        Bucket=BUCKET, Key=manifest_key(product),
        Body=json.dumps(fragment).encode("utf-8"),
        ACL="public-read", ContentType="application/json",
        CacheControl="no-cache",
    )
    return fragment


def _iso(compact):
    """20260624T202000 -> 2026-06-24T20:20:00Z (best-effort; pass-through if already ISO)."""
    m = _OBSTIME_FALLBACK_RE.search(compact)
    if not m:
        return compact
    c = m.group(1)
    return f"{c[0:4]}-{c[4:6]}-{c[6:8]}T{c[9:11]}:{c[11:13]}:{c[13:15]}Z"


def handler(event, context):
    results = []
    for record in event.get("Records", []):
        key = record["s3"]["object"]["key"]
        product = product_from_1k_key(key)
        if product is None:
            continue  # not a 1k still we care about
        event_time = record.get("eventTime", "")
        obstime = _obstime_for(BUCKET, key, event_time)
        results.append(_process_one(product, key, obstime))
    return {"processed": [r["id"] for r in results]}

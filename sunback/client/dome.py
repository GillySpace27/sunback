"""sunback-dome (SB-21, v1): the 48-hour Sun movies as domemaster files for a planetarium.

Lives in sunback/client, not sunback/run as the task file planned: importing
sunback.run loads the research runner and numpy, and this tool is stdlib only.

Division of work (Gilly's default, 2026-10-15): the RHEF filtering stays central,
in the sunback pipeline, and the planetarium does the projection on its own
machine. This command runs there: it reads manifest/index.json, downloads the
already-filtered 48-hour 1k movies, reprojects each to a square domemaster
(fisheye, the solar disk at the zenith) with ffmpeg, adds a caption, and writes
an .m3u playlist the dome's player can loop. The other reading of the owner wish
(the planetarium's hardware running RHEF itself) is left as an option for Gilly.

The source is 1k, so a 4k dome master is upscaled and will look soft.
Standard library only; needs ffmpeg (and ffprobe for the self-check) on PATH.

    sunback-dome --once --out ~/dome                 # one pass, rainbow product
    sunback-dome --product 171 --product 304 --out ~/dome   # refresh every hour
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from sunback.client import fetch

logger = logging.getLogger("sunback.dome")

# Same words as CREDIT in sunback/putter/AwsPutter.py (bound by test_client.py).
CREDIT = "Imagery courtesy of NASA/SDO and the AIA science team."
CAPTION = CREDIT + " Processed with RHEF."
DEFAULT_SIZE = 2048
DEFAULT_FOV = 90.0  # degrees of dome the 1k frame spans; a display choice, not a measured value
DEFAULT_INTERVAL_S = 3600


def ffmpeg_filters(ffmpeg="ffmpeg"):
    """Names of the filters this ffmpeg build has."""
    out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, check=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return names


def dome_filter(size, fov, filters, caption=CAPTION):
    """The -vf chain: flat frame to a square fisheye with the disk at the centre (zenith).

    v360 maps the flat movie onto a 180 degree fisheye; without v360 the frame is
    scaled to ``fov``/180 of the dome diameter and padded to a square, which is
    the same picture for a disk at the zenith, without the curvature.
    """
    if "v360" in filters:
        chain = [f"v360=input=flat:output=fisheye:ih_fov={fov}:iv_fov={fov}:h_fov=180:v_fov=180:w={size}:h={size}"]
    else:
        inner = int(size * fov / 180) // 2 * 2
        chain = [f"scale={inner}:{inner}", f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black"]
    if caption and "drawtext" in filters:
        text = caption.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        chain.append(f"drawtext=text='{text}':fontcolor=white@0.85:fontsize={max(8, size // 72)}"
                     f":x=(w-text_w)/2:y=h*0.80")
    chain.append("format=yuv420p")
    return ",".join(chain)


def reproject(src, dst, size=DEFAULT_SIZE, fov=DEFAULT_FOV, caption=CAPTION, ffmpeg="ffmpeg"):
    """Write the domemaster MP4 dst from the flat movie src; replaces dst atomically."""
    filters = ffmpeg_filters(ffmpeg)
    dst = Path(dst)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    for cap in ([caption, None] if caption else [None]):
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
               "-vf", dome_filter(size, fov, filters, cap), "-an", "-c:v", "libx264", "-crf", "20",
               "-movflags", "+faststart", str(tmp)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0:
            os.replace(tmp, dst)
            return dst
        if cap:
            logger.warning("caption failed (%s); writing without it", done.stderr.strip().splitlines()[-1:])
    raise OSError(f"ffmpeg failed on {src}: {done.stderr.strip()[-400:]}")


def download(url, dst):
    req = Request(url, headers={"User-Agent": fetch.USER_AGENT})
    tmp = Path(dst).with_suffix(".part")
    with urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    os.replace(tmp, dst)
    return Path(dst)


def one_pass(products, out, base_url, size, fov, caption, work):
    """Render every product once; returns the written domemaster paths in order."""
    index = fetch.read_index(base_url)
    by_id = {p.get("id"): p for p in index.get("products", [])}
    written = []
    for pid in products:
        frag = by_id.get(pid)
        key = frag and (frag.get("video_v") or frag.get("video"))
        if not key:
            logger.error("%s: no movie in manifest/index.json", pid)
            continue
        src = download(base_url.rstrip("/") + "/" + key, Path(work) / f"flat_{pid}.mp4")
        dst = reproject(src, Path(out) / f"dome_{pid}.mp4", size, fov, caption)
        logger.info("%s: %s (through %s)", pid, dst, frag.get("through") or frag.get("updated"))
        written.append(dst)
    if written:
        playlist = Path(out) / "dome.m3u"
        playlist.write_text("".join(f"{p.name}\n" for p in written), encoding="utf-8")
    return written


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="sunback-dome", description="The 48-hour Sun movies as domemaster MP4s.")
    ap.add_argument("--product", action="append", choices=fetch.PRODUCT_IDS,
                    help="channel; repeat for several (default rainbow)")
    ap.add_argument("--out", type=Path, default=Path("."), help="output folder (default here)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"square side in pixels (default {DEFAULT_SIZE})")
    ap.add_argument("--fov", type=float, default=DEFAULT_FOV,
                    help=f"degrees of the dome the Sun movie spans (default {DEFAULT_FOV:g})")
    ap.add_argument("--no-caption", action="store_true", help="leave out the credit caption")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S, help="seconds between refreshes")
    ap.add_argument("--base-url", default=fetch.BASE_URL, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        logger.error("sunback-dome needs ffmpeg on PATH (https://ffmpeg.org/download.html)")
        return 2
    products = args.product or ["rainbow"]
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / ".sunback-dome"
    work.mkdir(exist_ok=True)
    caption = None if args.no_caption else CAPTION
    told = False
    while True:
        try:
            written = one_pass(products, args.out, args.base_url, args.size, args.fov, caption, work)
        except OSError as e:
            logger.error("sunback-dome: %s", e)
            written = []
            if args.once:
                return 1
        if written and not told:
            print(f"Play in a loop with: ffplay -loop 0 {written[0]}   (or open {args.out / 'dome.m3u'})")
            told = True
        if args.once:
            return 0 if written else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())

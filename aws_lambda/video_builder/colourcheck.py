"""Colour check for the video-builder ffmpeg argv (SB-1). Not packed into the Lambda zip.

Encodes one PNG through the concat + libx264 argv that handler._build_video uses,
twice: with the bt709 tags and without them. Frame 0 of each MP4 is decoded back
to RGB the way a player would: the tagged stream with its tagged matrix (bt709),
the untagged stream with --untagged-matrix (default bt709, the matrix a player
that assumes HD video is bt709 picks for an untagged 1024x1024 stream). Prints the
mean absolute RGB difference from the PNG (0-255 scale) for both.

Stdlib plus ffmpeg on PATH (or --ffmpeg). Exit 0 when the tagged encode is closer
to the PNG than the untagged one, 1 when it is not, 3 when ffmpeg is missing.

    python aws_lambda/video_builder/colourcheck.py [--png PATH] [--ffmpeg PATH]
        [--untagged-matrix bt709|bt470bg] [--json]
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

SIZE = 1024
# The values SB-1 commits in handler._build_video (pinned by test_build_video_args.py).
BT709_VF = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
BT709_OUT_ARGS = ["-color_range", "tv", "-colorspace", "bt709"]


def write_test_png(path, size=SIZE):
    """A grey band over a saturated orange-to-blue ramp: where a matrix mismatch shows."""
    rows = []
    for y in range(size):
        row = bytearray([0])  # PNG filter type 0 (none)
        for x in range(size):
            if y < size // 4:
                r, g, b = 128, 128, 128
            else:
                t = x / (size - 1)
                r, g, b = int(255 * (1 - t)), int(140 * (1 - t) + 40 * t), int(255 * t)
            row += bytes((r, g, b))
        rows.append(bytes(row))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    with open(path, "wb") as fp:
        fp.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                 + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def encode_argv(ffmpeg, list_path, out_path, tagged):
    """The _build_video argv (VIDEO_FPS 18, X264_PRESET veryfast), with or without the tags."""
    argv = [ffmpeg, "-y", "-r", "18", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c:v", "libx264", "-preset", "veryfast"]
    if tagged:
        argv += ["-vf", BT709_VF]
    argv += ["-pix_fmt", "yuv420p"]
    if tagged:
        argv += BT709_OUT_ARGS
    return argv + ["-movflags", "+faststart", out_path]


def rgb_frame(ffmpeg, path, matrix=None):
    """Frame 0 as packed rgb24 bytes; ``matrix`` names the YUV matrix for a video."""
    vf = ["-vf", f"scale=in_color_matrix={matrix}:in_range=tv,format=rgb24"] if matrix else []
    return subprocess.run([ffmpeg, "-v", "error", "-i", path, "-frames:v", "1", *vf,
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                          check=True, capture_output=True).stdout


def mean_abs_diff(a, b):
    if len(a) != len(b):
        raise ValueError(f"frame sizes differ: {len(a)} vs {len(b)} bytes")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", help="source still (default: a generated 1024x1024 ramp)")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--untagged-matrix", default="bt709", choices=["bt709", "bt470bg"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        print(f"UNCHECKED: no ffmpeg at {args.ffmpeg!r}")
        return 3
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    result = {"ffmpeg": version, "untagged_matrix": args.untagged_matrix}
    with tempfile.TemporaryDirectory() as work:
        png = os.path.abspath(args.png) if args.png else os.path.join(work, "ramp.png")
        if not args.png:
            write_test_png(png)
        list_path = os.path.join(work, "frames.txt")
        with open(list_path, "w") as fp:
            fp.write(f"file '{png}'\n")
        source = rgb_frame(ffmpeg, png)
        for name, tagged in (("untagged", False), ("tagged", True)):
            mp4 = os.path.join(work, f"{name}.mp4")
            subprocess.run(encode_argv(ffmpeg, list_path, mp4, tagged), check=True, capture_output=True)
            matrix = "bt709" if tagged else args.untagged_matrix
            result[name] = round(mean_abs_diff(source, rgb_frame(ffmpeg, mp4, matrix)), 3)
    result["ok"] = result["tagged"] < result["untagged"]
    if args.json:
        print(json.dumps(result))
    else:
        print(version)
        print(f"mean |RGB| difference, untagged argv (read as {args.untagged_matrix}): {result['untagged']}")
        print(f"mean |RGB| difference, tagged argv (read as bt709): {result['tagged']}")
        print("PASS: the tagged encode is closer to the still" if result["ok"]
              else "FAIL: the tagged encode is not closer to the still")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

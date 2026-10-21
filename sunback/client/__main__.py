"""Wallpaper client 2.0 (SB-21): ``python -m sunback.client``.

Reads manifest/<id>.json, downloads the newest 1k still only when it changed, sets
it as the desktop picture and prints when it was taken. Standard library only.

    python -m sunback.client                     # first run asks for a channel and interval
    python -m sunback.client --product 193 --once
    python -m sunback.client --per-screen 171,193,304   # macOS: one channel per screen
    python -m sunback.client --download-only --once     # fetch, do not touch the desktop

The old ``sunback-run`` script was retired on 2026-10-15.
"""
import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

from sunback.client import fetch, wallpaper

logger = logging.getLogger("sunback.client")
DEFAULT_PRODUCT = "171"
DEFAULT_INTERVAL_S = 60  # as run_client(delay=60) in sunback/run/run_client_background.py


def config_path():
    """client.json in the per-user config folder of each platform."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "sunback" / "client.json"
    if system == "Windows":
        return Path(os.environ.get("APPDATA") or Path.home()) / "sunback" / "client.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "sunback" / "client.json"


def cache_dir():
    """Where the two newest stills per product are kept."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "sunback" / "client"
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "sunback" / "client"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "sunback" / "client"


def load_config(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def choose(path, ask=input, out=print):
    """First-run chooser: pick a channel and an interval, save them, return them."""
    out("Which Sun would you like on your desktop?")
    for i, pid in enumerate(fetch.PRODUCT_IDS, 1):
        out(f"  {i:2d}. {pid}")
    pick = ask(f"Number 1-{len(fetch.PRODUCT_IDS)} [{DEFAULT_PRODUCT}]: ").strip()
    product = DEFAULT_PRODUCT
    if pick.isdigit() and 1 <= int(pick) <= len(fetch.PRODUCT_IDS):
        product = fetch.PRODUCT_IDS[int(pick) - 1]
    every = ask(f"Check every how many seconds [{DEFAULT_INTERVAL_S}]: ").strip()
    interval = int(every) if every.isdigit() and int(every) > 0 else DEFAULT_INTERVAL_S
    cfg = {"product": product, "interval": interval}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    out(f"Saved {path}; change it any time with --choose.")
    return cfg


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sunback.client",
                                 description="Set the desktop picture to the newest image of the Sun.")
    ap.add_argument("--product", choices=fetch.PRODUCT_IDS, help=f"channel (default {DEFAULT_PRODUCT})")
    ap.add_argument("--interval", type=int, help=f"seconds between checks (default {DEFAULT_INTERVAL_S})")
    ap.add_argument("--once", action="store_true", help="check once and exit")
    ap.add_argument("--per-screen", metavar="IDS",
                    help="macOS: comma-separated channels, one per screen in order, repeating")
    ap.add_argument("--choose", action="store_true", help="run the first-run chooser again")
    ap.add_argument("--download-only", action="store_true", help="fetch the image but leave the desktop alone")
    ap.add_argument("--base-url", default=fetch.BASE_URL, help=argparse.SUPPRESS)
    ap.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def plan(args, cfg):
    """[(product, screen index or None)] for one pass."""
    if args.per_screen:
        ids = [s.strip() for s in args.per_screen.split(",") if s.strip()]
        unknown = [i for i in ids if i not in fetch.PRODUCT_IDS]
        if unknown or not ids:
            raise SystemExit(f"--per-screen: unknown channel(s) {unknown}; choose from {', '.join(fetch.PRODUCT_IDS)}")
        n = len(wallpaper.screens()) if platform.system() == "Darwin" else 0
        if n == 0:
            logger.warning("--per-screen works on macOS only; using %s on every screen", ids[0])
            return [(ids[0], None)]
        return [(ids[i % len(ids)], i) for i in range(n)]
    return [(args.product or cfg.get("product") or DEFAULT_PRODUCT, None)]


def one_pass(jobs, args, cache, force=False):
    """Fetch every job's product and set the ones that changed (all of them when force). Returns the number set."""
    changed = 0
    for product, screen in jobs:
        path, when, new = fetch.fetch_still(product, cache, args.base_url)
        if not (new or force):
            continue
        changed += 1
        if not args.download_only:
            if screen is None:
                wallpaper.set_wallpaper(path)
            else:
                wallpaper.set_wallpaper_macos(path, wallpaper.screens()[screen])
        where = "" if screen is None else f" on screen {screen + 1}"
        print(f"{fetch.describe(product, when)}{where}", flush=True)
    return changed


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    cfg_file = config_path()
    cfg = load_config(cfg_file)
    if args.choose or (not cfg and not args.product and not args.per_screen and sys.stdin.isatty()):
        cfg = choose(cfg_file)
    interval = args.interval or cfg.get("interval") or DEFAULT_INTERVAL_S
    cache = args.cache_dir or cache_dir()
    jobs = plan(args, cfg)
    first = True
    while True:
        try:
            one_pass(jobs, args, cache, force=first)
            first = False
        except (OSError, ValueError, NotImplementedError) as e:
            if args.once:
                logger.error("sunback client: %s", e)
                return 1
            logger.error("sunback client: %s; trying again in %s s", e, interval)
        if args.once:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())

"""The ``sunback-run`` command, kept only to say where the wallpaper went.

``sunback-run`` (the wallpaper client in releases up to 0.6.17.3) was retired on
2026-10-15. Anyone who upgrades and still types it gets this notice instead of
"command not found". Standard library only, like the rest of sunback.client.
"""
import sys

HELIOGRAM_URL = "https://gilly.space/heliogram/"

NOTICE = f"""\
sunback-run has been superseded by Heliogram, a Mac app that plays the last
48 hours of the Sun as your desktop picture, a screen saver or a wall:

    {HELIOGRAM_URL}

sunback-run is no longer supported and has been retired; this command only
prints this notice. The sunback package itself remains Gilly's research
pipeline.
"""


def main(argv=None):
    sys.stderr.write(NOTICE)
    return 1


if __name__ == "__main__":
    sys.exit(main())

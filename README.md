Solar Background Updater
==============================
[//]: # (Badges)
<a href="https://github.com/GillySpace27/sunback/actions/workflows/tests.yml" target="_blank">![tests](https://github.com/GillySpace27/sunback/actions/workflows/tests.yml/badge.svg?branch=master)</a>
<a href="https://github.com/GillySpace27/sunback/tree/master" target="_blank">![GitHub last commit (branch)](https://img.shields.io/github/last-commit/GillySpace27/sunback/master)</a>
<!-- 2026-10-02 (SB-10): Travis: .travis.yml moved to attic/.travis.yml; badge kept for the record: <a href="https://travis-ci.com/GillySpace27/sunback" target="_blank">![Travis Build Status](https://travis-ci.com/GillySpace27/sunback.svg?branch=master)</a> -->
<!-- 2026-10-02 (SB-10): AppVeyor: appveyor.yml moved to attic/appveyor.yml; badge kept for the record: <a href="https://ci.appveyor.com/project/GillySpace27/sunback/" target="_blank">![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/ji7e0pm5xxckf6rq/branch/master?svg=true)</a> -->
<a href="https://sunback.readthedocs.io/en/latest/?badge=latest" target="_blank">![Documentation Status](https://readthedocs.org/projects/sunback/badge/?version=latest)</a>

<a href="https://pypi.org/project/sunback/" target="_blank"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/sunback"></a>
<a href="https://pypi.org/project/sunback/" target="_blank">![PyPI](https://img.shields.io/pypi/v/sunback)</a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/sunback">
<a href="https://github.com/GillySpace27/sunback/" target="_blank">![GitHub repo size](https://img.shields.io/github/repo-size/GillySpace27/sunback)</a>
<a href="https://opensource.org/licenses/BSD-3-Clause" target="_blank">![GitHub](https://img.shields.io/github/license/GillySpace27/sunback)</a>

<a href="https://codeclimate.com/github/GillySpace27/sunback/maintainability"><img src="https://api.codeclimate.com/v1/badges/f7ae86dc9703d4a7eec6/maintainability" /></a>
<a href="https://codeclimate.com/github/GillySpace27/sunback/test_coverage"><img src="https://api.codeclimate.com/v1/badges/f7ae86dc9703d4a7eec6/test_coverage" /></a>
<a href="https://www.codacy.com/manual/GillySpace27/sunback" target="_blank">![Codacy Badge](https://api.codacy.com/project/badge/Grade/a47b3701e7544010a4708d923a71fedb)</a>
<!-- 2026-10-02 (SB-10): Codecov: .codecov.yml moved to attic/.codecov.yml, and no workflow uploads coverage; badge kept for the record: <a href="https://codecov.io/gh/GillySpace27/sunback/branch/master" target="_blank">![codecov](https://codecov.io/gh/GillySpace27/sunback/branch/master/graph/badge.svg)</a> -->


This program is the core analysis package that Dr. Gilly uses for his research. This code is very modular and can perform many different data pipelines for images with arbitrary and custom filtering steps. Sunback runs analysis on images from AIA, updates Gilly's personal website, and allows users to run a daemon on their machines to update their computer desktop background.

## Working here

Before changing anything, read [AGENTS.md](https://github.com/GillySpace27/sunback/blob/master/AGENTS.md):
what is production, what must never be run, and how to prove a change works.
Run every offline check with `bash devtools/check.sh`.

## Installation

For a live Sun on your desktop, use [Heliogram](https://gilly.space/heliogram/).
The old `sunback-run` wallpaper client was retired on 2026-10-15; its code is in
`attic/sunback/run/run_client_background.py`.

### Github Installation Instructions

``` zsh
git clone https://github.com/GillySpace27/sunback.git
cd sunback
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

### PyPi Installation Instructions:
``` bash
pip install sunback
```

Check contribute.md to see how you can help out!

## Wallpaper

The new client reads one small public file per channel (`manifest/<id>.json`),
downloads the newest 1k image only when it changed, sets it as your desktop
picture and prints when it was taken, for example `171 at 14:20 UTC (7 min ago)`.
It uses only the Python standard library (plus pyobjc on macOS, which sunback
already installs there).

``` bash
python -m sunback.client                    # first run: pick a channel and an interval
python -m sunback.client --product 193      # a channel by id, checked every 60 s
python -m sunback.client --once             # check once and exit
python -m sunback.client --per-screen 171,193,304   # macOS: one channel per screen
python -m sunback.client --choose           # pick again
```

Channels: `rainbow`, `171`, `193`, `211`, `304`, `335`, `94`, `131`, `1600`,
`1700`, `composite_uv`, `dem`. The choice is saved in `client.json` in your
config folder (`~/Library/Application Support/sunback` on macOS, `%APPDATA%\sunback`
on Windows, `~/.config/sunback` on Linux). The `sunback-run` script is retired. Only macOS has been tested; the Windows (registry) and
GNOME (gsettings) setters are the old ones, moved unchanged and not yet tried on
those systems.

## For planetariums

`sunback-dome` turns the 48-hour Sun movies into square domemaster (fisheye)
MP4s with the solar disk at the zenith, for a planetarium's own player. The
filtering (RHEF) is done centrally by this pipeline; the projection runs on the
planetarium's machine, which needs `ffmpeg` installed.

``` bash
sunback-dome --once --out ~/dome                          # rainbow, one pass
sunback-dome --product 171 --product 304 --out ~/dome     # refresh every hour
ffplay -loop 0 ~/dome/dome_rainbow.mp4                    # or loop ~/dome/dome.m3u
```

Options: `--size` (square side in pixels, default 2048), `--fov` (how many
degrees of the dome the movie spans, default 90), `--no-caption`. The source
movies are 1k, so large dome masters are upscaled and look soft. Each movie
carries the caption "Imagery courtesy of NASA/SDO and the AIA science team.
Processed with RHEF." RHEF output is a visualization, not a calibrated radiance.

This is version 1 of one reading of the plan (projection on site, filtering
central). Running RHEF itself on planetarium hardware is the other option and
is Gilly's call.

### Copyright

Copyright (c) 2019, [C. R. Gilly](https://gilly.space) [ORCiD](https://orcid.org/0000-0003-0021-9056)


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

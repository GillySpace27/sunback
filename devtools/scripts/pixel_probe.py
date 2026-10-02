#!/usr/bin/env python3
"""Fixed render for comparing two reducer images (SB-13).

Render mode writes a fixed synthetic five-frame AIA 171 stack, integrates it
with sunback.utils.time_integration.integrate_frames (median, the reducer
default), runs sunkit_image.radial.rhef with method="scipy" and vignette=None
(as RHEFProcessor.do_work does with do_vignette off), applies the AIA 171 IDL
colour table from sunback.science.color_tables and saves a PNG with matplotlib.
It prints one JSON object with library versions and SHA-256 digests and keeps
rhef.npy and probe.png in --out. Run it from the same checkout in two images:
equal digests mean both environments render this chain identically.

Compare mode needs only numpy: it prints the two digests and the maximum
absolute difference of the rhef arrays, and exits 0 when both digests match,
1 when they differ.

Usage, from the repository root:
  python -m devtools.scripts.pixel_probe --out DIR > DIR.json
  python -m devtools.scripts.pixel_probe --compare DIR_A DIR_B
"""

import argparse
import hashlib
import json
import pathlib
import sys


def render(out):
    import astropy.units as u
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import sunkit_image
    import sunkit_image.radial as radial
    import sunpy
    import sunpy.map

    from sunback.__tests__.fixtures.make_fits import make_synthetic_aia_fits
    from sunback.science.color_tables import aia_color_table
    from sunback.utils.time_integration import integrate_frames

    out.mkdir(parents=True, exist_ok=True)
    maps = []
    for seed in range(5):
        path = make_synthetic_aia_fits(out / f"frame{seed}.fits", wave="0171", shape=(256, 256), seed=seed)
        maps.append(sunpy.map.Map(path))
    stacked = integrate_frames([m.data for m in maps], method="median")
    smap = sunpy.map.Map(stacked, maps[-1].meta)
    data = np.asarray(
        radial.rhef(smap, upsilon=None, vignette=None, method="scipy", progress=False).data, dtype="float64"
    )
    np.save(out / "rhef.npy", data)
    plt.imsave(out / "probe.png", np.nan_to_num(data, nan=0.0), cmap=aia_color_table(171 * u.angstrom),
               vmin=0.0, vmax=1.0, origin="lower")
    report = {
        "numpy": np.__version__,
        "sunpy": sunpy.__version__,
        "sunkit_image": sunkit_image.__version__,
        "matplotlib": matplotlib.__version__,
        "nan_count": int(np.isnan(data).sum()),
        "data_sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
        "png_sha256": hashlib.sha256((out / "probe.png").read_bytes()).hexdigest(),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def compare(dir_a, dir_b):
    import numpy as np

    a = np.load(dir_a / "rhef.npy")
    b = np.load(dir_b / "rhef.npy")
    same_png = (dir_a / "probe.png").read_bytes() == (dir_b / "probe.png").read_bytes()
    same_data = a.shape == b.shape and a.tobytes() == b.tobytes()
    diff = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else float("nan")
    print(json.dumps({"same_data": same_data, "same_png": same_png, "max_abs_diff": diff}, sort_keys=True))
    return 0 if same_data and same_png else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed render for comparing two reducer images (SB-13).")
    parser.add_argument("--out", type=pathlib.Path, help="render into this directory")
    parser.add_argument("--compare", nargs=2, type=pathlib.Path, metavar=("DIR_A", "DIR_B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.out:
        return render(args.out)
    parser.error("give --out DIR or --compare DIR_A DIR_B")


if __name__ == "__main__":
    sys.exit(main())

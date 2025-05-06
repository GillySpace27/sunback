import numpy as np
import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE
import sunkit_image.enhance as enhance
import sunkit_image.radial as radial
import timeit
import csv
from tqdm import tqdm

import warnings
import logging
from astropy.io.fits.verify import VerifyWarning as FitsVerifyWarning
from astropy.io import fits

fits.conf.warn_on_missing_end = False
fits.conf.verify = 'silentfix'
fits.conf.enable_record_val_warnings = False

warnings.filterwarnings("ignore", category=FitsVerifyWarning)

# Load SunPy sample AIA 171 image
aia_map = sunpy.map.Map(AIA_171_IMAGE)
input_data = aia_map

# Define filter functions
def run_mgn():
    return enhance.mgn(input_data)

def run_rhef():
    return radial.rhef(input_data)

def run_nrgf():
    return radial.nrgf(input_data)

def run_wow():
    return enhance.wow(input_data)


def benchmark(func, num_runs):
    import statistics
    times = []
    result = None

    for _ in range(num_runs):
        start_time = timeit.default_timer()
        result_local = func()
        end_time = timeit.default_timer()

        times.append(end_time - start_time)
        result = result_local

    # Determine output shape
    if isinstance(result, np.ndarray):
        output_shape = result.shape
    elif isinstance(result, tuple) and isinstance(result[0], np.ndarray):
        output_shape = result[0].shape
    elif hasattr(result, 'data'):
        output_shape = result.data.shape
    else:
        output_shape = "Unknown"

    if output_shape != input_data.data.shape:
        output_shape_str = f"{output_shape} (MISMATCH)"
    else:
        output_shape_str = str(output_shape)

    result_entry = {
        "Filter Name": func.__name__,
        "Average Time (s)": f"{statistics.mean(times):.4f}",
        "Time StdDev (s)": f"{statistics.stdev(times):.4f}" if len(times) > 1 else "0.0000",
        "Output Shape": output_shape_str,
        "Status": "Success",
        "Message": ""
    }

    return result_entry

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark sunkit_image filters on AIA 171 sample")
    parser.add_argument("--num_runs", type=int, default=2, help="Number of runs for each filter")
    args = parser.parse_args()

    print(f"\nRunning each filter {args.num_runs} time(s) in a subprocess...\n")

    results = []
    for f in tqdm([run_mgn, run_wow, run_rhef, run_nrgf]):
        res = benchmark(f, args.num_runs)
        results.append(res)

    # Write to CSV
    output_file = "./benchmark_results.csv"
    with open(output_file, mode="w", newline="") as csvfile:
        fieldnames = [
            "Filter Name",
            "Average Time (s)",
            "Time StdDev (s)",
            "Output Shape",
            "Status",
            "Message"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        print(" ".join(fieldnames))
        for res in results:
            writer.writerow(res)
            print(
                f"\n{res['Filter Name']} → Status: {res['Status']}, "
                f"Time: {res['Average Time (s)']} ± {res['Time StdDev (s)']} s"
            )

    print(f"\n✅ Benchmark results saved to {output_file}\n")
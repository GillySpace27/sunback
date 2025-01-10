## Imports  ------------------------------------------------
import os
# import matplotlib.pyplot as plt
# %matplotlib inline
# %matplotlib ipympl
# %matplotlib notebook
import xarray as xr
changed=False
from tqdm import tqdm

## Helper Functions  ------------------------------------------------
def where():
    print("Current Directory: \n{}".format(os.getcwd()))

def change_directory(directory, changed):

    dir_out=os.path.join(directory, "src")
    if not directory in os.getcwd():
        os.chdir(os.path.join(os.getcwd(), dir_out))
    if not changed:
        new_path = os.path.join(os.getcwd(), "..")
        os.chdir(new_path)
    changed=True
#     where()

## More Imports  ------------------------------------------------

# Change Path
if not changed:
    change_directory('sunback', changed)
    changed = True
new_path = os.path.abspath("/srv/data/shared/notebooks/cgilly/sunback/src")
os.chdir(new_path)

# Import
from sunback.fetcher.LocalFetcher import LocalSingleFetcher, LocalCdfFetcher
from sunback.processor.ImageProcessorCV import ImageProcessorCV #, ImageProcessorNetCDF
from sunback.processor.QRNProcessor import QRNSingleShotProcessor
from sunback.science.parameters import Parameters
from sunback.run import Runner, SingleRunner


def run(img_path, verb=False, confirm=True):
    os.chdir(os.path.dirname(img_path))
    if verb:
        where()
        print("    ", os.path.basename(img_path))

    p = Parameters()
    p.use_image_path(img_path)
    p.batch_name("Single")
    p.do_single = True
    p.run_type("Process a Single Image")
    p.do_one(True, True)
    p.is_debug(True)
    p.destroy = False
    p.confirm_save = confirm
    # Set the Processesors
    p.fetchers(LocalCdfFetcher,          rp=True)  # Get the desired file
    p.processors([QRNSingleShotProcessor],  rp=True)  # Apply the QRN Filter

    if True:
        SingleRunner(p).start(verb=verb)
    else:
        print("Running Silently...", end='')
        with open('log.txt') as os.sys.stdout:
            SingleRunner(p).start(verb=verb)
        print("Done!")

def run_batch(batch_directory, verb=False, confirm=False):
    items = os.listdir(batch_directory)
    all_cdf=[x for x in items if '.nc' in x]
    raw_cdf=[x for x in all_cdf if "filtered" not in x]
    abs_cdf=[os.path.join(batch_directory, x) for x in raw_cdf]
    abs_cdf.sort()
    print("Looking in \n  {}".format(batch_directory))
    print("    Found {} images".format(len(raw_cdf)), flush=True)
    if verb:
        print("   ", raw_cdf, flush=True)

    # Run the code on the images:
    if not verb:
        std = os.sys.stdout
        os.sys.stdout = open("log.txt", "w+")

    for im_path in tqdm(abs_cdf):
        run(im_path, verb=verb, confirm=confirm)

    if not verb:
        os.sys.stdout = std

if __name__ == "__main__":
    # Do something if this file is invoked on its own

    # Go to the directory with the single image_path
    use_directory = r"/srv/data/shared/notebooks/cgilly/sunback/src/sunback_images/Single/"
    use_image_name = r"AIA20210923_172100.nc"
    img_path = os.path.join(use_directory, use_image_name)

    run(img_path)




















































## Imports  ------------------------------------------------
import os
# import matplotlib.pyplot as plt
# %matplotlib inline
# %matplotlib ipympl
# %matplotlib notebook
import xarray as xr
changed=False

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
from fetcher.LocalFetcher import LocalSingleFetcher, LocalCdfFetcher
from processor.ImageProcessorCV import ImageProcessorCV #, ImageProcessorNetCDF
from processor.SRNSubProcessors import SRNSingleShotProcessor
from science.parameters import Parameters
from run import Runner, SingleRunner


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
    p.processors([SRNSingleShotProcessor],  rp=True)  # Apply the SRN Filter

    if True:
        SingleRunner(p).start(verb=verb)
    else:
        print("Running Silently...", end='')
        with open('log.txt') as os.sys.stdout:
            SingleRunner(p).start(verb=verb)
        print("Done!")

if __name__ == "__main__":
    # Do something if this file is invoked on its own   
    
    # Go to the directory with the single image
    use_directory = r"/srv/data/shared/notebooks/cgilly/sunback/src/sunback_images/Single/"
    use_image_name = r"AIA20210923_172100.nc"
    img_path = os.path.join(use_directory, use_image_name)

    run(img_path)




















































from calendar import timegm
from os.path import dirname, abspath, join, isdir, split
from os import makedirs, getcwd, listdir
from time import time, localtime, strftime

from astropy.io import fits


##  PATHS
def build_paths(self):
  """Make the file structure to hold the images"""
  self.local_wave_directory = join(self.params.download_path(), self.current_wave)
  self.fits_folder = join(self.local_wave_directory, "fits\\")
  self.image_folder = join(self.local_wave_directory, "png\\")
  self.movie_folder = abspath(join(self.local_wave_directory, "..\\movies\\"))
  makedirs(self.local_wave_directory, exist_ok=True)
  makedirs(self.image_folder, exist_ok=True)
  makedirs(self.movie_folder, exist_ok=True)
  
  
def set_output_paths(self):
  # self.params.local_fits_paths(self.temp_fits_pathbox)
  list_of_files = list_files_in_directory_absolute(self.fits_folder, 'fits')
  self.temp_fits_pathbox.extend(list_of_files)

def discover_best_data_directory():
    """Determine where to store the images"""
    # TODO find a good directory
    subdirectory_name = "sunback_images\\test"
    if __file__ in globals():
        ddd = dirname(abspath(__file__))
    else:
        ddd = abspath(getcwd())
    
    while "dropbox".casefold() in ddd.casefold():
        ddd = abspath(join(ddd, ".."))
    
    directory = join(ddd, subdirectory_name)
    if not isdir(directory):
        makedirs(directory)
    return directory


def find_done_paths(full_path):
    """Find pngs that have already been made"""
    path = dirname(full_path)
    save_path = path.replace("fits", "png")
    done_paths = [x.casefold() for x in listdir(save_path)]
    return done_paths


def list_files_in_directory(directory, extension="fits"):
    makedirs(directory, exist_ok=True)
    return [f.casefold() for f in listdir(directory) if f.endswith('.' + extension)]

def list_files_in_directory_absolute(directory, extension="fits"):
    makedirs(directory, exist_ok=True)
    out = [directory + f.casefold() for f in listdir(directory) if f.endswith('.' + extension)]
    return out

def get_paths(local_fits_paths, use_wavelengths, download_path):
    wave_bucket = []
    if len(local_fits_paths) == 0:
        for wave in use_wavelengths:
            wave_bucket.append(join(download_path, wave))
    return wave_bucket

## FILE IO
def load_file(self, path):
    """Load a fits file from disk"""
    with fits.open(path, cache=False) as hdul:
        hdul.verify('silentfix+warn')
        wave, t_rec = hdul[0].header['WAVELNTH'], hdul[0].header['T_OBS']
        image = hdul[0].data
        self.image_data = str(wave), str(wave), t_rec, image.shape
    return image

def load_fits_data(hdul, field='primary'):
    """Load a fits file from disk"""
    hdul.verify('silentfix+warn')
    
    try:
        hh = 0
        wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
    except:
        hh = 1
        wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
    
    center = [hdul[field].header['X0_MP'], hdul[field].header['Y0_MP']]
    
    return hdul[field].data, wave, t_rec, center

def save_fits_file(img_path, hdul, frame, name="gated"):
    """Load a fits file from disk"""
    with fits.open(img_path, cache=False, mode="update") as hdul:
        hdul.append(fits.ImageHDU(frame, name=name))
        hdul.verify('silentfix+warn')
        # hdul.writeto()


archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images

def load_series(params, base_dir_path=discover_best_data_directory()):
    """Loads the img series from disk"""
    download_path = params.download_path(base_dir_path)
    print("Loading PNGs from {}...".format(download_path), end='', flush=True)

    all_paths = listdir(download_path)
    png_paths = [join(download_path, path)
                  for path in all_paths if '.png' in path[-4:]]
    print("Success! {} Found\n".format(len(png_paths)))
    # sleep(1)
    params.local_img_paths(png_paths)
    return png_paths




 ## PRINTING
def print_header(seconds, base_path, debug):
    print("\nSunback SDO Image Manipulator \nWritten by Chris R. Gilly")
    print("Check out my website: http://gilly.space\n")
    print("Delay: {} Seconds".format(seconds))
    # print("Coronagraph Mode: {} \n".format(params.mode()))
    print("Base Directory: {}".format(base_path))
    
    if debug:
        print("DEBUG MODE\n")


def print_end_banner(stop):
    mode_string = "" if stop else ", Restarting Loop"
    print("\n_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print("Program Complete{}".format(mode_string))
    print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n")





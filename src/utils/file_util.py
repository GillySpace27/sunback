from calendar import timegm
from os.path import dirname, abspath, join, isdir, split
from os import makedirs, getcwd, listdir
from time import time, localtime, strftime

from astropy.io import fits

archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images

##  PATHS
# def build_paths(self):
#   """Make the file structure to hold the images"""
#   self.wave_directory = join(self.params.img_directory(), self.current_wave)
#   self.fits_folder = join(self.wave_directory, "fits\\")
#   self.image_folder = join(self.wave_directory, "png\\")
#   self.movie_folder = abspath(join(self.wave_directory, "..\\movies\\"))
#   makedirs(self.wave_directory, exist_ok=True)
#   makedirs(self.image_folder, exist_ok=True)
#   makedirs(self.movie_folder, exist_ok=True)


# def set_output_paths(self):
#     # self.params.local_fits_paths(self.temp_fits_pathbox)
#     list_of_files = list_files_in_directory_absolute(self.fits_folder, 'fits')
#     self.temp_fits_pathbox.extend(list_of_files)


def discover_best_data_directory():
    """Determine where to store the images"""
    subdirectory_name = "sunback_images"
    
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

#
# def find_done_paths(full_path):
#     """Find pngs that have already been made"""
#     path = dirname(full_path)
#     save_path = path.replace("fits", "png")
#     done_paths = [x.casefold() for x in listdir(save_path)]
#     return done_paths
#
#
# def list_files_in_directory(directory, extension="fits"):
#     makedirs(directory, exist_ok=True)
#     return [f.casefold() for f in listdir(directory) if f.endswith('.' + extension)]
#
#
# def list_files_in_directory_absolute(directory, extension="fits"):
#     makedirs(directory, exist_ok=True)
#     out = [directory + f.casefold() for f in listdir(directory) if f.endswith('.' + extension)]
#     return out
#
#
# def get_paths(local_fits_paths, use_wavelengths, download_path):
#     wave_bucket = []
#     if len(local_fits_paths) == 0:
#         for wave in use_wavelengths:
#             wave_bucket.append(join(download_path, wave))
#     return wave_bucket


## FILE IO
def save_frame_to_fits_file(fits_path, frame, field="filtered"):
    """Save a fits file to disk"""
    # print("Saving Frame to Fits File")
    with fits.open(fits_path, cache=False, mode="update") as hdul:
        hdul.verify('silentfix+warn')  # Then Verify
        what = fits.ImageHDU(frame, name=field)
        hdul.append(what)  # Write


def load_fits_field(fits_path, field='primary'):
    """Load a fits file from disk"""
    with fits.open(fits_path, cache=False) as hdul:
        hdul.verify('silentfix+warn')  # Verify
        return open_fits_hdul(hdul, field=field)  # Then Read


def open_fits_hdul(hdul, field='primary'):
    """Load a fits file from disk"""
    hdul.verify('silentfix+warn')  # Verify
    
    field_hdul = hdul[field]
    image = field_hdul.data
    center = [field_hdul.header['X0_MP'], field_hdul.header['Y0_MP']]
    
    hInd = determine_hIndex(hdul)
    found_hdul = hdul[hInd]
    wave = found_hdul.header['WAVELNTH']
    t_rec = found_hdul.header['T_OBS']
    
    return image, wave, t_rec, center


def determine_hIndex(hdul):
    """Find out which hInd has the data"""
    for hInd in range(10):
        try:
            var = hdul[hInd].header['WAVELNTH']
            break
        except Exception as e:
            print(hInd, e)
    return hInd
    
    


## Path Loading
# def load_all_paths(params):
#     load_img_paths(params)
#     load_fits_paths(params)

def load_img_paths(params, img_directory=None, ext=".png", absolute=True):
    """Finds the img series paths on disk"""
    paths, abs_paths = load_path_set(params.img_directory(img_directory), ext)
    return params.local_img_paths(abs_paths if absolute else paths)


def load_fits_paths(params, fits_directory=None, ext=".fits", absolute=True):
    """Finds the fits series paths on disk"""
    paths, abs_paths = load_path_set(params.fits_directory(fits_directory), ext)
    return params.local_fits_paths(abs_paths if absolute else paths)


def load_path_set(directory, ext='.fits'):
    """Gets the paths to matching ext files in given directory"""
    # print("Loading {} from {}...".format(ext, directory), end='', flush=True)
    all_paths = listdir(directory)
    ext_paths = [path for path in all_paths if ext in path]
    abs_ext_paths = [join(directory, path) for path in ext_paths]
    return ext_paths, abs_ext_paths


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

    
    
    
    # hdul.writeto()
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Then Read
    # wave, t_rec = hdul[0].header['WAVELNTH'], hdul[0].header['T_OBS']
    # image = hdul[0].data
    # image_data = str(wave), str(wave), t_rec, image.shape
    
# return image, image_data
from calendar import timegm
from os.path import dirname, abspath, join, isdir, split
from os import makedirs, getcwd, listdir
from time import time, localtime, strftime

from astropy.io import fits

archive_url = "http://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images

##  PATHS
# def build_paths(self):
#   """Make the file structure to hold the images"""
#   self.wave_directory = join(self.params.imgs_top_directory(), self.current_wave)
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


# def find_root_directory():
#     """Determine where to store the images"""
#     subdirectory_name = "sunback_images"
#
#     if __file__ in globals():
#         ddd = dirname(abspath(__file__))
#     else:
#         ddd = abspath(getcwd())
#
#     while "dropbox".casefold() in ddd.casefold():
#         ddd = abspath(join(ddd, ".."))
#
#     directory = join(ddd, subdirectory_name)
#     if not isdir(directory):
#         makedirs(directory)
#     return directory

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
#         for current_wave in use_wavelengths:
#             wave_bucket.append(join(download_path, current_wave))
#     return wave_bucket

#
# ## FILE IO
# def save_frame_to_fits_file(fits_path, out_array, field1="filtered"):
#     """Save a fits file to disk"""
#     # print("Saving Frame to Fits File")
#     with fits.open(fits_path, cache=False, reprocess_mode="update") as hdul:
#         # hdul.verify('silentfix+ignore')  # Then Verify
#         fit_frame = fits.ImageHDU(out_array, name=field1)
#         if field1 not in hdul:
#             hdul.append(fit_frame)  # Write
#         else:
#             hdul[field1] = fit_frame  # Write
#         hdul.close(output_verify='ignore')
#
#
# def load_this_fits_frame(fits_path, field1=-1):
#     """Load a fits file from disk"""
#     with fits.open(fits_path, cache=False) as hdul:
#         hdul.verify('silentfix+warn')  # Verify
#         return open_fits_hdul(hdul, field1=field1)  # Then Read
#
#
# def open_fits_hdul(hdul, field1=-1):
#     """Load a fits file from disk"""
#     hdul.verify('silentfix+ignore')  # Verify
#     hInd = determine_hIndex(hdul)
#     found_hdul = hdul[hInd]
#     field_hdul = hdul[field1] if hdul[field1] else found_hdul
#
#     field_exists = hasattr(field_hdul, "data") and field_hdul.data is not None
#     image_path = field_hdul.data if field_exists else found_hdul.data
#
#     wave = found_hdul.header['WAVELNTH']
#     t_rec = found_hdul.header['T_OBS']
#     center = [found_hdul.header['X0_MP'], found_hdul.header['Y0_MP']]
#
#
#     return image_path, wave, t_rec, center
#
#
# def determine_hIndex(hdul):
#     """Find out which hInd has the data"""
#     for hInd in range(10):
#         try:
#             hdul[hInd].header['WAVELNTH']
#             hdul[hInd].data
#             break
#         except Exception as e:
#             # print(hInd, e)
#             pass
#     return hInd
#







    # hdul.writeto()













    # Then Read
    # current_wave, t_rec = hdul[0].header['WAVELNTH'], hdul[0].header['T_OBS']
    # in_object = hdul[0].data
    # image_data = str(current_wave), str(current_wave), t_rec, in_object.shape

# return in_object, image_data
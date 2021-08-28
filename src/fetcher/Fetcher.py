# from utils.file_util import __find_ext_files_in_directory
import sys

# from utils.file_util import load_imgs_paths, load_fits_paths #, load_all_paths
from processor.Processor import Processor


class Fetcher(Processor):
    """Gets some data"""
    description = "Use an Unnamed Fetcher"
    filt_name = "Base Fetcher Class"
    # def __init__(self, params=None):
    #     # Initialize class variables
    #     self.load(params)
    
    def more_init(self):
        self.local_wave_directory = None
        self.image_folder = None
        self.movie_folder = None
        self.fits_folder = None
        self.fido_result = None
        self.fido_num = None

        self.local_fits_paths = []
        self.local_img_paths = []
        self.requested_files = []
        self.redownload = []
        self.file_size_mode = None
        self.temp_fits_pathbox = []
        self.waves_to_do = []

        self.start_time, self.start_time_long, self.start_string = '', '', ''
        self.end_time, self.end_time_long, self.end_time_string = '', '', ''
    
    def fetch(self, params=None):
        raise NotImplementedError()
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def load_fits(self, fits_directory=None, absolute=True):
    #     load_fits_paths(self.params, fits_directory, absolute=absolute)
    #     self.params.n_fits = len(self.params.local_fits_paths())
    #     print("   Loaded {} Fits Files From {}".format(self.params.n_fits, self.params.imgs_directory()))
    #
    # def load_imgs(self, img_directory=None, absolute=True):
    #     load_imgs_paths(self.params, img_directory, absolute=absolute)
    #     self.params.n_imgs = len(self.params.local_imgs_paths())
    #     print("   Loaded {} Img Files From {}".format(self.params.n_fits, self.params.fits_directory()))
    #
    # def load(self, fits_directory=None, img_directory=None, absolute=True):
    #     self.load_fits(fits_directory, absolute)
    #     self.load_imgs(img_directory, absolute)

# def load_all(self):
#     load_all_paths(self.params)

# def load_imgs_paths(self, imgs_directory=None):
#     """Loads the img series from disk"""
#     ext = ".png"
#     fits_paths, abs_fits_paths = \
#         load_set(self.params.imgs_directory(imgs_directory), ext)
#     self.params.local_fits_paths(fits_paths)
#     return fits_paths, abs_fits_paths
#
# def load_fits_paths(self, fits_directory=None):
#     """Loads the fits series from disk"""
#     ext = ".fits"
#     fits_paths, abs_fits_paths = \
#         load_set(self.params.fits_directory(fits_directory), ext)
#     self.params.local_fits_paths(fits_paths)

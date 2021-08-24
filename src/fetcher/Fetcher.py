# from utils.file_util import load_path_set
from utils.file_util import load_img_paths, load_fits_paths #, load_all_paths


class Fetcher:
    """Gets some data"""
    
    def __init__(self):
        # Initialize class variables
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
        self.params = None
    
    def fetch(self):
        raise NotImplementedError()
    
    def load_fits(self, fits_directory=None, absolute=True):
        load_fits_paths(self.params, fits_directory, absolute=absolute)
        self.params.n_fits = len(self.params.local_fits_paths())
        print("   Loaded {} Fits Files From Disk".format(self.params.n_fits))
    
    def load_imgs(self, img_directory=None, absolute=True):
        load_img_paths(self.params, img_directory, absolute=absolute)
        self.params.n_imgs = len(self.params.local_img_paths())
        print("   Loaded {} Img Files From Disk".format(self.params.n_fits))
    
    # def load_all(self):
    #     load_all_paths(self.params)
    
    # def load_imgs(self, img_directory=None):
    #     """Loads the img series from disk"""
    #     ext = ".png"
    #     fits_paths, abs_fits_paths = \
    #         load_set(self.params.img_directory(img_directory), ext)
    #     self.params.local_fits_paths(fits_paths)
    #     return fits_paths, abs_fits_paths
    #
    # def load_fits(self, fits_directory=None):
    #     """Loads the fits series from disk"""
    #     ext = ".fits"
    #     fits_paths, abs_fits_paths = \
    #         load_set(self.params.fits_directory(fits_directory), ext)
    #     self.params.local_fits_paths(fits_paths)

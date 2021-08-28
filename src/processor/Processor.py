from os import listdir
from os.path import join, dirname
from time import strftime

import cv2
from astropy.io import fits
from tqdm import tqdm

from utils.file_util import save_frame_to_fits_file


class Processor:
    """Top Level Class"""
    # name = 'data'
    in_name = 'primary'
    out_name = 'filtered'
    filt_name = "baseClass"
    description = "Use an Unnamed Processor"
    do_png = False
    quietly = True
    do_function = None
    params = None
    current_wave = None
    proc_name = None
    n_fits=0
    n_imgs=0
    
    def __init__(self, params=None):
        self.load(params)
    
    def plan(self):
        """Find the name of this processor and print"""
        self.proc_name = str(self.__class__).split(".")[-1][:-2]
        print('   ', self.proc_name, ":", self.description)
    
    ## M1: Look for files in a directory and return their paths ##
    
    def load(self, params=None, fits_directory=None, imgs_directory=None,
             absolute=True, in_name=None, out_name=None, quietly=True):
        """ M1
        Create and return two lists
            the fits files in params.fits_directory()
            the img  files in params.imgs_directory()
        """
        #  Refresh Params and Load Paths
        if params is not None:
            self.params = params
        if self.params is not None:
            self.quietly = quietly
            self.set_name(in_name, out_name, self.params.batch_name())
            
            if self.params.do_one():
                self.current_wave = self.params.do_one()
                self.params.build_paths(self.current_wave)
            else:
                self.current_wave = "rainbow"
                self.params.build_paths(self.current_wave)
            
            self.load_paths(fits_directory, imgs_directory, absolute)

            
    def load_paths(self, fits_directory=None, imgs_directory=None, absolute=True):
        """ Determines and lists the files that exist in the given directories"""
        #  Determine Directory
        self.params.fits_directory(fits_directory)
        self.params.imgs_directory(imgs_directory)
        
        #  Determine Paths
        fits_paths = self.load_fits_paths(absolute)
        img_paths = self.load_imgs_paths(absolute)
        
        self.n_fits = len(self.params.local_fits_paths())
        self.n_imgs = len(self.params.local_imgs_paths())
        
        return fits_paths, img_paths
    
    def load_fits_paths(self, absolute=True, ext=".fits"):
        """ Creates a List of the existant fits files in the fits_directory"""
        paths, abs_paths = self.__find_ext_files_in_directory(self.params.fits_directory(), ext)
        out_paths = self.params.local_fits_paths(abs_paths if absolute else paths)
        self.params.n_fits = len(self.params.local_fits_paths())
        if not self.quietly: ("   Found {} {} Files in {}".format(self.params.n_fits, ext, self.params.fits_directory()))
        return out_paths
    
    def load_imgs_paths(self, absolute=True, ext=".png"):
        """ Creates a List of the existant img files in the imgs_directory"""
        paths, abs_paths = self.__find_ext_files_in_directory(self.params.imgs_directory(), ext)
        out_paths = self.params.local_imgs_paths(abs_paths if absolute else paths)
        self.params.n_imgs = len(self.params.local_imgs_paths())
        if not self.quietly: print("   Found {} {} Files in {}".format(self.params.n_imgs, ext, self.params.imgs_directory()))
        return out_paths
    
    @staticmethod
    def __find_ext_files_in_directory(directory, ext='.fits'):
        """Returns the paths to matching ext files in given directory"""
        ext_paths = [path for path in listdir(directory) if ext in path]
        abs_ext_paths = [join(directory, path) for path in ext_paths]
        return ext_paths, abs_ext_paths

    # def count_objects(self, basket):
    #     self.n_obj = 0
    #     for obj in basket:
    #         self.n_obj += 1
    #     pass
    
    ########################################
    ## M2: For Every File in Path, do Func##
    
    def process(self, params=None):
        self.load(params)
        if self.do_png:
            self.process_img_series()
        else:
            self.process_fits_series()
    
    def set_function(self, func):
        self.do_function = func
    
    def set_name(self, in_name=None, out_name=None, name=None):
        if in_name: self.in_name = in_name
        if out_name: self.out_name = out_name
        if name: self.name = name
    
    def process_fits_series(self, params=None):
        """Apply the function to all necessary fits files"""
        # self.load(params)
        print(self.filt_name + "...", flush=True)
        
        self.ii = 0
        fits_paths = self.params.local_fits_paths()
        if len(fits_paths) > 0:
            for ii, fits_path in enumerate(tqdm(fits_paths, desc="  ")):
                self.modify_one_fits(fits_path, self.do_function)

                self.ii = ii
        if self.ii > 0:
            print("    Successfully Processed {} Files \n".format(self.ii), flush=True)
        else:
            print("    No Files Found")
            
    def modify_one_fits(self, fits_path, function):
        frame = function(fits_path, self.in_name).get()
        save_frame_to_fits_file(fits_path, frame, field=self.out_name)
        return frame
    
    ##############################################################
    def process_img_series(self, params=None):
        """Apply the function to all necessary img files"""
        # self.load(params)
        self.process_all_wavelengths(params)
    
    def process_all_wavelengths(self, params):
        """Run the process on all of the wavelengths"""
        self.load(params)
        print(self.filt_name + "...", flush=True)
        
        folders = self.get_folders()
        for wave in folders:
            which = self.params.do_one()
            if which and wave not in which:
                continue
            self.process_one_wavelength(wave)
    
    def get_folders(self):
        base = self.params.base_directory()
        bName = self.params.batch_name()
        folders = listdir(base)
        if "fits" in folders:
            folders = listdir(dirname(base))
        elif bName in folders:
            folders = listdir(join(base, bName))
        return folders
    
    def process_one_wavelength(self, wave):
        raise NotImplementedError()
        
        # fail_count = 0
        # img_paths = self.params.local_imgs_paths()
        # for ii, img_path in enumerate(tqdm(img_paths, desc="  ")):
        #     try:
        #         self.modify_one_img(img_path, self.do_function)
        #     except Exception as e:
        #         print(e)
        #         fail_count += 1
        #     self.ii = ii
        # print("    Success! {} Files Processed\n".format(self.ii+1))
    
    # def modify_one_img(self, img_path, function):
    #     in_object = function(img_path, self.in_field).get()
    #     # save_frame_to_fits_file(img_path, in_object, get_field=self.out_name)
    #     return in_object
    
    #
    # def process_one_wavelength(self, wave):
    #
    #     images = load_imgs_paths(self.params)
    #     img_directory = self.params.imgs_directory()
    #
    #     if len(images) > 0:
    #         in_object = cv2.imread(join(img_directory, images[0]))
    #         height, width, layers = in_object.shape
    #         final_name = self.video_name_stem.format("_raw.avi")
    #         print(final_name)
    #         video_avi = cv2.VideoWriter(final_name, 0, self.params.frames_per_second(), (width, height))
    #
    #         for in_object in tqdm(images, desc=">Writing Movie {}".format(wave), unit="in_object"):
    #             # print(join(self.image_folder, in_object))
    #             im = cv2.imread(join(self.image_folder, in_object))
    #             video_avi.write(im)
    #
    #         cv2.destroyAllWindows()
    #         video_avi.release()
    #     else:
    #         print("No png Images Found")
    #
    
    # def load_ext_paths(self, img_directory=None, absolute=True, ext=".png"):
    #     """        Creates a List of the existant img files in the imgs_directory
    #     """
    #     img_directory = self.params.imgs_directory(img_directory)
    #     img_paths  = self.params.local_imgs_paths
    #     num_found = self.params.n_fits
    #
    #     paths, abs_paths = self.__find_ext_files_in_directory(img_direct, ext)
    #     out_paths = abs_paths if absolute else paths
    #     out = img_paths(out_paths)
    #     self.params.n_imgs = len(self.params.local_imgs_paths())
    #     print("   Loaded {} Img Files From {}".format(num_found, self.params.imgs_directory()))
    #     return out
    
    # Initialize class variables
    # self.local_wave_directory = None
    # self.image_folder = None
    # self.movie_folder = None
    # self.fits_folder = None
    # self.fido_result = None
    # self.fido_num = None
    #
    # self.local_fits_paths = []
    # self.local_imgs_paths = []
    # self.requested_files = []
    # self.redownload = []
    # self.file_size_mode = None
    # self.temp_fits_pathbox = []
    # self.waves_to_do = []
    #
    # self.start_time, self.start_time_long, self.start_string = '', '', ''
    # self.end_time, self.end_time_long, self.end_time_string = '', '', ''
    # self.params = None
    
    # def load_imgs_paths(self, directory, parameter_setter, absolute=True, ext=".png"):
    #     """ Creates a List of the existant img files in the imgs_directory"""
    #     out_paths = self.load_ext_paths(directory, parameter_setter, absolute=absolute, ext=ext)
    #     self.params.imgs_directory(directory)
    #     self.params.n_imgs = len(out_paths)
    #     return out_paths
    #
    # def load_ext_paths(self, directory, parameter_setter, absolute=True, ext=".png"):
    #     """ Creates a List of the ext files in the directory"""
    #     paths, abs_paths = self.__find_ext_files_in_directory(directory, ext)
    #     out_paths = parameter_setter(abs_paths if absolute else paths)
    #     out_count = len(out_paths)
    #     print("   Loaded {} {} Files From {}".format(out_count, ext, directory))
    #     return out_paths
    a = 0

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


# ii=0

# def __init__(self, params):
# self.params = params
# self.done_paths = []
# self.skipped = 0
# self.name='data'
# self.in_field = 'primary'
# self.out_name = 'filtered'
# self.skipped = 0
# self.ii = 0

# def process_fits(self, params=None):
#     if params is not None:
#         self.params = params
#         load_fits_paths(self.params)
#
#     print(self.name+"...", flush=True)
#     self.modify_fits_series()
#     print("    Success! {} Files Filtered\n".format(self.ii+1))

# def modify_fits_series(self):
#     """Processes the fits series"""
#     for ii, img_path in enumerate(tqdm(self.params.local_fits_paths(), desc="  ")):
#         try:
#             in_object = self.modify_fits(img_path, self.in_field)
#         except Exception as e:
#             print(e)
#         self.ii = ii

# def modify_fits(self, img_path, function, in_field=None, out_name=None):
#     if in_field: self.in_field = in_field
#     if out_name: self.out_name = out_name
#     in_object = function(img_path, self.in_field).get()
#     save_frame_to_fits_file(img_path, in_object, get_field=self.out_name)
#     return in_object

# def modify_one_fits(self, img_path, function, in_field=None, out_name=None):
#     raise NotImplementedError()


# def process_all_wavelengths(self, p):
#     """Run the process on all of the wavelengths"""
#     self.__init__(p)
#
#     folders = listdir(self.params.base_directory())
#
#     for wave in folders:
#         which = self.params.do_one()
#         if which and wave not in which:
#             continue
#         self.process_one_wavelength(wave)
#
#
# def process_one_wavelength(self, wave):
#
#     images = load_imgs_paths(self.params)
#     img_directory = self.params.imgs_directory()
#
#     if len(images) > 0:
#         in_object = cv2.imread(join(img_directory, images[0]))
#         height, width, layers = in_object.shape
#         final_name = self.video_name_stem.format("_raw.avi")
#         print(final_name)
#         video_avi = cv2.VideoWriter(final_name, 0, self.params.frames_per_second(), (width, height))
#
#         for in_object in tqdm(images, desc=">Writing Movie {}".format(wave), unit="in_object"):
#             # print(join(self.image_folder, in_object))
#             im = cv2.imread(join(self.image_folder, in_object))
#             video_avi.write(im)
#
#         cv2.destroyAllWindows()
#         video_avi.release()
#     else:
#         print("No png Images Found")


# def build_paths(self, wave):
#     self.local_wave_directory = join(self.params.imgs_directory(), wave)
#     self.image_folder = join(self.local_wave_directory, 'png')
#     self.movie_folder = abspath(join(self.params.imgs_directory(), "movies\\"))
#     self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(wave, strftime('%m%d_%H%M'), '{}'))
#     # print(self.video_name_stem)
#     makedirs(self.movie_folder, exist_ok=True)

# images = [img for img in listdir(self.image_folder) if img.endswith(".png")] # and self.check_valid_png(img)]

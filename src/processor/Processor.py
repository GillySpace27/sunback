import os
import sys
from os import listdir, getcwd, makedirs
from os.path import join, dirname, abspath, isdir, basename
from time import strftime, sleep
import numpy as np
verb = False
import cv2
from astropy.io import fits
from tqdm import tqdm

# from utils.file_util import save_frame_to_fits_file
# from fetcher.FidoFetcher import vprint
from utils.file_util import discover_best_data_directory


def vprint(in_string, *args, **kwargs):
    if verb:
        print(in_string, *args, **kwargs)


class Processor:
    """Top Level Class"""
    # name = 'data'
    in_name = -1
    out_name = 'filtered'
    filt_name = "baseClass"
    batch_name = name = 'default_name'
    description = "Use an Unnamed Processor"
    progress_verb = " *    Processing Files"
    progress_unit = "files"
    style_mode = 'all'
    do_png = False
    quietly = True
    params = None
    current_wave = None
    proc_name = None
    changed = None
    original = None
    n_fits = None
    n_imgs = None
    ii = 0
    all_wavelengths = ['0171', '0193', '0211', '0304', '0131', '0335', '0094']
    all_wavelengths = ['0211', '0304', '0131', '0335']
    waves_to_do = all_wavelengths
    base_fits_dir = None
    base_imgs_dir = None
    base_absolute = None
    save_to_fits = True
    
    def __init__(self, params=None, quick=False):
        self.load(params, quick=quick)
    
    def plan(self):
        """Find the name of this processor and print"""
        self.proc_name = str(self.__class__).split(".")[-1][:-2]
        if 'null' not in self.proc_name.casefold():
            print('   ', self.proc_name, ":", self.description)
    
    def put(self, params=None):
        self.process(params)
    
    ##############################################################
    ## M1: Look for files in a directory and return their paths ##
    ##############################################################
    
    def load(self, params=None, fits_directory=None, imgs_directory=None,
             absolute=True, in_name=None, out_name=None, batch_name=None,
             quietly=True, wave=None, quick=False):
        """ M1
        Create and return two lists
            the fits files in params.fits_directory()
            the img  files in params.imgs_directory()
        """
        verb = not quietly
        if params is not None:
            self.params = params
        if self.params is not None:
            #  Refresh Params and Load Paths
            self.set_names(in_name, out_name, batch_name, quietly)
            self.set_base_directories(fits_directory, imgs_directory, absolute)
            self.set_waves_to_do(waves=None)
            self.set_current_wave(wave)
            if not quick:
                self.create_subdirectories()
            return self.load_paths(verb)
        else:
            # print("FAILED to load")
            pass
    
    # Define Targets
    def set_names(self, in_name=None, out_name=None, name=None, quietly=None):
        """Store the batch names into self"""
        if in_name:
            self.in_name = in_name
        if out_name:
            self.out_name = out_name
        if name:
            self.name = name
            self.params.batch_name(name)
        if quietly:
            self.quietly = quietly
    
    def set_base_directories(self, fits_directory=None, imgs_directory=None, absolute=None):
        """Store the directories into self"""
        if fits_directory:
            self.base_fits_dir = fits_directory
        if imgs_directory:
            self.base_imgs_dir = imgs_directory
        if absolute is not None:
            self.base_absolute = absolute
    
    def set_waves_to_do(self, waves=None):
        if waves is not None:
            self.waves_to_do = [waves]
        elif self.params.do_one():
            self.waves_to_do = [self.params.do_one()]
        else:
            self.waves_to_do = self.all_wavelengths
    
    def set_current_wave(self, wave=None):
        """Set the current wave parameter correctly"""
        if wave is not None:
            self.current_wave = wave
        elif self.params.do_one():
            self.current_wave = self.params.do_one()
        else:
            self.current_wave = self.all_wavelengths[0]
        self.set_current_wave_paths()
    
    def set_current_wave_paths(self):
        """Make the paths for current_wave"""
        # Define and Set Directories
        # print("Target: {}".format(self.current_wave))
        base_directory = self.discover_base_directory()
        self.params.base_directory(abspath(base_directory))
        self.params.imgs_directory(abspath(join(base_directory, 'png')))
        self.params.fits_directory(abspath(join(base_directory, 'fits')))
        self.params.movs_directory(abspath(join(base_directory, "..", 'MOVS')))
        self.params.time_path(join(base_directory, "image_times.txt"))
    
    def create_subdirectories(self):
        # Make Directories
        makedirs(self.params.imgs_directory(), exist_ok=True)
        makedirs(self.params.fits_directory(), exist_ok=True)
        if "background" not in self.params.movs_directory():
            makedirs(self.params.movs_directory(), exist_ok=True)
        # Save Parameters
        # self.params.save_to_txt(self.current_wave)
    
    def load_paths(self, verb=False):
        """ Determines and lists the files that exist in the given directories"""
        fits_paths, imgs_paths = self.load_fits_paths(), self.load_imgs_paths()
        self.print_load_banner(verb)
        return fits_paths, imgs_paths
    
    def print_load_banner(self, verb):
        if self.n_fits + self.n_imgs > 0 and verb:
            print(" *   Processing: {}".format(self.current_wave))
            vprint(" *   Loaded {} fits and {} imgs from {}\n".format(self.n_fits, self.n_imgs, self.params.base_directory()), verb)
    
    def load_fits_paths(self, absolute=True, ext=".fits"):
        """ Creates a List of the existant fits files in the fits_directory"""
        paths, abs_paths = self.__find_ext_files_in_directory(self.params.fits_directory(), ext)
        out_paths = self.params.local_fits_paths(abs_paths if absolute else paths)
        self.n_fits = self.params.n_fits = len(self.params.local_fits_paths())
        if not self.quietly: ("   Found {} {} Files in {}".format(self.params.n_fits, ext, self.params.fits_directory()))
        return out_paths
    
    def load_imgs_paths(self, absolute=True, ext=".png"):
        """ Creates a List of the existant img files in the imgs_directory"""
        paths, abs_paths = self.__find_ext_files_in_directory(self.params.imgs_directory(), ext)
        out_paths = self.params.local_imgs_paths(abs_paths if absolute else paths)
        self.n_imgs = self.params.n_imgs = len(self.params.local_imgs_paths())
        if not self.quietly: print("   Found {} {} Files in {}".format(self.params.n_imgs, ext, self.params.imgs_directory()))
        return out_paths
    
    @staticmethod
    def __find_ext_files_in_directory(directory, ext='.fits'):
        """Returns the paths to matching ext files in given directory"""
        ext_paths = [path for path in listdir(directory) if ext in path]
        abs_ext_paths = [join(directory, path) for path in ext_paths]
        return ext_paths, abs_ext_paths
    
    def discover_base_directory(self):
        """Define the root folder"""
        root = discover_best_data_directory()
        base_directory = join(root, self.params.batch_name())
        base_directory = join(base_directory, self.current_wave)
        return self.params.base_directory(base_directory)
    
    ########################################
    ## M2: For Every File in Path, do Func##
    ########################################
    # def set_function(self, func):
    #     """This is the function which gets applied to every file in the directory
    #     Must follow the form:
    #             out_frame = func(fits_path, frame_name)
    #     """
    #     self.do_fits_function = func
    
    def do_fits_function(self, fits_path, in_name):
        raise NotImplementedError
    
    def process(self, params=None):
        print(' v', self.filt_name + "...", flush=True)
        self.load(params, quietly=False)
        self.super_flush()
        
        if self.params is not None:
            if self.do_png:
                self.process_img_series()
            else:
                self.process_fits_series()
    
    ##  Fits Files
    def process_fits_series(self):
        """Apply the function to all necessary fits files"""
        if len(self.params.local_fits_paths()) > 0:
            for self.ii, fits_path in enumerate(tqdm(
                    self.params.local_fits_paths(),
                    unit=self.progress_unit,
                    desc=self.progress_verb)):
                
                self.modify_one_fits(fits_path)
        
        
        n_success = self.ii + 1 #- self.skipped
        if n_success > 1:
            
            print(" ^    Successfully Processed {} Files \n".format(n_success), flush=True)
        else:
            print(" ^    No Files Found")
    
    def modify_one_fits(self, fits_path):
        """Apply the given funtion to the given fits path"""
        self.confirm_fits_file(fits_path)
        try:
            output = self.do_fits_function(fits_path, self.in_name)
            try:
                frame = output.get()

            except AttributeError as e:
                # print(e)
                frame = output
        except IndexError as e:
            print("Failed to Load Fits Frame")
            print(e)
            return self.original
        if self.save_to_fits and frame is not None:
            self.save_frame_to_fits_file(fits_path, frame)
        return frame
    
    def confirm_fits_file(self, fits_path):
        if os.path.exists(fits_path):
            return
        else:
            raise FileNotFoundError
    
    ##  Img Files
    def process_img_series(self):
        """Apply the function to all necessary img files"""
        self.process_all_wavelengths()
    
    def process_all_wavelengths(self):
        """Run the process on all of the all_wavelengths"""
        # print(self.filt_name + ">>>", flush=True)
        
        folders = self.get_folders()
        for wave in folders:
            # which = self.params.do_one()
            if wave in self.waves_to_do and len(self.params.local_imgs_paths()) > 0:
                # if which and wave not in which:
                #     continue
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
    
    ########################################
    ## M3: Identify Directory of Interest ##
    ########################################
    
    def discover_best_root_directory(self, subdirectory_name="sunback_images"):
        """Determine where to store the images"""
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
    
    ############################
    ## M4: Save Frame to Fits ##
    ############################
    
    def save_frame_to_fits_file(self, fits_path, frame):
        """Save a fits file to disk"""
        # print("Saving Frame to Fits File")
        field = self.out_name
        with fits.open(fits_path, cache=False, mode="update") as hdul:
            # hdul.verify('silentfix+ignore')  # Then Verify
            if frame.dtype in [float, np.float32]:
                frame = self.smallify_frame(frame)
            fit_frame = fits.ImageHDU(frame, name=field)
            if field not in hdul:
                hdul.append(fit_frame)  # Write
            else:
                hdul[field] = fit_frame  # Write
            try:
                hdul.close(output_verify='ignore')
            except PermissionError as e:
                vprint('\n      Failed to Close HDU', "processor.py::  ", e)
    
    def load_last_fits_field(self, fits_path):
        """Load a fits file from disk"""
        return self.load_a_fits_field(fits_path, -1)
    
    def load_first_fits_field(self, fits_path):
        """Load a fits file from disk"""
        return self.load_a_fits_field(fits_path, 0)
    
    def load_a_fits_field(self, fits_path, field=0):
        """Load a fits file from disk"""
        with fits.open(fits_path, cache=False) as hdul:
            hdul.verify('silentfix+ignore')  # Verify
            self.ensure_no_double_filtering(hdul)
            wave, t_rec, center = self.get_fits_info(hdul)
            frame = hdul[field].data
        return frame, wave, t_rec, center
    
    def load_best_fits_field(self, fits_path):
        """Load a fits file from disk"""
        with fits.open(fits_path, cache=False) as hdul:
            hdul.verify('silentfix+ignore')  # Verify
            self.in_name= self.ensure_no_double_filtering(hdul)
            wave, t_rec, center = self.get_fits_info(hdul)
            frame = self.open_fits_hdul(hdul)
            
        return frame, wave, t_rec, center
    
    def smallify_frame(self, frame):
        mx = np.nanmax(frame)
        mn = np.nanmin(frame)
        normed = (frame - mn) / (mx - mn)
        
        scaled = normed * 2 ** 16
        average = np.uint16(np.round(np.nanmean(scaled)))
        de_NANed = np.nan_to_num(scaled, nan=average)
        compressed = de_NANed.astype(np.uint16)
        
        return compressed
    
    # def smallify_frame(self, frame):
    #
    #     return frame.astype(np.float16)
    
    def get_fits_info(self, hdul):
        # Load the original frame
        wave, t_rec, center = None, None, None
        for ii in range(len(hdul)):
            try:
                first_data_hdul = hdul[ii]
                wave = first_data_hdul.header['WAVELNTH']
                t_rec = first_data_hdul.header['T_OBS']
                center = [first_data_hdul.header['X0_MP'], first_data_hdul.header['Y0_MP']]
                break
            except KeyError as e:
                continue
        self.first_hIndex = ii
        return wave, t_rec, center
    
    def open_fits_hdul(self, hdul):
        """Load a fits file from disk"""

        # Load the called for frame
        if 'str' in str(type(self.in_name)):
            field_hdu = hdul[self.in_name]
        elif self.in_name is None:
            return None
        else:
            field_hdu = hdul[self.hdu_name_list[self.in_name]]
        return field_hdu.data
    
    def ensure_no_double_filtering(self, hdul):
        """Determine which frame of the input file to use on redo"""
        self.list_hdus(hdul)
        if self.in_name is None:
            return None
        reprocess_mode         = self.params.reprocess_mode()
        input_frame_name       = self.determine_in_frame_name()
        output_frame_name      = self.determine_out_frame_name()
        first_frame_name       = self.hdu_name_list[0]
        penultimate_frame_name = self.hdu_name_list[-2]
        
        if input_frame_name.casefold() == output_frame_name.casefold():
            # If you're about to redo a filter
            
            if reprocess_mode == 'skip' or reprocess_mode is False:
                # Skip it
                self.in_name = None
                # raise FileExistsError("Skipping File")
            elif reprocess_mode == 'redo' or reprocess_mode is True:
                # Go to the previous frame and remake
                self.in_name = penultimate_frame_name
            elif reprocess_mode == 'reset':
                # Go to the first frame and remake
                self.in_name = first_frame_name
            elif reprocess_mode == 'double':
                # Repeat the filter a second time
                self.in_name = output_frame_name
            else:
                raise NotImplementedError
        else:
            self.in_name = input_frame_name
        return self.in_name
 
    def determine_in_frame_name(self):
        # Determine the called-for input frame NAME
        if self.in_name is None:
            return None
        if type(self.in_name) is str:
            if self.in_name in self.hdu_name_list:
                input_frame_name = self.in_name.casefold()
            else:
                raise FileNotFoundError
        else:
            input_frame_name = self.hdu_name_list[self.in_name].casefold()
        return input_frame_name

    def determine_out_frame_name(self):
        # Determine the called-for output frame NAME
        if type(self.out_name) is str:
            output_frame_name = self.out_name.casefold()
        else:
            output_frame_name = self.hdu_name_list[self.out_name].casefold()
        return output_frame_name
   
    def determine_first_hIndex(self, hdul):
        """Find out which hInd has the data"""
        for hInd in range(10):
            try:
                a = hdul[hInd].header['WAVELNTH']
                a = hdul[hInd].data
                break
            except Exception as e:
                pass
        return hInd
    
    ## UTIL
    def get(self):
        """Return just the changed frome"""
        return self.changed
    
    def get_orig(self):
        """Return just the original frome"""
        return self.original
    
    def super_flush(self, many=5):
        """Flush the stdout many times"""
        for ii in range(many):
            sys.stdout.flush()
            sys.stderr.flush()
            sleep(0.1)
    
    def list_hdus(self, hdul):
        self.hdu_name_list = [name.name.casefold() for name in hdul]
        # print(self.hdu_name_list)
        # self.printout_hdul(hdul)
        return self.hdu_name_list
    
    def printout_hdul(self, hdul):
        print("\n\n**Examining Hdul**")
        
        for h_num in range(len(hdul)):
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")
            print("\n  HDUL #", h_num)
            for h_info in hdul.fileinfo(h_num):
                print("    ", h_num, " : ", h_info, "\t : \t", hdul.fileinfo(h_num)[h_info])
            
            to_find = ['bound method', "built-in method", 'method-wrapper']
            HDU = hdul[h_num]
            
            print("\n  Hdul Fields")
            self.print_without(HDU, to_find)
            
            for ff in to_find:
                self.print_with(HDU, ff)
    
    def print_without(self, HDU, not_wanted=None):
        print("      ** " + "Remainder without these:  " + str(not_wanted) + " **")
        ban_list = [str(it) for it in not_wanted]
        
        found_list = []
        for found_field_name in dir(HDU):
            found_field_value = getattr(HDU, found_field_name)
            found_field_value_string = str(found_field_value)
            skip = False
            for bb in ban_list:
                if bb in found_field_value_string:
                    skip = True
            if skip:
                continue
            
            found_list.append([found_field_name, found_field_value])
            
            # Print these
            out_string = "      {}".format("Misc") + ':: ' + found_field_name
            out_2 = "\t  :  \t" + found_field_value_string
            print("{0: <35}".format(out_string.replace("\n", " ")), out_2)
        print("\n\n")
        return found_list
    
    def print_with(self, HDU, wanted=None):
        print("    ** " + str(wanted) + " **")
        found_list = []
        for found_field_name in dir(HDU):
            found_field_value = getattr(HDU, found_field_name)
            found_field_value_string = str(found_field_value)
            
            if wanted in str(found_field_value_string):
                found_list.append([found_field_name, found_field_value])
                out_string = "      {}".format(wanted) + ':: ' + found_field_name
                out_2 = "\t  :  \t" + found_field_value_string
                print("{0: <35}".format(out_string), out_2)
        print("\n\n")
        
        return found_list
        
        # fail_count = 0
        # img_paths = self.params.local_imgs_paths()
        # for ii, img_path in enumerate(tqdm(img_paths, desc="  ")):
        #     try:
        #         self.modify_one_img(img_path, self.do_fits_function)
        #     except Exception as e:
        #         print(e)
        #         fail_count += 1
        #     self.ii = ii
        # print("    Success! {} Files Processed\n".format(self.ii+1))
    
    # def modify_one_img(self, img_path, function):
    #     in_object = function(img_path, self.in_field).get()
    #     # save_frame_to_fits_file(img_path, in_object, get_field=self.out_name)
    #     return in_object

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


# def build_paths(self, wave):
#     self.local_wave_directory = join(self.params.imgs_directory(), wave)
#     self.image_folder = join(self.local_wave_directory, 'png')
#     self.movie_folder = abspath(join(self.params.imgs_directory(), "movies\\"))
#     self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(wave, strftime('%m%d_%H%M'), '{}'))
#     # print(self.video_name_stem)
#     makedirs(self.movie_folder, exist_ok=True)

# images = [img for img in listdir(self.image_folder) if img.endswith(".png")] # and self.check_valid_png(img)]

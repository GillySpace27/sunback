import os
import sys
from copy import copy, deepcopy
from os import listdir, getcwd, makedirs
from os.path import join, dirname, abspath, isdir, basename
import astropy.units as u
from time import sleep

import cv2
import numpy as np

from run import SingleRunner
from science.color_tables import aia_color_table

# import cv2
from astropy.io import fits
from tqdm import tqdm

import matplotlib.pyplot as plt
# %matplotlib notebook
# from utils.file_util import save_frame_to_fits_file
# from fetcher.FidoFetcher import vprint
# from utils.file_util import find_root_directory

verb = True

def vprint(in_string, *args, **kwargs):
    if verb:
        print(in_string, *args, **kwargs)


class Processor:
    """Top Level Class"""
    # name = 'data'
    in_name = -1
    out_name = 'filtered'
    filt_name = "Base Processor Class"
    batch_name = name = 'default_name'
    description = "Use an Unnamed Processor"
    run_type = "General Base Processor Class"
    progress_stem = " *    {} Files"
    progress_verb = "Processing"
    progress_string = progress_stem.format(progress_verb)
    finished_verb = "Processed"
    progress_unit = "files"
    run_type_string = "Default Actions"
    out_path = None
    
    style_mode = 'all'
    do_png = False
    quietly = True
    params = None
    # current_wave = 'rainbow'
    proc_name = None
    modified_image = None
    original_image = None
    n_fits = None
    n_imgs = None
    ii = 0
    # all_wavelengths = ['0211', '0304', '0131', '0335']
    
    # waves_to_do = all_wavelengths
    dont_ignore = False
    keyframes = []
    _reprocess_mode = None
    base_fits_dir = None
    base_imgs_dir = None
    base_absolute = None
    save_to_fits = True
    can_use_keyframes = False
    this_file_name = os.path.basename(__file__)
    paper_out = []
    
    load_print_latch = True
    found_limb_radius = None
    int_tm_tot = None
    fits_folder = None
    absolute_min = None
    curve_out_array = None
    ensured = False
    hdu_name_list = None
    file_basename = None
    image_data = None
    changed_flat = None
    can_use_keyframes = False
    use_keyframes = None
    skipped = 0
    all_file_paths = None
    n_all_frames = None
    n_do_frames = None
    long_list = []
    fits_path = None
    first_hIndex = 0
    short_list = []
    out_dtype = "float32"
    
    def __init__(self, params=None, quick=False, rp=None):

        self.reprocess_mode(rp)
        self.load(params, quick=quick)
        if self.params:
            self.run_type_str = "\\item ({}) {}".format(self.this_file_name, self.run_type)
            self.paper_out.append(self.run_type_str)
    
    @staticmethod
    def plan(self):
        """Find the name of this processor and print"""
        self.proc_name = self.filt_name
        if 'null' not in self.proc_name.casefold():
            proc_name = self.proc_name + "\t : \t" + self.description
            print('      ' + proc_name)
            
    def put(self, params=None):
        self.process(params)

    def fetch(self, params=None):
        self.process(params)
    
    def reprocess_mode(self, flag=None):
        if flag is not None:
            if type(flag) is not bool:
                if 'skip' == flag:
                    flag = False
                if "redo" == flag:
                    flag = True
            self._reprocess_mode = flag
        
        return self._reprocess_mode
    
    ##############################################################
    ## M1: Look for files in a directory and return their paths ##
    ##############################################################
    
    def load(self, params=None, fits_directory=None, imgs_directory=None,
             absolute=True, in_name=None, out_name=None, batch_name=None,
             quietly=True, wave=None, quick=False):
        """ M1
        Create and return two lists
            the fits files in params.fits_directory()
            the img  files in params.imgs_top_directory()
        """
        verb = not quietly
        if params is not None:
            self.params = params
        self.set_base_directories(fits_directory, imgs_directory, absolute)
        
        self.set_names(in_name, out_name, batch_name, quietly)
        self.progress_string = self.progress_stem.format(self.progress_verb)
        
        if self.params is not None:
            #  Refresh Params and Load Paths
            self.name = self.params.batch_name(batch_name)
            self.super_flush()
            self.params.set_current_wave(wave)
            self.select_keyframe_subset()
            # self.params.create_subdirectories()  #Gender
            fits_paths, imgs_paths = self.load_paths(verb)
            return fits_paths, imgs_paths
 
    # def clean_directory(self):
    #     to_rep = "D:/"
    #     if not self.params.base_directory()[0] == to_rep[0]:
    #         self.params.base_directory(self.params.base_directory().replace(to_rep,""))
    #         if self.out_path:
    #             self.out_path = self.out_path.replace(to_rep,"")
    
    # Define Targets
    def set_names(self, in_name=None, out_name=None, name=None, quietly=None):
        """Store the batch names into self"""
        if in_name:
            self.in_name = in_name
        if out_name:
            self.out_name = out_name
        if name:
            self.name = name
        if quietly:
            self.quietly = quietly
    
    def set_base_directories(self, fits_directory=None, imgs_directory=None, absolute=None):
        """Store the directories into self"""
        if fits_directory:
            self.base_fits_dir = fits_directory
        elif self.base_fits_dir is None:
            self.base_fits_dir = self.params.fits_directory()
        if imgs_directory:
            self.base_imgs_dir = imgs_directory
        elif self.base_imgs_dir is None:
            self.base_imgs_dir = self.params.mods_directory()
            
        if absolute is not None:
            self.base_absolute = absolute
    
    def load_paths(self, verb=False):
        """ Determines and lists the files that exist in the given directories"""
        fits_paths, imgs_paths = self.load_fits_paths(), self.load_imgs_paths()
        self.print_load_banner(verb)
        return fits_paths, imgs_paths
    
    def print_load_banner(self, verb=False):
        if self.n_fits + self.n_imgs > 0 and verb:
            print('\r v {}...  ------------------------------------------------  v'.format(self.filt_name), flush=True)
            if self.progress_verb.casefold() in ["binning"]:
                exp = self.params.exposure_time_seconds()
                print(" *    Exposure Time is {} seconds, which is {} frames".format(exp, exp / 12))
            print(" +    {}: {}, Redo = {}".format(self.progress_verb, self.params.current_wave(), self.reprocess_mode()))
            vprint(" +    Using {} fits and {} imgs from {}\n".format(self.n_fits, self.n_imgs, self.params.base_directory()))
    
    def load_fits_paths(self, absolute=True, ext=".fits"):
        """ Creates a List of the existant fits files in the fits_directory"""
        paths, abs_paths = self.__find_ext_files_in_directory(self.params.fits_directory(), ext)
        self.fits_folder = self.params.fits_directory()
        out_paths = self.params.local_fits_paths(abs_paths if absolute else paths)
        self.n_fits = self.params.n_fits = len(self.params.local_fits_paths())
        if not self.quietly: ("   Found {} {} Files in {}".format(self.params.n_fits, ext, self.params.fits_directory()))
        return out_paths
    
    def load_imgs_paths(self, absolute=True, ext=".png"):
        """ Creates a List of the existant img files in the imgs_top_directory"""
        paths, abs_paths = self.__find_ext_files_in_directory(self.params.mods_directory(), ext)
        out_paths = self.params.local_imgs_paths(abs_paths if absolute else paths)
        self.n_imgs = self.params.n_imgs = len(self.params.local_imgs_paths())
        if not self.quietly: print("   Found {} {} Files in {}".format(self.params.n_imgs, ext, self.params.imgs_top_directory()))
        return out_paths
    
    @staticmethod
    def __find_ext_files_in_directory(directory, ext='.fits'):
        """Returns the paths to matching ext files in given directory"""
        makedirs(directory, exist_ok=True)
        ext_paths = [path for path in listdir(directory) if ext in path]
        abs_ext_paths = [join(directory, path) for path in ext_paths]
        return ext_paths, abs_ext_paths
    
    def load_fits_image(self, fits_path=None, in_name=None):
        """open the fits file and grab the necessary data"""
        
        if fits_path is not None:
            self.fits_path = os.path.normpath(fits_path)
        if self.fits_path is None:
            self.fits_path = self.params.local_fits_paths()[0]
        
        frame, wave, t_rec, center, int_time, img_type = self.load_best_fits_field(self.fits_path, in_name)
        
        if frame is not None and img_type.casefold() != 'dark':
            self.params.original_image = np.asarray(frame, dtype=np.float32)
            self.params.original_image2 = np.asarray(frame, dtype=np.float32)
            self.params.modified_image = copy(self.params.original_image)
            
            self.params.cmap = aia_color_table(int(wave) * u.angstrom)
            
#             self.original_flat = self.original_image.flatten()
#             self.changed_flat = self.modified_image.flatten()
            
            self.image_data = str(wave), self.fits_path, t_rec, frame.shape
            self.file_basename = basename(self.fits_path)
            self.set_centerpoint(center)
            self.params.image_data = self.image_data
            return True
        else:
            print("Skipped Fits!")
            if img_type.casefold() == 'dark':
                self.delete_fits_and_png(fits_path)
            return False
    
    def delete_fits_and_png(self, fits_path):
        # fitsPath = join(self.fits_folder, filename[:-5] + '.fits')
        pngPath = fits_path.replace("fits", "png")
        try:
            os.remove(fits_path)
        except PermissionError as e:
            print(e)
        try:
            os.remove(pngPath)
        except FileNotFoundError as e:
            print(e)
            pass
    
    def plot_two(self, name="Algorithm Result", bounds=None):
        fig, (ax0, ax1) = plt.subplots(1,2, sharex=True, sharey=True, num=name)

        org = self.prep_one(self.params.original_image)
        mod = self.prep_one(self.params.modified_image)
        
        # self.view_original(fig, ax0)
        ax0.imshow(org, cmap = self.params.cmap)
        ax1.imshow(mod, cmap = self.params.cmap)
        
        if bounds is None:
            ax1.set_xlim((0,1500))
            ax1.set_ylim((600,2000))
        
        # else:
        #     ax1.set_xlim((3400,4000))
        #     ax1.set_ylim((2300,3200))
        
        ax0.set_title("Original")
        ax1.set_title("Changed")
        
        plt.tight_layout()
        
        plt.show()

    def prep_one(self, img):
        minmin = np.nanmin(img)
        return img - minmin
    
    def set_centerpoint(self, center):
        """Parse the centerpoint and ensure correct scaling"""
        self.params.center = center
        image_edge = self.params.original_image.shape
        center_given = np.abs(self.params.center)
        
        Top_Tolerance = 0.65
        Bottom_Tolerance = 0.35
        count = 0
        while count < 10:
            ratio = center_given / image_edge
            if np.array(ratio > Top_Tolerance).any():
                center_given *= 0.5
            elif np.array(ratio < Bottom_Tolerance).any():
                center_given *= 2
            else:
                break
        self.params.center = center_given
    
    def select_keyframe_subset(self):
        """Sets the list of which frames get used as keyframes
        This function only runs once, sort of an __init__
        """
        # if self.dont_ignore:
        self.use_keyframes = (self.params.fixed_cadence_keyframes() or self.params.fixed_number_keyframes()) and self.can_use_keyframes
        if self.use_keyframes:
            self.keyframes = self.pick_keyframes()
            pass
        else:
            self.keyframes = self.pick_keyframes(use_all=True)
        pass
        # self.dont_ignore = False
    
    def pick_keyframes(self, use_all=False):
        """Decide which frames to use in the analysis"""
        # self.load(self.params, wave=self.params.current_wave)
        self.params.set_current_wave()
        if self.all_file_paths in [None, []]:
            self.all_file_paths = self.load_fits_paths()
        self.long_list = copy(self.all_file_paths)
        self.n_all_frames = len(self.long_list)
        n_paths = len(self.long_list)
        # self.short_list = []
        if self.n_all_frames < 10:
            use_all = True
            
        if use_all:
            self.short_list = self.long_list
        
        elif self.params.fixed_cadence_keyframes():
            # Fixed Cadence of one out of every {} frames
            self.short_list = self.long_list[::self.params.fixed_cadence_keyframes()]
        
        elif self.params.fixed_number_keyframes():
            #  Fixed Number of Keyframes
            skip = max(n_paths // self.params.fixed_number_keyframes(), 1)
            self.short_list = self.long_list[::skip]
        self.n_do_frames = len(self.short_list)
        return self.short_list
    
    def print_keyframes(self):
        if self.params.fixed_cadence_keyframes():
            print("\r *    >> KeyFrames: Fixed Cadence of one out of every {} frames".format(self.params.fixed_cadence_keyframes()))
        elif self.params.fixed_number_keyframes():
            print("\r *    >> KeyFrames: Fixed Number of Keyframes: {}".format(self.params.fixed_number_keyframes()))
        
        print(" *    >> Selected {} keyframes out of {} total frames".format(self.n_do_frames, self.n_all_frames))
        # print(" *    >>Selected {} keyframes out of {} total frames".format(len(self.short_list), len(self.long_list)))
        
        self.super_flush(many=10)
    
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
    
    def do_img_function(self):
        raise NotImplementedError
    
    def cleanup(self):
        pass
    
    def setup(self):
        pass
    
    def process(self, params=None):
        """Load the parameters and run the algorithm"""
        self.load(params, quietly=False)
        self.super_flush()
        
        if self.params is not None:
            if self.params.do_single:
                self.process_image()
            elif self.do_png:
                self.process_img_series()
            else:
                self.process_fits_series()
                
    def process_image(self):
        """Apply the function to a single image"""
        self.setup()
        self.modify_one_image()
        self.cleanup()
        
    ##  Run on Fits Files
    def process_fits_series(self):
        """Apply the function to all necessary fits files"""
        n_fits_path = len(self.keyframes)
        self.skipped = 0
        
        if n_fits_path > 0:
            self.setup()
            for self.ii, fits_path in enumerate(tqdm(
                    self.keyframes,
                    unit=self.progress_unit,
                    desc=self.progress_string,
                    # probesize='50M',
                    )):
                
                # print("NUM = ", self.ii)
                out = self.modify_one_fits(fits_path)
                if out is None:
                    self.skipped += 1
            self.cleanup()
        
        n_success = self.ii + 1 - self.skipped
        if n_success + self.skipped > 1:
            if n_success == 0:
                print(" X x X-- Skipped all {} Files --xXxXxXxXxXxXxXxXxXxXxX \n".format(self.skipped))
            else:
                print(" ^ ^ ^Successfully {} {} Files ({} skipped) \n".format(self.finished_verb, n_success, self.skipped), flush=True)
                print(" ^ ------------------------------------------------------------------------  ^\n")
        else:
            print(" ^    No Files Found\n")
    def confirm_fits_file(self, fits_path):
        if os.path.exists(fits_path):
            return True
        else:
            raise FileNotFoundError
        
    def modify_one_fits(self, fits_path):
        """Apply the given funtion to the given fits path"""
        self.confirm_fits_file(fits_path)
        output = self.do_fits_function(fits_path, self.in_name)
        try:
            frame = output.get()
        except AttributeError as e:
            # print(e)
            frame = output
    
        if self.save_to_fits and frame is not None:
            self.save_frame_to_fits_file(fits_path, frame, dtype=self.out_dtype)
            # def save_frame_to_fits_file(fits_path, frame, out_name=None, dtype="float32"):

        return frame
    


    ##  Img Files
    
    def modify_one_image(self,):
        """Apply the given funtion to the given fits path"""
        
        try:
            self.params.modified_image = self.do_img_function()
        except NotImplementedError as e:
            self.params.modified_image = self.do_fits_function(self.params.use_image_path())

        # try:
        #     frame = output.get()
        # except AttributeError as e:
        #     # print(e)
        #     frame = output
        #
        # # if self.save_to_fits and frame is not None:
        # #     self.save_frame_to_fits_file(fits_path, frame)
        # return frame


    
    def process_img_series(self):
        """Apply the function to all necessary img files"""
        self.do_one_wave(self.params.current_wave())
        
        # self.process_all_wavelengths()
    
    def process_all_wavelengths(self):
        """Run the process on all of the all_wavelengths"""
        # print(self.filt_name + ">>>", flush=True)
        
        folders = self.get_folders()
        for wave in folders:
            self.do_one_wave(wave)
    
    def do_one_wave(self, wave):
        if wave in self.params.waves_to_do:
            self.load(wave=wave)
            if len(self.params.local_imgs_paths()) > 0:
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
    
    def save_frame_to_fits_file(self, fits_path, frame, out_name=None, dtype="float32"):
        """Save a fits file to disk"""
        # print("Saving Frame to Fits File")
        if out_name is None:
            field = self.out_name
        else:
            field = out_name
        good_frame = np.any(frame)
        
        if good_frame:
            
            frame2 = frame.astype(dtype)
            
            with fits.open(fits_path, cache=False, mode="update", ignore_missing_end=True) as hdul:
                hdul.verify('silentfix+ignore')  # Then Verify
                self.remove_blank_frames(hdul) # THis might not work
                fit_frame = fits.ImageHDU(frame2, name=field, header=self.header)
                # fit_frame = fit_frame
                if field not in hdul:
                    hdul.append(fit_frame)  # Write
                else:
                    hdul[field] = fit_frame  # Write
                    
                hdul = self.delete_further_hdus(hdul, field)
                hdul[0].header['total_int_time'] = self.int_tm_tot
                hdul.close(output_verify='fix')
    
            
    def make_shortcut(self, file_in_path=None, shortcut_out_path=None, doAppend=True):
        path = self.params.shortcut_directory(shortcut_out_path)
        # import os, winshell, win32com.client, Pythoncom
        import os, win32com.client
        
        basename = os.path.basename(file_in_path)
        basename = basename.replace("___raw.avi", '')
        basename = basename.replace("__comp.avi", '')
        basename = basename.replace("_small.avi", '')
        if doAppend: path = os.path.join(path, '{}.lnk'.format(basename))
        # print(path)
        
        
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = path
        shortcut.IconLocation = file_in_path
        shortcut.save()
    
    def delete_further_hdus(self, hdul, field):
        try:
            self.list_hdus(hdul)
            ii = self.hdu_name_list.index(field)+1
            return hdul[0:ii]
        except ValueError as e:
            # print(e)
            return hdul
        
    def load_last_fits_field(self, fits_path):
        """Load a fits file from disk"""
        return self.load_a_fits_field(fits_path, -1)
    
    def load_first_fits_field(self, fits_path):
        """Load a fits file from disk"""
        fields = self.load_a_fits_field(fits_path, 0)
        if fields[0] is None:
            fields = self.load_a_fits_field(fits_path, 1)
        return fields
    
    def load_a_fits_field(self, fits_path, field=None):
        """Load a fits file from disk"""
        with fits.open(fits_path, cache=False, ignore_missing_end=True) as hdul:
            hdul.verify('silentfix+ignore')  # Verify
            self.list_hdus(hdul)
            # self.in_name = self.find_correct_in_name(hdul)
            frame = self.load_single_frame(hdul, field)
            wave, t_rec, center, int_time, self.found_limb_radius = self.get_fits_info(hdul)
            
        return frame, wave, t_rec, center, int_time
    
    def load_single_frame(self, hdul, field=None):
        try:
            if field is not None:
                self.in_name = field
            frame = None
            if self.in_name is not None:
                if self.in_name in hdul:
                    hdu = hdul[self.in_name]
                elif self.in_name =='original':
                    in_name = "COMPRESSED_IMAGE"
                    if in_name in hdul:
                        hdu = hdul[in_name]
                else:
                    hdu = hdul[-1]
                    self.params.png_frame_name = hdu.name
                frame = deepcopy(hdu.data)
            return frame
        except OSError as e:
            print(e)
            return frame
        except (UnboundLocalError, TypeError) as e:
            print("load single frame:: ", e)
            return None
    
    def load_best_fits_field(self, fits_path, in_name=None):
        """Load a fits file from disk"""
        with fits.open(fits_path, cache=False, ignore_missing_end=True) as hdul:
            hdul.verify('silentfix+ignore')  # Verify
            self.list_hdus(hdul)
            if in_name is None:
                self.in_name = self.set_hdul_in_name(hdul=hdul)
            else:
                self.in_name = in_name
            wave, t_rec, center, int_time, self.found_limb_radius = self.get_fits_info(hdul)
            frame, header = self.open_fits_hdul(hdul)
            img_type = header['IMG_TYPE']
        return frame, wave, t_rec, center, int_time, img_type
    
    def set_hdul_in_name(self, fits_path=None, hdul=None, field=None):
        """Determine the right in_name given any kind of input"""
        if field is not None:
            self.in_name = field
        if fits_path:
            hdul = fits.open(fits_path, cache=False, ignore_missing_end=True)
            self.in_name = self.find_correct_in_name(hdul)
            hdul.verify("silentfix+ignore")
            hdul.close()
        elif hdul:
            self.in_name = self.find_correct_in_name(hdul)
        return self.in_name
    
    def remove_blank_frames(self, hdul):
        # vprint("Blank Frame Ran")
        to_replace_list = ['COMPRESSED_IMAGE', '']
        for to_replace in to_replace_list:
            if to_replace in hdul:
                names = ['original']
                for ii, item in enumerate(hdul):
                    if item.name == to_replace:
                        if len(names):
                            item.name = names.pop(0)
                        else:
                            item.name = 'unknown_{}'.format(ii)
    
    # Curves Save and Load
    def prep_save_outs(self):
        """Prepare the scalar_out_curve for writing"""
        if self.outer_min is None:
            return None
        self.scalar_out_curve = np.zeros(len(self.outer_min))
        if self.found_limb_radius:
            self.scalar_out_curve[0] = self.fit_limb_radius
        if self.absolute_min:
            self.scalar_out_curve[1] = self.absolute_min
            self.scalar_out_curve[2] = self.absolute_max
        if self.savgol_filtered_inner_maximum is None:
            self.savgol_filtered_outer_maximum = np.empty_like(self.outer_min)
            self.savgol_filtered_inner_minimum = np.empty_like(self.outer_min)
            self.savgol_filtered_inner_maximum = np.empty_like(self.outer_min)
            self.savgol_filtered_outer_minimum = np.empty_like(self.outer_min)
        
        out_list = [self.outer_min, self.inner_min, self.inner_max, self.outer_max, self.scalar_out_curve]
        out_list.extend([self.output_abscissa, self.savgol_filtered_outer_maximum, self.savgol_filtered_inner_maximum,
                         self.savgol_filtered_inner_minimum, self.savgol_filtered_outer_minimum,
                         self.abs_max, self.abs_min,
                         ])
        # out_list.append([self.savgol_filtered_absol_maximum, self.savgol_filtered_absol_minimum])
        self.curve_descriptions = ["outer_min", "inner_min", "inner_max", "outer_max",
                     ["scalar_out_curve", "fit_limb_radius", "abs_min", "abs_max"], "output_abscissa",
                     "savgol_filtered_outer_maximum", "savgol_filtered_inner_maximum",
                     "savgol_filtered_inner_minimum", "savgol_filtered_outer_minimum", 'smooth_abs_max','smooth_abs_min']

        
        none_check = [item is not None for item in out_list]
        self.do_save = np.all(none_check)
        self.curve_out_array = np.asarray(out_list)
        return self.do_save
    
    def unpack_save_ins(self):
        """Prepare the scalar_out_curve for writing"""
        self.outer_min, self.inner_min, self.inner_max, \
        self.outer_max, self.scalar_in_curve, self.output_abscissa, \
        self.savgol_filtered_outer_maximum, self.savgol_filtered_inner_maximum, \
        self.savgol_filtered_inner_minimum, self.savgol_filtered_outer_minimum, \
        self.abs_max, self.abs_min, = np.loadtxt(self.params.curve_path())
        
        self.fit_limb_radius = self.scalar_in_curve[0]
        self.absolute_min = self.scalar_in_curve[1]
        self.absolute_max = self.scalar_in_curve[2]
    
    def save_curves(self, banner=True, extra_line=False):  #
        """Save the curves so they don't have to be recalculated"""
        self.super_flush()
        if banner:
            if extra_line:
                vprint("\r *\n *    Saving Radial Curves...", end='')
            else:
                vprint("\r *        Saving Radial Curves...", end='')
            
        if self.prep_save_outs():
            curve_path = self.params.curve_path()
            descr_path = curve_path.replace("curve.txt", "curve_names.txt")
            makedirs(os.path.dirname(curve_path), exist_ok=True)
            
            with open(descr_path, mode='w') as fp:
                for desc, item in zip(self.curve_descriptions, self.curve_out_array):
                    len_item = str(len(item))
                    # len_desc = str(len(desc))
                    fp.write(str(desc) + " : len=" + len_item)
            np.savetxt(curve_path, self.curve_out_array)
            if banner: vprint("Success!")
        else:
            vprint("Skipping Save Curves!")
    
    # def save_smoothed_curves(self):
    #     print(" *\n *    Saving Smoothed Curves...", end='')
    #         if self.prep_smooth_save_outs(): #self.do_save:
    #             np.savetxt(self.params.curve_path(), self.curve_out_array)
    #             print("Success!")
    #         else:
    #             print("Skipping Save Curves!")
    
    def load_curves(self, force=None, verb=False):
        """Load the curves so they don't have to be recalculated"""
        lc = self.load_print_latch
        if os.path.exists(self.params.curve_path()):
            if self.absolute_min is None or force:
                if lc: vprint("\r *    Loading Radial Curves...", end='')
                try:
                    self.unpack_save_ins()
                    # if verb: self.super_flush("Success!\n")
                    if lc: vprint("Success!", flush=True)
                    if False: print('', flush=True)
                    self.load_print_latch = False
                except ValueError as e:
                    print("Failed to load Radial Curves: {}".format(e))
                    raise e
        else:
            print("No Curves to Load")
                
                
                # self.image_learn()
                # self.save_curves()
            
            # if hdul['primary'].data is None:
            #     hdul['primary'].data = hdul['original'].data + 0
            #     hdul[0].name = 'primary'
            #     hdul['primary'].header = hdul['primary'].header + hdul['original'].header
            #     del hdul['original']
            #
            # # hdul.writeto(self.fits_path, output_verify="ignore", overwrite=True)
            #
            # for hdu in hdul:
            #
            #     try:
            #         del hdu.header['OSCNMEAN']
            #     except KeyError as e:
            #         pass
            #     try:
            #         del hdu.header['OSCNRMS' ]
            #     except KeyError as e:
            #         pass
            #
            
            # del hdul['primary'].header['OSCNMEAN']
            # del hdul['primary'].header['OSCNRMS']
            #
            # hdul[1].header['OSCNMEAN'] = 0.
            # hdul[1].header['OSCNRMS' ] = 0.
            #
            # hdul[0].verify('silentfix')
            # hdul[1].verify('silentfix')
            # a=1
            # hdul.verify('silentfix')
            #
            # print("mean  ", hdul[0].header['OSCNMEAN'])
            # print("rms   ", hdul[0].header['OSCNRMS' ])
            # print()
            # all_head_list0 = [('0 ', x, hdul[0].header[x]) for x in hdul[0].header]
            # all_head_list1 = [('1 ', x, hdul[1].header[x]) for x in hdul[1].header]
            # [print(x) for x in sorted(all_head_list0) if "OSCN" in x[1]]
            # [print(x) for x in sorted(all_head_list1) if "OSCN" in x[1]]
            #
            # # [print(x) for x in sorted(all_head_list) if "OSCN" in x]
            # a=1
            
            # print()
            # [print(x.name, '\t\t', x) for x in hdul]
            # a=1
            # print(self.list_hdus(hdul))
            # print()
            
            # data_frame = hdul[to_delete.pop(0)]
            # data_frame.name = 'PRIMARY'
            
            # for de in to_delete:
            #     del hdul[de]
            # hdul.update(self.fits_path)
            # [print(x.name, '\t\t', x) for x in hdul]
    
    def smallify_frame(self, frame):
        return frame
        mx = np.nanmax(frame)
        mn = np.nanmin(frame)
        normed = (frame - mn) / (mx - mn)
        
        scaled = normed * 2 ** 16
        average = np.uint16(np.round(np.nanmean(scaled)))
        de_NANed = np.nan_to_num(scaled, nan=average)
        compressed = de_NANed.astype(np.uint16)
        
        return compressed
    
    def get_fits_info(self, hdul):
        # Load the original out_array
        wave, t_rec, center, int_time, found_limb_radius = None, None, None, None, None
        ii = 0
        for ii in range(len(hdul)):
            try:
                first_data_hdul = hdul[ii]
                self.header = first_data_hdul.header
                wave = first_data_hdul.header['WAVELNTH']
                t_rec = first_data_hdul.header['T_OBS']
                center = [first_data_hdul.header['X0_MP'], first_data_hdul.header['Y0_MP']]
                int_time = first_data_hdul.header['EXPTIME']
                found_limb_radius = first_data_hdul.header['R_SUN']
                while found_limb_radius > first_data_hdul.header['NAXIS1']:
                    found_limb_radius /= 4.0
                break
            except KeyError as e:
                continue
        self.first_hIndex = ii
        self.params.found_limb_radius = found_limb_radius
        return wave, t_rec, center, int_time, found_limb_radius
    
    def open_fits_hdul(self, hdul):
        """Load a fits file from disk"""
        
        # Load the called for out_array
        if 'str' in str(type(self.in_name)):
            field_hdu = hdul[self.in_name]
        elif self.in_name is None:
            return None
        else:
            field_hdu = hdul[self.hdu_name_list[self.in_name]]
        
        data = None
        try:
            data = field_hdu.data
            header = hdul[1].header
        except TypeError:
            vprint("Processor: 705 !Failed to Load Frame!")
        return data, header
    
    def determine_penultimate_frame_name(self, hdul=None):
        if not self.hdu_name_list:
            self.list_hdus(hdul)
        get = -2 if len(self.hdu_name_list) > 1 else -1
        penultimate_frame_name = self.hdu_name_list[get]
        if penultimate_frame_name == 'primary':
            penultimate_frame_name = self.hdu_name_list[get + 1]
        return penultimate_frame_name
    
    def determine_first_frame_name(self, hdul=None):
        if not self.hdu_name_list:
            self.list_hdus(hdul)
        return self.hdu_name_list[0]
    
    def determine_last_frame_name(self, hdul=None):
        if not self.hdu_name_list:
            self.list_hdus(hdul)
        return self.hdu_name_list[-1]
    
    def find_correct_in_name(self, hdul, in_name=None):
        """Determine which out_array of the input file to use on redo"""
        # if self.ensured:
        #     return self.in_name
        # self.ensured = True
    
    
        if in_name is not None:
            self.in_name = in_name
        if self.in_name is None:
            self.in_name = -1
    
    
        reprocess_mode = self.params.reprocess_mode(self.reprocess_mode())
    
        self.hdu_name_list = self.list_hdus(hdul)
    
        input_frame_name = self.determine_in_frame_name()
        first_frame_name = self.hdu_name_list[0]
        second_frame_name = self.hdu_name_list[1]
        last_frame_name = self.hdu_name_list[-1]
        output_frame_name = self.determine_out_frame_name()
        penultimate_frame_name = self.determine_penultimate_frame_name()
        try:
            previous_frame_name = self.hdu_name_list[self.hdu_name_list.index(output_frame_name)-1]
        except ValueError:
            previous_frame_name = penultimate_frame_name
    
    
        filter_already_applied = filter_applied_last = False
        if output_frame_name.casefold() in [x.casefold() for x in self.hdu_name_list if type(x) is str]:
            filter_already_applied = True
        if input_frame_name.casefold() == output_frame_name.casefold():
            filter_applied_last = True
            # If you're about to redo a filter
    
        if filter_already_applied or filter_applied_last:
            if reprocess_mode == 'skip' or reprocess_mode is False:
                # Skip it
                self.in_name = None
                # raise FileExistsError("Skipping File")
            elif reprocess_mode == 'redo' or reprocess_mode is True:
                # Go to the previous out_array and remake
                self.in_name = previous_frame_name
            elif reprocess_mode == 'reset':
                # Go to the first out_array and remake
                self.in_name = first_frame_name
            elif reprocess_mode == 'double':
                # Repeat the filter a second time
                self.in_name = output_frame_name
            elif reprocess_mode == 'add':
                # Repeat the filter a second time
                self.in_name = output_frame_name
                self.out_name = self.out_name + "_redo"
            else:
                raise NotImplementedError
        else:
            self.in_name = input_frame_name
            if self.in_name == 'primary':
                self.in_name = second_frame_name
        hdul.verify('silentfix+ignore')
        return self.in_name
    
    def determine_in_frame_name(self):
        # Determine the called-for input out_array NAME
        if self.in_name is None:
            return None
        # self.in_name = self.set_hdul_in_name(self.fits_path)
        if type(self.in_name) is str:
            if self.in_name in self.hdu_name_list:
                input_frame_name = self.in_name.casefold()
            else:
                input_frame_name = self.hdu_name_list[0]
                # print("HDU Not Found, using {}".format(input_frame_name))
                # raise FileNotFoundError
        else:
            input_frame_name = self.hdu_name_list[self.in_name].casefold()
        return input_frame_name
    
    def determine_out_frame_name(self):
        # Determine the called-for output out_array NAME
        if type(self.out_name) is str:
            output_frame_name = self.out_name.casefold()
        else:
            output_frame_name = self.hdu_name_list[self.out_name].casefold()
        return output_frame_name
    
    def determine_first_hIndex(self, hdul):
        """Find out which hInd has the data"""
        hInd = 0
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
        """Return just the modified_image frome"""
        return self.params.modified_image
    
    def get_orig(self):
        """Return just the original_image frome"""
        return self.params.original_image
    
    def super_flush(self, txt=None, end=None, many=5):
        """Flush the stdout many times"""
        if txt:
            print(txt, flush=True, end=end)
        for ii in range(many):
            sys.stdout.flush()
            sys.stderr.flush()
            
    
    def list_hdus(self, hdul):
        self.hdu_name_list = [frame.name.casefold() for frame in hdul]
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
    
    def view_original(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots(num='Input Image')
        ax.set_title("Preview of Start Frame: {}".format(self.params.hdu_name))
        minmin = np.min(self.params.original_image)
        img = np.sqrt(np.asarray(self.params.original_image - minmin, dtype=np.float32))
        ax.imshow(img, cmap=self.params.cmap)
    
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
        
        
    @staticmethod
    def write_video_in_directory(directory=None, file_name=None, fps=10,
                                 folder_name=None, desc=None, key_string='keyframe', fullpath=None, destroy=False, shortcut=False):
        """Make a video out of whatever directory it's pointed at"""
        video_avi = None
        file_name = file_name or 'default_videoname.avi'
        try:
            if fullpath is not None:
                folder = os.path.dirname(fullpath)
                good_paths = [join(folder, f) for f in listdir(folder) if ('png' in f and not os.path.isdir(join(folder, f)))]
                video_path = fullpath.replace(".png", ".avi")
            else:
                radial_directory = directory
                # makedirs(radial_directory, exist_ok=True)
                video_path = radial_directory + "\\" + file_name
                good_paths = [radial_directory + "\\" + f for f in listdir(radial_directory) if 'png' in f]
                
                if desc is None:
                    desc = " *    Writing Video {}".format(basename(directory))
                
            
            # Initialize the Machine
            if len(good_paths):
                first_path = good_paths[0]
                height, width, _ = cv2.imread(first_path).shape
                video_avi = cv2.VideoWriter(video_path, 0, fps, (width, height))
        
                # Write the Frames
                for img_path in tqdm(good_paths, desc=desc, unit="frames"):
                    video_avi.write(cv2.imread(img_path))
                    if destroy:
                        os.remove(img_path)
                    # for img_path in good_paths:
            else:
                print('VideoProcessor:: There are no images yet. Make them first.')
                
        finally:
            # Shut it all down
            cv2.destroyAllWindows()
            if video_avi is not None:
                video_avi.release()
            # if shortcut:
            #     import winshell
            #     self.params.basename()
        # print(" ^    Successfully {} from {} images! ({} skipped)".format(self.finished_verb, ii, self.skipped))
 
 
 
 
 
 
 
 
 
 
 
 
        
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
#     self.local_wave_directory = join(self.params.imgs_top_directory(), wave)
#     self.image_folder = join(self.local_wave_directory, 'png')
#     self.movie_folder = abspath(join(self.params.imgs_top_directory(), "movies\\"))
#     self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(wave, strftime('%m%d_%H%M'), '{}'))
#     # print(self.video_name_stem)
#     makedirs(self.movie_folder, exist_ok=True)

# images = [img for img in listdir(self.image_folder) if img.endswith(".png")] # and self.check_valid_png(img)]

import os
from copy import copy
from os import makedirs
from os.path import join, dirname, basename
import numpy as np
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_pointing_table, get_correction_table
from scipy.signal import savgol_filter
from scipy.stats import stats
from science.color_tables import aia_color_table
import astropy.units as u

import sunpy.map

import aiapy.data.sample as sample_data
from aiapy.calibrate import normalize_exposure, register, update_pointing

from processor.Processor import Processor
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.ioff()

do_dprint = False
verb = False


def dprint(txt, **kwargs):
    if do_dprint:
        print(txt, **kwargs)


def vprint(in_string, *args, **kwargs):
    if verb:
        print(in_string, *args, **kwargs)


class SunPyProcessor(Processor):
    """This class template holds the code for the Sunpy Processors"""
    name = filt_name = "Sunpy Processor"
    description = "Apply sunpy effets to images"
    progress_verb = 'Normalizing'
    finished_verb = "Normalized"
    out_name = "sunpy"
    
    # Flags
    show_plots = True
    do_png = False
    renew_mask = True
    can_initialize = True
    
    # Parse Inputs
    def __init__(self, params=None, quick=False, rp=None, in_name="LEV1"):
        """Initialize the main class"""
        super().__init__(params, quick, rp)
        self.in_name = in_name
    
    def setup(self):
        pass
    
    def do_fits_function(self, fits_path=None, in_name=None, image=True):
        """Calls the do_work function on a single fits path if indicated"""
        if self.load_fits_image(fits_path, in_name=in_name):
            if (not self.use_keyframes) or (self.fits_path in self.keyframes):
                if self.should_run():
                    return self.do_work()
        return None
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.should_run():
            print("Template Ran")
        self.out_name = "lev15"
        # self.params.raw_image
        return self.params.modified_image
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        pass
    
    def should_run(self):
        """Decide of the processor should run on this file"""
        return True


class AIA_PREP_Processor(SunPyProcessor):
    """This class holds the code for the AIA_PREP Processor"""
    name = filt_name = "AIA_PREP"
    description = "Apply AIA_PREP to images"
    progress_verb = 'Preping'
    finished_verb = "Prepped"
    
    # Flags
    show_plots = True
    do_png = False
    renew_mask = True
    can_initialize = True
    can_use_keyframes = False
    
    # Parse Inputs
    def __init__(self, params=None, quick=False, rp=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp)
        self.level_1_maps = None
        self.level_15_maps = None
        self.correction_table = None
        self.pointing_table = None
        self.pointing_end = None
        self.pointing_start = None
        self.in_name_possibles  = ["Quantile", "T_Integrated", "LEV1"]
        self.in_name = ["Quantile", "T_Integrated", "LEV1"]
        self.out_name_stem = "LEV1p5_{}"
        self.params.modified_image = None
        pass
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.should_run():
            self.select_maps()
            self.get_aia_prep_data()
            self.do_AIA_PREP()
            self.save_out()
        return self.params.modified_image
    
    def select_maps(self):
        self.out_name = self.out_name_stem.format(self.in_name[0])
        while self.out_name.casefold() in self.hdu_name_list:
            self.in_name = self.in_name_possibles.pop(0)
            self.out_name = self.out_name_stem.format(self.in_name[0])
            
            # frame, wave, t_rec, center, int_time, img_type = self.load_best_fits_field(self.fits_path, in_name)
            
        self.load_fits_image(self.fits_path, in_name=self.in_name)
        self.params.header["LVL_NUM"] = 1.5
        self.level_1_maps = [sunpy.map.Map((self.params.raw_image, self.params.header))]

    def get_aia_prep_data(self):
        if self.correction_table is None:
            # We get the pointing table outside of the loop for the relevant time range.
            # Otherwise you're making a call to the JSOC every single time.
            self.pointing_start = self.level_1_maps[0].date - 3 * u.h
            self.pointing_end = self.level_1_maps[-1].date + 3 * u.h
            self.pointing_table = get_pointing_table(self.pointing_start, self.pointing_end)
            # The same applies for the correction table.
            self.correction_table = get_correction_table()

    # def do_AIA_PREP(self):
    #     self.level_15_maps = []
    #     for a_map in self.level_1_maps:
    #         map_updated_pointing = update_pointing(a_map, pointing_table=self.pointing_table)
    #         map_registered = register(map_updated_pointing)
    #         map_degradation = correct_degradation(map_registered, correction_table=self.correction_table)
    #         map_normalized = normalize_exposure(map_degradation)
    #         self.level_15_maps.append(map_normalized)
    
    def do_AIA_PREP(self):
        self.level_15_maps = []
        for a_map in self.level_1_maps:
            map_updated_pointing = update_pointing(a_map, pointing_table=self.pointing_table)
            map_registered = register(map_updated_pointing)
            map_degradation = correct_degradation(map_registered, correction_table=self.correction_table)
            map_normalized = normalize_exposure(map_degradation)
            map_double_normed = map_normalized / np.nanmax(map_normalized.data)
            out = map_double_normed if 'q' in self.out_name.casefold() else map_degradation
            self.level_15_maps.append(out)
            
    def save_out(self):
        # Plot
        # self.plot_lev1p5(plot_result=True)
        
        # Get the Data
        done_map = self.level_15_maps[0]
        self.params.modified_image = done_map.data
        
        # Get the Header
        self.header = self.params.header = sunpy.io.fits.header_to_fits(done_map.meta)

        
    
    def plot_lev1p5(self, plot_result=True):
        two_maps = [self.level_15_maps[0]]
        if plot_result:
            lev1 = np.sqrt(self.level_1_maps[0].data)
            lev1_map = sunpy.map.Map((lev1, self.params.header))
            two_maps.append(lev1_map)
            
            sequence = sunpy.map.Map(two_maps, sequence=True)
            sequence.peek(resample=0.25, annotate=True)
            plt.show(block=True)
    
    #     if not self.header:
    #         print("No header Loaded")
    #         return False
    #     self.can_use_keyframes = True
    #     not_dark = self.header["IMG_TYPE"] == "LIGHT"
    #     not_weak = self.header["EXPTIME"] > 1.0
    #     set_to_make = self.params.remake_norm_curves() or self.reprocess_mode()
    #     not_made_yet = not os.path.exists(self.params.curve_path()) or self.outer_min is None
    #     frame_is_not_loaded = self.params.raw_image is None
    #     self.go_ahead = not_weak & not_dark and (set_to_make or not_made_yet or frame_is_not_loaded)
    #     return self.go_ahead
    # #
    
    ###################
    ##   Main Calls  ##
    ###################
    
    #######################################

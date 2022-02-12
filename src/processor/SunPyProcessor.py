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
        self.in_name = ["T_Integrated", "LEV1"]
        self.out_name = "LEV1p5_{}"
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.should_run():
            self.select_maps()
            self.get_aia_prep_data()
            self.do_AIA_PREP()
        return self.params.modified_image
    
    def select_maps(self):
        self.params.header["LVL_NUM"] = 1.5
        self.level_1_maps = [sunpy.map.Map((self.params.raw_image, self.params.header))]
        self.level_1_maps.append(sunpy.map.Map((self.params.raw_image, self.params.header)))

    def get_aia_prep_data(self):
        if self.correction_table is None:
            # We get the pointing table outside of the loop for the relevant time range.
            # Otherwise you're making a call to the JSOC every single time.
            self.pointing_start = self.level_1_maps[0].date - 3 * u.h
            self.pointing_end = self.level_1_maps[-1].date + 3 * u.h
            self.pointing_table = get_pointing_table(self.pointing_start, self.pointing_end)
            # The same applies for the correction table.
            self.correction_table = get_correction_table()

    def do_AIA_PREP(self):
        self.level_15_maps = []
        for a_map in self.level_1_maps:
            map_updated_pointing = update_pointing(a_map, pointing_table=self.pointing_table)
            map_registered = register(map_updated_pointing)
            map_degradation = correct_degradation(map_registered, correction_table=self.correction_table)
            map_normalized = normalize_exposure(map_degradation)
            self.level_15_maps.append(map_normalized)
        
        self.plot_lev1p5(plot_result=False)
        
        self.params.modified_image = self.level_15_maps[-1].data
        self.out_name = self.out_name.format(self.in_name[0])
    
    def plot_lev1p5(self, plot_result=True):
        if plot_result:
            self.level_15_maps.append(self.level_1_maps)
            sequence = sunpy.map.Map(self.level_15_maps, sequence=True)
            sequence.peek()
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

import os
from copy import copy
import time
from os import makedirs
from os.path import join, dirname, basename
import matplotlib.pyplot as plt

import astropy.units as u
import sunpy.data.sample
import sunpy.map

import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins
import aiapy
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
    raw_map = None
    
    # Parse Inputs
    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp, in_name)
        self.tm = 0
        self.radial_bin_edges = equally_spaced_bins(inner_value=0.0, nbins=300) * u.R_sun
        self.in_name = in_name or self.params.master_frame_list


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
        self.in_name = self.params.master_frame_list
        
        self.last_wave = None
        self.psf = None
        self.level_1_maps = None
        self.level_15_maps = None
        self.correction_table = None
        self.pointing_table = None
        self.pointing_end = None
        self.pointing_start = None
        self.out_name = "lev1p5"
        self.params.modified_image = None
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.should_run():
            self.get_aia_prep_data()
            self.do_AIA_PREP()
            # if self.params.destroy:
            # self.remove_unprocessed_frames2()
            return self.params.modified_image
        return None
    
    def do_AIA_PREP(self):
        self.level_15_maps = []
        for a_map in self.level_1_maps:
            
            if False:
                a_map = self.deconvolve_psf(a_map)
            
            map_updated_pointing = self.get_updated_pointing(a_map)
            
            # Execute AIA_PREP
            map_registered = register(map_updated_pointing)
            map_degradation = correct_degradation(map_registered, correction_table=self.correction_table)
            map_normalized = normalize_exposure(map_degradation)
            map_double_normed = map_normalized / np.nanmax(map_normalized.data)
            out = map_double_normed if 'q' in self.out_name.casefold() else map_degradation
            self.level_15_maps.append(out)
            
        done_map = self.level_15_maps[0]
        self.params.modified_image = done_map.data
        self.header = self.params.header = sunpy.io.fits.header_to_fits(done_map.meta)

    def get_aia_prep_data(self, force=False):
        self.params.header["LVL_NUM"] = 1.5
        self.level_1_maps = [sunpy.map.Map((self.params.raw_image, self.params.header))]
        if self.correction_table is None or force:
            # We get the pointing table outside of the loop for the relevant time range.
            # Otherwise you're making a call to the JSOC every single time.
            self.pointing_start = self.level_1_maps[0].date - 3 * u.h
            self.pointing_end = self.level_1_maps[-1].date + 3 * u.h
            self.pointing_table = get_pointing_table(self.pointing_start, self.pointing_end)
            # The same applies for the correction table.
            self.correction_table = get_correction_table()
    
    def deconvolve_psf(self, a_map):
        import aiapy.psf as psf
        if not a_map.wavelength == self.last_wave or self.psf is None:
            # Make the psf map if needed
            self.psf = psf.psf(a_map.wavelength)
        # Deconvolve the PSF
        m_deconvolved = aiapy.psf.deconvolve(a_map, psf=self.psf)
        return m_deconvolved
    
    def get_updated_pointing(self, a_map, one_deep=True):
        # Get the new pointing information
        try:
            map_updated_pointing = update_pointing(a_map, pointing_table=self.pointing_table)
            return map_updated_pointing
        except IndexError as e:
            # If it fails
            if one_deep:
                # For the first time, re-prep the data
                self.get_aia_prep_data(force=True)
                # Return a recursion of this funtion
                return self.get_updated_pointing(a_map, one_deep=False)
            else:
                raise e
    
    def plot_lev1p5(self, plot_result=True):
        two_maps = [self.level_15_maps[0]]
        if plot_result:
            lev1 = np.sqrt(self.level_1_maps[0].data)
            lev1_map = sunpy.map.Map((lev1, self.params.header))
            two_maps.append(lev1_map)
            
            sequence = sunpy.map.Map(two_maps, sequence=True)
            sequence.peek(resample=0.25, annotate=True)
            plt.show(block=True)


class NRGFProcessor(SunPyProcessor):
    """This class template holds the code for the Sunpy Processors"""
    name = filt_name = "NRGF Processor"
    description = "Apply NRGF effets to images"
    progress_verb = 'Normalizing'
    finished_verb = "Normalized"
    out_name = "NRGF"
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        self.params.modified_image = radial.nrgf(self.raw_map,
                                                 self.radial_bin_edges, application_radius=0.00 * u.R_sun).data
        return self.params.modified_image


class FNRGFProcessor(SunPyProcessor):
    """This class template holds the code for the Sunpy Processors"""
    name = filt_name = "FNRGF Processor"
    description = "Apply FNRGF effets to images"
    progress_verb = 'Normalizing'
    finished_verb = "Normalized"
    out_name = "FNRGF"
    
    # Parse Inputs
    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp, in_name)
        self.order = 20
        self.attenuation_coefficients = radial.set_attenuation_coefficients(self.order)
    
    def do_work(self):
        self.params.modified_image = radial.fnrgf(self.raw_map, self.radial_bin_edges,
                                                  self.order, self.attenuation_coefficients).data
        return self.params.modified_image


class IntEnhanceProcessor(SunPyProcessor):
    """Implementation of: https://docs.sunpy.org/projects/sunkit-image/en/stable/api/sunkit_image.radial.intensity_enhance.html#sunkit_image.radial.intensity_enhance
        Which is clled Intensity_enhance, but seems like it's a version of AIR_RFILT that divides curve that is fitted to the data
        Technically this is similar to SRN, I will be interested to see how it performs
    """
    name = filt_name = "Intensity_Enhance Sunpy Processor"
    description = "Apply Intensity_Enhance to the images"
    progress_verb = 'Filtering'
    finished_verb = "Normalized"
    out_name = "int_enhance"
    
    def do_work(self):
        self.params.modified_image = radial.intensity_enhance(self.raw_map, self.radial_bin_edges).data
        return self.params.modified_image


class MSGNProcessor(SunPyProcessor):
    """This class template holds the code for the Sunpy Processors"""
    name = filt_name = "MSGN Processor"
    description = "Apply MSGN effets to images"
    progress_verb = 'Normalizing'
    finished_verb = "Normalized"
    out_name = "MSGN"
    
    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp, in_name)
        self.in_name = self.params.aftereffects_in_name or in_name or self.in_name
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        import sunkit_image.enhance as enhance
        self.params.modified_image = enhance.mgn(self.params.raw_image)
        return self.params.modified_image

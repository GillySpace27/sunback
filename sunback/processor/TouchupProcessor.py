import os
from copy import copy
import time
from os import makedirs
from os.path import join, dirname, basename

import h5py
import matplotlib.pyplot as plt

import astropy.units as u
import sunpy.data.sample
# import sunpy.map

import sunkit_image.radial as radial
from astropy.io import fits
from scipy import ndimage
from sunkit_image.utils import equally_spaced_bins
import aiapy
import numpy as np
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_pointing_table, get_correction_table
from scipy.signal import savgol_filter
from scipy.stats import stats
from tqdm import tqdm

from sunback.science.color_tables import aia_color_table
import astropy.units as u

# import sunpy.map

import aiapy.data.sample as sample_data
from aiapy.calibrate import normalize_exposure, register, update_pointing

from sunback.processor.Processor import Processor
import warnings

from sunback.utils.RHT.rht import rht
from sunback.utils.RHT.rht.convRHT import unsharp_mask

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

## I think the touchup might be being called multiple times



class TouchupProcessor(Processor):
    """This class template holds the code for the Sunpy Processors"""
    name = filt_name = "RHT Processor"
    description = "Apply the Rolling Hour Transform to images"
    progress_verb = 'Normalizing'
    finished_verb = "Normalized"
    out_name = "RHT"

    # Flags
    show_plots = True
    renew_mask = True
    can_initialize = True
    raw_map = None
    # do_png = False

    # Parse Inputs
    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp, in_name)
        # self.tm = time.time()
        self.in_name = self.params.aftereffects_in_name or "lev1p5"

    def do_work(self):
        print("")


        self.save_frame(self.frame, self.fits_path)

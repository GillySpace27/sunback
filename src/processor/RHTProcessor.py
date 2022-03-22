import os
from copy import copy
import time
from os import makedirs
from os.path import join, dirname, basename

import h5py
import matplotlib.pyplot as plt

import astropy.units as u
import sunpy.data.sample
import sunpy.map

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

from science.color_tables import aia_color_table
import astropy.units as u

import sunpy.map

import aiapy.data.sample as sample_data
from aiapy.calibrate import normalize_exposure, register, update_pointing

from processor.Processor import Processor
import warnings

from utils.RHT.rht import rht
from utils.RHT.rht.convRHT import unsharp_mask

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


class RHTProcessor(Processor):
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
        self.tm = 0
        self.radial_bin_edges = equally_spaced_bins(inner_value=0.0, nbins=300) * u.R_sun
        self.RHT_out_list = []
        self.rht_cube = None
        self.theta = None
        self.in_name = self.params.aftereffects_in_name or "lev1p5"
        
    def do_work(self):
        print("")
        
        self.outroot = self.make_temp_dir(self.fits_path)
        
        self.get_cube()
        
        self.params.modified_image = np.sum(self.rht_cube, axis=0)
        
        plot_angles = False
        if plot_angles:
            self.plot_angles()
        
        return np.transpose(self.params.modified_image)


    def get_cube(self):
        files_all = os.listdir(self.outroot)
        h5_files = [x for x in files_all if".h5" in x]
        have_file = len(h5_files) > 0
        
        # Don't Redo
        if have_file and not self.params.reprocess_mode():
            file_name = h5_files[0]
            file_path = os.path.join(self.params.temp_directory(), file_name)
            print(" *   Loading h5 file...", end="")
            with h5py.File(file_path, 'r') as f:
                self.rht_cube = np.asarray(f['rht_cube']).T
                n_theta = self.rht_cube.shape[0]
                self.theta = np.linspace(0, 2*np.pi, n_theta)
            print("done!")
        # Do Redo
        else:
            (self.rht_cube, self.theta) = rht.main(source=self.fits_path,
                                                   data=self.params.raw_image, conv=True, outroot=self.outroot)

    def plot_angles(self):
        print("Plotting angle images...", end="")
        # The first dimension of this cube is theta
        self.angle_dir = os.path.join(self.outroot, "angles")
        makedirs(self.angle_dir,exist_ok=True)
        
        self.plot_one_angle(np.nan, self.params.modified_image)
    
        for ii, img in tqdm(enumerate(self.rht_cube), desc=" * Plotting "):
            self.plot_one_angle(self.theta[ii] , img)
            
        print("done!")
        
    def plot_one_angle(self, theta_rad, img):
        theta = theta_rad / np.pi * 180
        theta_clean = "{:0.1f}".format(theta)
        angle_path = os.path.join(self.angle_dir, "{}.png".format(theta_clean))
        
        # img[img==0.] = np.nan
        
        [big, small] = np.nanpercentile(img, [99.5, 0.05])
        img = (img - small) / (big-small)
        plt.ioff()
        fig, ax = plt.subplots()
        
        ax.imshow(img, origin="lower", interpolation="None")
        
        x1, y1 = 3800, 3800
        r = 150
        dx = r * np.sin(theta_rad)
        dy = r * np.cos(theta_rad)
        
        plt.arrow(x1, y1, dx, dy, width=20, head_width=75)
        
        
        ax.set_title("Angle: {} degrees".format(theta_clean))
        fig.savefig(angle_path, dpi=500)
        # plt.show(block=True)
        plt.close(fig)
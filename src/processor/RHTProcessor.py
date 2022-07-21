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
    progress_verb = 'Analyzing'
    finished_verb = "Examined"
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
        self.in_name = in_name or "lev1p5"
        
        if len(self.params.aftereffects_in_name) > 0:
            self.in_name = self.params.aftereffects_in_name.pop(0)
        
        print("Evaluating frame: {}".format(self.in_name))
        self.tic()
    
    def do_work(self):
        print("")
        
        self.outroot = self.make_temp_dir(self.fits_path)
        
        self.get_cube()
        
        self.params.modified_image = np.sum(self.rht_cube, axis=0)
        
        plot_angles = False
        if plot_angles:
            self.plot_angles()
        
        return np.transpose(self.params.modified_image)
    
    def load_last_run(self, h5_files):
        file_name = h5_files[0]
        file_path = os.path.join(self.params.temp_directory(), file_name)
        print(" *   Loading h5 file...", end="")
        with h5py.File(file_path, 'r') as f:
            self.rht_cube = np.asarray(f['rht_cube']).T
            n_theta = self.rht_cube.shape[0]
            self.theta = np.linspace(0, 2 * np.pi, n_theta)
        print("done!")
    
    def get_cube(self):
        files_all = os.listdir(self.outroot)
        h5_files = [x for x in files_all if ".h5" in x]
        have_file = len(h5_files) > 0
        
        # Don't Redo
        if have_file and False:  # not self.params.reprocess_mode():
            self.load_last_run(h5_files)
        # Do Redo
        else:
            self.make_cube()
    
    def make_cube(self):
        print("Making Cube...")
        
        use_image = self.params.raw_image + 0

        use_image = self.mask_out_sun(use_image)

        filtered_image = self.spatially_filter_image(use_image)
        
        
        1/0
        
        
        # binary_image = self.make_RHT_binary_map(filtered_image)
        
        # self.run_RHT_algorithm(binary_image)


    
    
    def make_RHT_binary_map(self, use_image):
        print("Binary Filtering...")
        
        brightness_map_04 = use_image >=0.4
        brightness_map_05 = use_image >=0.5
        brightness_map_06 = use_image >=0.6

        
        if True:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(16,12), sharex="all", sharey="all")
            ax0.imshow(use_image, cmap='gray')
            ax0.set_title('Original')
            ax0.set_axis_off()
            
            ax1.imshow((brightness_map_04), cmap='gray')
            ax1.set_title('Binary Map 0.4')
            ax1.set_axis_off()
            
            ax2.imshow((brightness_map_05), cmap='gray') # hsv is cyclic, like angles
            ax2.set_title('Binary Map 0.5')
            ax2.set_axis_off()
     
            ax3.imshow((brightness_map_06), cmap='gray') # hsv is cyclic, like angles
            ax3.set_title('Binary Map 0.6')
            ax3.set_axis_off()
            
            # fig.suptitle("{}, x1= {}, x2 = {}".format(self.in_name, x1, x2))
            plt.xlim((1500, 2500))
            plt.ylim((3600,4096))
            plt.tight_layout()
            plt.show(block=True)
        
        pass
        
        
    def spatially_filter_image(self, use_image):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
        from scipy import signal
        import matplotlib.pyplot as plt
        print("Spatial Filtering...")
        # scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],          #Compute the gradient of an image by 2D convolution with a complex Scharr operator. (Horizontal operator is real, vertical is imaginary.) Use symmetric boundary condition to avoid creating edges at the image boundaries.
        #                    [-10+0j, 0+ 0j, +10 +0j],
        #                    [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
        
        lp_wind = 9  # Options are 3, 5, 7, 9, 11 and different for each image
        hp_wind = 9
        
        low_pass_window  = np.ones((lp_wind, lp_wind))/(lp_wind**2)
        high_pass_window = -np.ones((hp_wind, hp_wind))/(hp_wind**2)
        high_pass_window[hp_wind//2, hp_wind//2] = 1 - 1/(hp_wind**2)
        
        kernal_1 = low_pass_window
        kernal_2 = high_pass_window
        
        lowpass_img = signal.convolve2d(use_image, kernal_1, boundary='symm', mode='same')
        highpass_img = signal.convolve2d(use_image, kernal_2, boundary='symm', mode='same')
        grad = lowpass_img - highpass_img
        
        unsharp = use_image - lowpass_img
        
        print("Histogramming...")
        import cv2
        from PIL import Image
        
        to_use_image = unsharp #grad # use_image
        
        thresh = np.nanpercentile(to_use_image, [1, 99])
        
        normed = self.norm_formula(to_use_image, *thresh)
        normed[normed>1] = 1.
        normed[normed<0] = 0.
        
        smooshed = np.round(normed * 255).astype(np.uint8)
        
        # smooshed = smooshed.astype(np.uint8)
        
        # plt.scatter(smooshed[::10000], 20, (0,1))
        # plt.show(block=True)
        #
        #
        print("Canning...")
        output_image = cv2.Canny(smooshed, 60, 150)
        
        print("Plotting...")
        
        # input_image = Image.fromarray(use_image)
        # image = cv2.cvtColor(use_image, cv2.COLOR_BGR2GRAY )
        a=1
        # output_image= Image.fromarray(use_image)
        # img = cv2.imdecode(use_image, flags=cv2.IMREAD_GRAYSCALE)
        # edged_image = cv2.imdecode(np.zeros_like(use_image), flags=cv2.IMREAD_GRAYSCALE)
        #
        
        
        #Plot
        if True:
            fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2,3, figsize=(16,12), sharex="all", sharey="all")
            ax0.imshow(np.absolute(use_image), cmap='gray')
            ax0.set_title('Original')
            ax0.set_axis_off()
            
            ax1.imshow(np.absolute(lowpass_img), cmap='gray')
            ax1.set_title('Lowpass')
            ax1.set_axis_off()
            # import matplotlib.markers as markers
            # marker = markers.MarkerStyle(marker='s', fillstyle='none')
            
            
            from matplotlib.patches import Rectangle, Circle
            ax1.add_patch(Rectangle((2200,3900),lp_wind,lp_wind, zorder=1000, fill=False, edgecolor='b'))
            ax2.add_patch(Rectangle((2200,3900),hp_wind,hp_wind, zorder=1000, fill=False, edgecolor='r'))
            
            ax3.add_patch(Rectangle((2200,3900),lp_wind,lp_wind, zorder=1000, fill=False, edgecolor='b'))
            ax3.add_patch(Rectangle((2200,3900),hp_wind,hp_wind, zorder=1000, fill=False, edgecolor='r'))
            
            ax2.imshow(np.log10(np.absolute(highpass_img)), cmap='gray') # hsv is cyclic, like angles
            ax2.set_title('Highpass')
            ax2.set_axis_off()
     
            ax3.imshow(np.absolute(grad), cmap='gray') # hsv is cyclic, like angles
            ax3.set_title('Low - High')
            ax3.set_axis_off()

            ax4.imshow(np.absolute(smooshed), cmap='gray')
            ax4.set_title('uint8')
            ax4.set_axis_off()
            
            ax5.imshow(np.absolute(output_image), cmap='gray')
            ax5.set_title('Canny Edged')
            ax5.set_axis_off()
            
            x1 = 2080
            x2 = 2255
            y1 = 3850
            y2 = 4070
            
            
            fig.suptitle("{}, x1= {}, x2 = {}".format(self.in_name, lp_wind, hp_wind))
            plt.xlim((x1, x2))
            plt.ylim((y1, y2))
            self.maximizePlot()
            plt.subplots_adjust(
                top=0.934,
                bottom=0.015,
                left=0.008,
                right=0.992,
                hspace=0.081,
                wspace=0.0)
            plt.show(block=True)
        
        
    
    def run_RHT_algorithm(self, binary_image):
        r_sun_pixels = self.params.header["R_SUN"]
        
        # Width of the RHT Window
        # Boe 2020 says 0.08 to 1.0 Rs. Example given of 31
        W_RHT = 0.08 * r_sun_pixels
        # W_RHT = 55.
        
        # Amount of radial smear in the algorithm
        # Boe 2020 says 0.02 to 0.4 Rs
        smear = 0.02 * r_sun_pixels
        # smear = 11.
        
        # Schad 2017 says that f=0.25 is a good choice
        frac = 0.70
        
        # import scipy.signal.windows as wind
        # window_length_pixels
        
        # wind.boxcar()
        
        ## Run RHT Algorithm
        (self.rht_cube, self.theta) = rht.main(source=self.fits_path,
                                               data=binary_image,
                                               conv=True,
                                               outroot=self.outroot,
                                               wlen=W_RHT,
                                               smr=smear,
                                               frac=frac,
                                               )
    
    def plot_angles(self):
        print("Plotting angle images...", end="")
        # The first dimension of this cube is theta
        self.angle_dir = os.path.join(self.outroot, "angles_{}".format(self.in_name))
        makedirs(self.angle_dir, exist_ok=True)
        
        self.plot_one_angle(np.nan, self.params.modified_image)
        
        for ii, img in tqdm(enumerate(self.rht_cube), desc=" * Plotting "):
            self.plot_one_angle(self.theta[ii], img)
        
        print("done!")
    
    def plot_one_angle(self, theta_rad, img):
        theta = theta_rad / np.pi * 180
        theta_clean = "{:0.1f}".format(theta)
        angle_path = os.path.join(self.angle_dir, "{}.png".format(theta_clean))
        
        # img[img==0.] = np.nan
        
        [big, small] = np.nanpercentile(img, [99.5, 0.05])
        img = (img - small) / (big - small)
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


import os
from copy import copy
import time
from os import makedirs
from os.path import join, dirname, basename
# from astropy.convolution import convolve, convolve_fft, Box2DKernel, CustomKernel, Gaussian2DKernel

import cv2
import h5py
import matplotlib.pyplot as plt

import astropy.units as u
import sunpy.data.sample
import sunpy.map

import sunkit_image.radial as radial
from astropy.io import fits
from scipy import ndimage, signal
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
        self.nan_map = None
        self.r_bar = None
        self.h_xy = None
        self.smol = False
        self.shrink_F = 1
        self.thresholded = None
        self.tm = 0
        self.radial_bin_edges = equally_spaced_bins(inner_value=0.0, nbins=300) * u.R_sun
        self.RHT_out_list = []
        self.rht_cube = None
        self.theta = None
        self.in_name = in_name or "lev1p5"
        self.params.modified_image = None
        
        if len(self.params.aftereffects_in_name) > 0:
            self.in_name = self.params.aftereffects_in_name.pop(0)
        
        print("Evaluating frame: {}".format(self.in_name))
        # self.tic()
    
    def do_work(self):
        print("")
        
        self.run_RHT()
        
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
    
    def is_there_saved_data(self):
        files_all = os.listdir(self.outroot)
        h5_files = [x for x in files_all if ".h5" in x]
        have_file = len(h5_files) > 0
        return have_file, h5_files
    
    def run_RHT(self):
        self.outroot = self.make_temp_dir(self.fits_path)
        
        have_file, h5_files = self.is_there_saved_data()
        # Don't Redo
        if have_file and False:  # not self.params.reprocess_mode():
            self.load_last_run(h5_files)
        # Do Redo
        else:
            self.make_RHT_cube()
    
    def get_angles_from_bitmap(self, image, outroot=None, thresh=0.95, w_factor=1.0):
        outroot = outroot or self.outroot
        cube, theta, r_bar = self.run_RHT_algorithm(image, outroot=outroot, w_factor=w_factor)
        if self.rht_cube is None:
            self.rht_cube = cube
        else:
            self.rht_cube += cube
        
        weighted_thetas = theta[:, None, None] * cube
        summed_weights = np.sum(cube, axis=0)
        summed_weights_theta = np.sum(weighted_thetas, axis=0)
        
        weighted_theta = summed_weights_theta / summed_weights
        weighted_theta *= 180 / np.pi
        weighted_theta = self.donut_the_sun(weighted_theta)
        
        too_low = r_bar < thresh
        r_bar[too_low] = np.nan
        weighted_theta[too_low] = np.nan
        
        return weighted_theta, r_bar
    
    def combine_RHT_runs(self, runs):
        theta_map = self.combine_runs(runs)
        
        # Find Nans and nan angles
        theta_map1, runs = self.do_nan_run(theta_map, runs, 0.8)
        theta_map2, runs = self.do_nan_run(theta_map, runs, 1.2)
        return theta_map, runs
    
    def combine_runs(self, runs, weighted=True):
        if weighted:
            theta_map, runs = self.combine_runs_weighted(runs)
        else:
            theta_map, runs = self.combine_runs_maxima(runs)
        return theta_map, runs

    def combine_runs_maxima(self, runs):
        # This Doesnt Work
        thetas, r_maps = zip(*runs)
        stack_theta = np.dstack(thetas)
        stack_r = np.dstack(r_maps)
        
        
        # nan_array = np.ones_like(stack_r[:,:,0]) * np.nan
        
        no_data = np.nansum(stack_r, axis=2)==0
        best_ind = np.argmax(stack_r, axis=2)
        theta_map = np.zeros_like(r_maps[0])
        
        shape = theta_map.shape
        for ii in np.arange(shape[0]):
            for jj in np.arange(shape[1]):
                best = best_ind[ii,jj]
                theta_map[ii, jj] = stack_theta[ii, jj, best]
        theta_map[no_data] = np.nan
        
        # for index, argmax in enumerate(best_ind):
        #     theta_map[index] = stack_r argmax
        
        
        
        theta_map = stack_theta[best_ind]
        # best_ind[no_data] = -1
        
        stack_mult = stack_theta * stack_r
        numerator = np.nansum(stack_mult, axis=2)
        denominator = np.nansum(stack_r, axis=2)
        theta_map = np.divide(numerator, denominator)
        return theta_map, runs
    
    def combine_runs_weighted(self, runs):
        thetas, r_maps = zip(*runs)
        stack_theta = np.dstack(thetas)
        stack_r = np.dstack(r_maps)
        
        stack_mult = stack_theta * stack_r
        numerator = np.nansum(stack_mult, axis=2)
        denominator = np.nansum(stack_r, axis=2)
        theta_map = np.divide(numerator, denominator)
        return theta_map, runs
    
    def do_nan_run(self, theta_map, runs, w_factor=1.0):
        nan_map = np.isnan(theta_map)
        self.nan_map = nan_map
        theta_nan, r_bar_nan = self.get_angles_from_bitmap(nan_map, w_factor=w_factor)
        r_bar_nan[self.radius <= 1] = 0
        runs.append((theta_nan, r_bar_nan))
        
        # Combine the Maps Again
        theta_map = self.combine_runs(runs)
        return theta_map, runs
        
        # TODO
        # theta_map = np.nanmean(np.dstack((weighted_theta_reg, weighted_theta_inv)), axis=2)
        # theta_map = np.nanmean(weighted_theta_reg[None, :, :], weighted_theta_inv[None, :, :])
        
        # Plot
        # plt.imshow(np.zeros_like(theta_map), cmap='gray')
        # plt.imshow(theta_map, cmap='hsv', interpolation='None')
        # plt.show(block=True)
        
        # box = np.dstack((r_bar_reg, r_bar_inv, r_bar_edg))
        # plt.imshow(box); plt.show(block=True)
        
        # inds = np.argmax(box, axis=2)
        
        # reg_inds = inds == 0
        # inv_inds = inds == 1
        # edg_inds = inds == 2
        
        #
        # # theta_map = np.nanmean(stack_theta, axis=2)
        #
        #
        # weighted_theta_all = theta_reg*r_bar_reg + theta_inv*r_bar_inv + theta_edg*r_bar_edg
        # weighting_all = r_bar_reg + r_bar_inv + r_bar_edg
        # theta = weighted_theta_all / weighting_all
        #
        # # theta_map = np.nan * np.ones_like(weighted_theta_reg)
        # theta_map[reg_inds] = theta_reg[reg_inds]
        # theta_map[inv_inds] = theta_inv[inv_inds]
        # theta_map[edg_inds] = theta_edg[edg_inds]
        
        # = np.nanmean(, axis=2)
    
    def change_to_angle_from_radial(self, theta_map):
        coord_theta = (self.theta_array * 180 / np.pi)
        shift_theta = 180 - np.mod(coord_theta, 180).T
        from_radial_theta = np.abs(theta_map - shift_theta)
        
        plt.imshow(from_radial_theta, cmap='hsv', origin="lower")
        plt.show(block=True)
    
    def make_sobel(self, thresholded_image):
        sobel_64 = cv2.Sobel(thresholded_image, cv2.CV_64F, 1, 0, ksize=1)
        abs_64 = np.absolute(sobel_64)
        sobel_8u = np.uint8(abs_64)
        return sobel_8u
    
    def make_RHT_cube(self):
        print(" * Making Cube...", flush=True)
        
        # Get Inputs in Line
        self.prep_inputs(shrink=True)
        
        # Make the thresholded images
        thresholded_image = self.segmentation_jing11()
        inv_thresholded_image = self.donut_the_sun(255 - thresholded_image)
        sobel_8u = self.make_sobel(thresholded_image)
        
        # Run RHT on all of them
        thresh = 0.95
        reg = self.get_angles_from_bitmap(thresholded_image, thresh=thresh)
        inv = self.get_angles_from_bitmap(inv_thresholded_image, outroot=self.outroot + "_inv", thresh=thresh)
        edg = self.get_angles_from_bitmap(sobel_8u, outroot=self.outroot + "_sobel", thresh=thresh)
        
        # Combine the RHT runs
        runs = [reg, inv, edg]
        theta_map, runs = self.combine_RHT_runs(runs)
        nan = runs[-1]
        
        # Add those together to get full coverage
        self.suptitle = "Regular + Inv"
        
        from_radial_theta = self.change_to_angle_from_radial(theta_map)
        
        # self.params.modified_image = theta_map
        self.params.modified_image = from_radial_theta
        
        r_vec_3d = np.dstack((reg[1], inv[1], edg[1]))
        # plt.imshow(r_vec_3d); plt.show(block=True)
        
        # plt.imshow(weighted_theta_reg, interpolation="none", cmap="brg")
        # plt.show(block=True)
        # output = np.zeros_like(thresholded_image)
        # doplot = True
        # for t_values, weights in zip(theta, total_cube):
        #     output += t_values * weights
        #     if doplot:
        #         plt.imshow(weights)
        #         plt.show(block=True)
        
        #########################
        
        # plt.imshow(inv_thresholded_image, interpolation="None"); plt.show(block=True)
        
        # # print("Edge")
        # t1, t2 = 35, 95
        # canny_image = cv2.Canny(thresholded_image, t1, t2)
        #
        # kernel = np.ones((2,2), 'uint8')
        #
        # kernal = 1/5 * np.array([[0, 1, 0],          #Compute the gradient of an image by 2D convolution with a complex Scharr operator. (Horizontal operator is real, vertical is imaginary.) Use symmetric boundary condition to avoid creating edges at the image boundaries.
        #                          [1, 1, 1],
        #                          [0, 1, 0]]) # Gx + j*Gy
        #
        # canny_image_dialated = cv2.dilate(canny_image, kernel, iterations=1)
        # mag, direct = self.compute_scharr_image_gradient
        
        # Output dtype = cv2.CV_8U
        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        
        asdf = 1
        
        if True:
            print(" ** Plot")
            # , (ax6, ax7, ax8)
            fig, ((ax0, ax1, ax2, ax3, ax4), (axw, axx, axy, axz, axa), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(3, 5, sharex=True,
                                                                                                                   sharey=True)  # , figsize=(6, 15))
            
            # Row 1 : Binary Images
            ax0.imshow(thresholded_image, cmap='gray', interpolation="None")
            ax0.set_title('thresholded_image')
            ax0.set_axis_off()
            
            ax1.imshow(inv_thresholded_image, cmap='gray', interpolation="None")
            ax1.set_title('inv_thresholded_image')
            ax1.set_axis_off()
            
            ax2.imshow(sobel_8u, cmap='gray', interpolation="None")
            ax2.set_title('sobel_8u')
            ax2.set_axis_off()
            
            ax3.imshow(self.nan_map, cmap='gray', interpolation="None")
            ax3.set_title('nan_map')
            ax3.set_axis_off()
            
            # ax4.imshow(self.nan_map, cmap='gray', interpolation="None")
            # ax4.set_title('nan_map')
            # ax4.set_axis_off()
            
            # Row 2: Weighted Theta Maps
            axw.imshow(reg[0], cmap='hsv', interpolation="None", vmin=0, vmax=180)  # hsv is cyclic, like angles
            axw.set_title('weighted_theta_reg')
            axw.set_axis_off()
            
            axx.imshow(inv[0], cmap='hsv', interpolation="None", vmin=0, vmax=180)  # hsv is cyclic, like angles
            axx.set_title('weighted_theta_inv')
            axx.set_axis_off()
            
            axy.imshow(edg[0], cmap='hsv', interpolation="None", vmin=0, vmax=180)  # hsv is cyclic, like angles
            axy.set_title('weighted_theta_edg')
            axy.set_axis_off()
            
            axz.imshow(nan[0], cmap='hsv', interpolation="None", vmin=0, vmax=180)  # hsv is cyclic, like angles
            axz.set_title('theta_nan')
            axz.set_axis_off()
            
            axa.imshow(np.zeros_like(theta_map), cmap='gray')
            axa.imshow(theta_map, cmap='hsv', interpolation="None", vmin=0, vmax=180)
            axa.set_title('theta_all')
            axa.set_axis_off()
            
            # Row 3: R_bar confidence
            ax6.imshow(reg[1], cmap='brg', interpolation="None", vmin=thresh, vmax=1.)
            ax6.set_title('r_bar_reg')
            ax6.set_axis_off()
            
            ax7.imshow(inv[1], cmap='brg', interpolation="None", vmin=thresh, vmax=1.)
            ax7.set_title('r_bar_inv')
            ax7.set_axis_off()
            
            ax8.imshow(edg[1], cmap='brg', interpolation="None", vmin=thresh, vmax=1.)
            ax8.set_title('r_bar_edg')
            ax8.set_axis_off()
            
            ax9.imshow(nan[1], cmap='brg', interpolation="None", vmin=thresh, vmax=1.)
            ax9.set_title('r_bar_nan')
            ax9.set_axis_off()
            
            # ax10.imshow(nan[1], cmap='brg', interpolation="None", vmin=thresh, vmax=1.)
            #
            ax10.imshow(r_vec_3d, interpolation="None")
            ax10.patch.set(hatch='x', edgecolor='lightgrey')
            ax10.set_title('r_vec_3d')
            ax10.set_axis_off()
            
            # # New Figure
            # fig1, ((axA, axB), (axC, axD)) = plt.subplots(2,2)
            # axA.imshow(np.zeros_like(theta_map), cmap='gray')
            # axA.imshow(theta_map, cmap='hsv', interpolation="None", vmin=0, vmax=180)
            # axA.set_title('theta_map')
            # axA.set_axis_off()
            #
            # axB.imshow(np.zeros_like(theta_map), cmap='gray')
            # axB.imshow(from_radial_theta, cmap='hsv', interpolation="None", vmin=0, vmax=180)
            # axB.set_title('theta_map')
            # axB.set_axis_off()
            #
            # # ax9.imshow(from_radial_theta, cmap='hsv', interpolation="None", vmin=0, vmax=180) # hsv is cyclic, like angles
            # # ax9.set_title('from_radial_theta')
            # # ax9.set_axis_off()
            
            self.adjust_rht_plot(fig, zoom=False, shrink=self.shrink_F)
            # self.adjust_rht_plot(fig1, zoom=False, shrink=self.shrink_F)
            plt.show(block=True)
            # plt.savefig("{}\\angles.png".format(r"C:\Users\chgi7364\Dropbox\AB_Interesting_Stuff\Projects\sunback_proj\src\run\renders"))
            # plt.close(fig)
        # plt.imshow(self.theta_array*180/np.pi); plt.show(block=True)
        
        plot_angles = False
        if plot_angles:
            self.plot_angles()
        
        print("Cube Completed!")
    
    def find_h_xy(self, H_XY=None, fudge=0.25):
        """find the reduced hxy matrix, which is set to 0 below a thresh"""
        H_XY = H_XY if H_XY is not None else self.rht_cube
        h_xy = H_XY + 0
        
        thresh = np.max(H_XY) - fudge
        h_xy[H_XY < thresh] = 0
        return h_xy
    
    def find_weighted_sums(self, h_xy=None):
        # theta_map = np.nanmean(np.dstack((weighted_theta_reg, weighted_theta_inv)), axis=2)
        h_xy = h_xy if h_xy is not None else self.h_xy
        
        # Find the normalizing factor
        sum_of_hxy = np.nansum(h_xy, axis=0)
        theta = self.theta[:, None, None]
        
        # Find Cbar
        cos_2theta = np.cos(2 * theta)
        cos_sum_of_hxy = np.nansum(h_xy * cos_2theta, axis=0)
        c_bar = cos_sum_of_hxy / sum_of_hxy
        
        # Find Sbar
        sin_2theta = np.sin(2 * theta)
        sin_sum_of_hxy = np.nansum(h_xy * sin_2theta, axis=0)
        s_bar = sin_sum_of_hxy / sum_of_hxy
        
        # Find thetaBar
        theta_bar = 0.5 * np.arctan2(s_bar, c_bar)
        theta_bar[c_bar < 0] += np.pi
        
        # Find RBar
        r_bar = np.sqrt(c_bar ** 2 + s_bar ** 2)
        
        stdev_circ = np.sqrt(-2 * np.log(r_bar))
        
        return r_bar, stdev_circ
    
    def find_RHT_error(self, H_XY=None):
        h_xy = self.find_h_xy(H_XY=H_XY)
        r_bar, stdev_circ = self.find_weighted_sums(h_xy)
        return r_bar
    
    def prep_inputs(self, shrink=False):
        print("   * Conditioning Inputs...")
        if shrink: self.resize_image()
        self.init_image_frames()
        mdi = self.mask_out_sun(self.params.modified_image)
        self.params.modified_image = self.vignette(mdi)
    
    def resize_image(self, img=None, want_rez=1024):
        print("   * Shrinking Rez to {}...".format(want_rez))
        img = img or self.params.raw_image
        from utils.array_util import reduce_array
        self.params.raw_image, self.params.center = reduce_array(img, self.params.center, want_rez)
        self.params.rez = self.header["NAXIS1"] = want_rez
        self.init_image_frames()
        self.shrink_F = 4 if want_rez == 1024 else 2 if want_rez == 2048 else 1
        self.parse_resize_args(self.shrink_F)
        self.make_radius()
        self.make_vignette(vignette_radius=1.19)
        self.smol = True
    
    def segmentation_jing11(self, use_image=None, doplot=False):
        """Use a series of filters to segment the image into a binary map that
            actually matches the fine structure in the corona.
        """
        print("   * Segmenting Image...")
        use_image = use_image or self.params.modified_image
        use_image_int8 = self.smsh_img_255(use_image)
        
        #########################
        # Highpass and Lowpass
        if self.smol:
            kern, sigma, blur_f = 21, 7, 1.0
        else:
            kern, sigma, blur_f = 41, 18, 1.0
        highpass_img, lowpass_img = self.highpass_filt(use_image_int8, kern, sigma, blur_f)
        
        #########################
        # Bandpass
        kern2, sigma2 = 11, 0.5
        bandpass_img = self.lowpass_filt(highpass_img, kern2, sigma2)
        
        #########################
        # Threshold
        thresh_img, thresh = self.threshold_the_image(bandpass_img)
        # thresh_img_adapt = self.adaptive_threshold_the_image(bandpass_img)
        
        #########################
        # Vignette and mask
        highpass_img = self.donut_the_sun(highpass_img)
        lowpass_img = self.donut_the_sun(lowpass_img)
        bandpass_img = self.donut_the_sun(bandpass_img)
        thresh_img = self.donut_the_sun(thresh_img)
        # thresh_img_adapt = self.donut_the_sun(thresh_img_adapt)
        
        if doplot:
            print(" ** Plot")
            # , (ax6, ax7, ax8)
            fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, sharex=True, sharey=True)  # , figsize=(6, 15))
            ax0.imshow(use_image_int8, cmap='gray', interpolation="None")
            ax0.set_title('Original')
            ax0.set_axis_off()
            
            ax1.imshow(lowpass_img, cmap='gray', interpolation="None")
            ax1.set_title('Gaussian Blur, sigma = {}, bf={}'.format(sigma, blur_f))
            ax1.set_axis_off()
            
            ax2.imshow(self.smsh_img_255(highpass_img), cmap='gray', interpolation="None")  # hsv is cyclic, like angles
            ax2.set_title('High Pass')
            ax2.set_axis_off()
            
            ax3.imshow(self.smsh_img_255(bandpass_img), cmap='gray', interpolation="None")  # hsv is cyclic, like angles
            ax3.set_title('Bandpass, s1 = {}, s2 = {}'.format(sigma, sigma2))
            ax3.set_axis_off()
            
            # ax4.imshow(canny_image, cmap='gray') # hsv is cyclic, like angles
            # ax6.set_title('Canny Edges: t1={}, t2= {}'.format(t1, t2))
            # ax6.set_axis_off()
            
            ax4.imshow(thresh_img, cmap='gray', interpolation="None")  # hsv is cyclic, like angles
            ax4.set_title('Thresholded, thresh= {:0.8}'.format(thresh))
            ax4.set_axis_off()
            
            # ax5.imshow(thresh_img_adapt, cmap='gray', interpolation="None") # hsv is cyclic, like angles
            # ax5.set_title('Thresholded, thresh=Adaptive')
            # ax5.set_axis_off()
            
            self.adjust_rht_plot(fig, zoom=False, shrink=self.shrink_F)
            plt.show(block=True)
            asdf = 1
        
        self.thresholded = thresh_img.astype(np.uint8)
        return self.thresholded
    
    def threshold_the_image(self, image, mu=7 / 8):
        print("   * Threshold")
        # Threshold the Image
        thresh = mu * np.nanmedian(image)
        th, thresh_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        return thresh_img, th
    
    def adaptive_threshold_the_image(self, image):
        print("   * Threshold")
        # Threshold the Image
        th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 0)
        return th3
    
    def run_RHT_algorithm(self, binary_image, outroot=None, H_XY=None, w_factor=1.0):
        outroot_selected = outroot or self.outroot
        if outroot_selected is not self.outroot:
            self.outroot = outroot_selected
            self.make_temp_dir(self.outroot)
            makedirs(self.outroot, exist_ok=True)
        
        # H_XY = H_XY or self.rht_cube
        r_sun_pixels = self.params.header["R_SUN"]
        
        #############################################################
        # Set the width of the RHT Window
        # Boe 2020 says 0.08 to 1.0 Rs. Example given of 31
        # W_RHT = np.round(0.08 * r_sun_pixels).astype(np.uint8)//2
        
        ff = 5 - self.shrink_F
        W_RHT = 31 * ff * w_factor
        
        W_RHT = self.ensure_odd(int(W_RHT))
        print(W_RHT)
        # W_RHT = 55.
        
        #############################################################
        # Set the amount of radial smear in the algorithm
        # Boe 2020 says 0.02 to 0.4 Rs
        smear = 3  # np.round(0.02 * r_sun_pixels).astype(np.uint8)
        # smear = 11.
        
        #############################################################
        frac = 0.6
        
        ## Run RHT Algorithm
        print("\n V V Starting RHT V V")
        # source = self.fits_path
        (H_XY, self.theta) = rht.main(source=self.fits_path,
                                      data=binary_image,
                                      conv=True,
                                      outroot=outroot_selected,
                                      wlen=W_RHT,
                                      smr=smear,
                                      frac=frac,
                                      )
        
        r_bar = self.find_RHT_error(H_XY)
        
        print(" ^  ^  Success!  ^  ^ ")
        return H_XY, self.theta, r_bar
    
    ##############################
    # Plotting Angles
    def plot_angles(self, outroot=None):
        outroot_selected = outroot or self.outroot
        print("\n V Plotting angle images")
        # The first dimension of this H_XY is theta
        self.angle_dir = os.path.join(outroot_selected, "angles_{}".format(self.in_name))
        makedirs(self.angle_dir, exist_ok=True)
        
        self.plot_one_angle(np.nan, self.params.modified_image)
        
        for ii, img in tqdm(enumerate(self.rht_cube), desc=" * Plotting "):
            self.plot_one_angle(self.theta[ii], img)
        
        print("^ Success! Images saved to {}".format(self.angle_dir))
    
    def plot_one_angle(self, theta_rad, img):
        plt.ioff()
        fig, ax = plt.subplots()
        
        if np.isnan(theta_rad):
            theta_clean = "Sum of All"
            ax.set_title(theta_clean)
            angle_path = os.path.join(self.angle_dir, "a_Sum.png")
        
        else:
            theta = theta_rad / np.pi * 180
            theta_clean = "{:0.1f}".format(theta)
            ax.set_title("Angle: {} degrees".format(theta_clean))
            angle_path = os.path.join(self.angle_dir, "{}.png".format(theta_clean))
        
        # img[img==0.] = np.nan
        
        # [big, small] = np.nanpercentile(img, [99, 0.05])
        # img = (img - small) / (big - small)
        ax.imshow(img, origin="lower", interpolation="None")
        
        # Draw the Arrow
        ff = self.shrink_F
        x1, y1 = 3800 // ff, 3800 // ff
        r = 150 // ff
        dx = r * np.sin(-theta_rad)
        dy = r * np.cos(-theta_rad)
        plt.arrow(x1, y1, dx, dy, width=20 // ff, head_width=75 // ff)
        
        # Save
        fig.savefig(angle_path, dpi=500 // ff)
        # plt.show(block=True)
        plt.close(fig)
    
    def adjust_rht_plot(self, fig, lp_wind=0, hp_wind=0, zoom=True, shrink=None):
        
        ff = shrink or 1
        if zoom:
            x1 = 2080 // ff
            y1 = 3850 // ff
            x2 = 2255 // ff
            y2 = 4070 // ff
        else:
            x1 = 2048 // ff
            y1 = 3481 // ff
            x2 = 3526 // ff
            y2 = 4096 // ff
        
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
    
    ##############################
    # Helper Functions
    def donut_the_sun(self, input_img):
        return self.mask_out_sun(self.vignette(input_img))
    
    def ensure_odd(self, number):
        if ~number % 2:
            number += 1
        return number
    
    def highpass_filt(self, image, kern_in=41, sigma_in=18, blur_f=1.0):
        print("   * Highpass")
        gaussian_blur = self.lowpass_filt(image, kern_in, sigma_in)
        highpass_img = image - blur_f * gaussian_blur
        return highpass_img, gaussian_blur
    
    def lowpass_filt(self, image, kern_in=11, sigma_in=1.0):
        print("   * Lowpass")
        # LowPass the Input
        ff = self.shrink_F
        kern2 = kern_in // ff
        sigma2 = sigma_in / ff
        while kern2 < 2 * sigma2:
            kern2 += 1
        kern2 = self.ensure_odd(kern2)
        
        gaussian_blur = cv2.GaussianBlur(src=image, ksize=(kern2, kern2),
                                         sigmaX=sigma2, sigmaY=sigma2)
        return gaussian_blur
    
    def smsh_img_255(self, use_img):
        # print(" * Smooshing to 255...")
        thresh = np.nanpercentile(use_img, [0.5, 99.5])
        normed = self.norm_formula(use_img, *thresh)
        normed[normed >= 1] = 1.
        normed[normed <= 0] = 0.
        smooshed = np.round(normed * 255).astype(np.uint8)
        return smooshed
    
    ##############################
    # Depricated
    def compute_scharr_image_gradient(self, use_image):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
        from scipy import signal
        scharr = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                           # Compute the gradient of an image by 2D convolution with a complex Scharr operator. (Horizontal operator is real, vertical is imaginary.) Use symmetric boundary condition to avoid creating edges at the image boundaries.
                           [-10 + 0j, 0 + 0j, +10 + 0j],
                           [-3 + 3j, 0 + 10j, +3 + 3j]])  # Gx + j*Gy
        
        grad = signal.convolve2d(use_image, scharr, boundary='symm', mode='same')
        magnitude = np.absolute(grad)
        direction = np.angle(grad)
        if True:
            fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(1, 3)  # , figsize=(6, 15))
            ax_orig.imshow(use_image, cmap='gray')
            ax_orig.set_title('Original')
            ax_orig.set_axis_off()
            ax_mag.imshow(np.log10(magnitude), cmap='gray')
            ax_mag.set_title('Gradient magnitude')
            ax_mag.set_axis_off()
            ax_ang.imshow(direction, cmap='hsv')  # hsv is cyclic, like angles
            ax_ang.set_title('Gradient orientation')
            ax_ang.set_axis_off()
            self.adjust_rht_plot(fig)
            plt.show(block=True)
            asdf = 1
        return magnitude, direction
    
    def DEP_spatially_filter_image(self, use_image):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
        from scipy import signal
        import matplotlib.pyplot as plt
        print("Spatial Filtering...")
        
        from astropy.convolution import convolve, convolve_fft, Box2DKernel, CustomKernel
        
        lp_wind = 9  # Options are 3, 5, 7, 9, 11 and different for each image
        hp_wind = 9
        
        low_pass_window = Box2DKernel(lp_wind) * (1 / (lp_wind ** 2))
        # high_pass_window = Box2DKernel(hp_wind)*(-1/(hp_wind**2))
        # high_pass_window[hp_wind//2, hp_wind//2] = 1 - 1/(hp_wind**2)
        
        # low_pass_window  = np.ones((lp_wind, lp_wind))* (1/(lp_wind**2))
        high_pass_window = np.ones((hp_wind, hp_wind)) * (-1 / (hp_wind ** 2))
        high_pass_window[hp_wind // 2, hp_wind // 2] = 2 - 1 / (hp_wind ** 2)
        
        kernal_1 = low_pass_window
        kernal_2 = high_pass_window
        
        # lowpass_img = signal.convolve2d(use_image, kernal_1, boundary='symm', mode='same')
        lowpass_img = convolve(use_image, kernal_1)
        # highpass_img = signal.convolve2d(use_image, kernal_2, boundary='symm', mode='same')
        highpass_img = convolve(use_image, kernal_2, nan_treatment='fill')
        grad = lowpass_img - highpass_img
        
        unsharp = use_image - lowpass_img
        
        print("Smooshing...")
        smooshed = self.smsh_img_255(lowpass_img)
        
        print("Canning...")
        import cv2
        output_image = cv2.Canny(smooshed, 20, 60)
        
        print("Plotting...")
        
        # input_image = Image.fromarray(use_image)
        # image = cv2.cvtColor(use_image, cv2.COLOR_BGR2GRAY )
        a = 1
        # output_image= Image.fromarray(use_image)
        # img = cv2.imdecode(use_image, flags=cv2.IMREAD_GRAYSCALE)
        # edged_image = cv2.imdecode(np.zeros_like(use_image), flags=cv2.IMREAD_GRAYSCALE)
        #
        
        # Plot
        if True:
            fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(16, 12), sharex="all", sharey="all")
            ax0.imshow(np.absolute(use_image), cmap='gray')
            ax0.set_title('Original')
            ax0.set_axis_off()
            
            ax1.imshow(np.absolute(lowpass_img), cmap='gray')
            ax1.set_title('Lowpass')
            ax1.set_axis_off()
            # import matplotlib.markers as markers
            # marker = markers.MarkerStyle(marker='s', fillstyle='none')
            
            location = (2200, 3900) // self.shrink_F
            
            from matplotlib.patches import Rectangle, Circle
            ax1.add_patch(Rectangle(location, lp_wind, lp_wind, zorder=1000, fill=False, edgecolor='b'))
            ax2.add_patch(Rectangle(location, hp_wind, hp_wind, zorder=1000, fill=False, edgecolor='r'))
            
            ax3.add_patch(Rectangle(location, lp_wind, lp_wind, zorder=1000, fill=False, edgecolor='b'))
            ax3.add_patch(Rectangle(location, hp_wind, hp_wind, zorder=1000, fill=False, edgecolor='r'))
            
            ax2.imshow(np.log10(np.absolute(highpass_img)), cmap='gray')  # hsv is cyclic, like angles
            ax2.set_title('Highpass')
            ax2.set_axis_off()
            
            ax3.imshow(np.absolute(grad), cmap='gray')  # hsv is cyclic, like angles
            ax3.set_title('Low - High')
            ax3.set_axis_off()
            
            ax4.imshow(np.absolute(smooshed), cmap='gray')
            ax4.set_title('uint8')
            ax4.set_axis_off()
            
            ax5.imshow(np.absolute(output_image), cmap='gray')
            ax5.set_title('Canny Edged')
            ax5.set_axis_off()
            
            self.adjust_rht_plot(fig, shrink=self.shrink_F)
            plt.show(block=True)
    
    def DEP_make_RHT_binary_map(self, use_image):
        print("Binary Filtering...")
        
        brightness_map_04 = use_image >= 0.4
        brightness_map_05 = use_image >= 0.5
        brightness_map_06 = use_image >= 0.6
        
        if True:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(16, 12), sharex="all", sharey="all")
            ax0.imshow(use_image, cmap='gray')
            ax0.set_title('Original')
            ax0.set_axis_off()
            
            ax1.imshow((brightness_map_04), cmap='gray')
            ax1.set_title('Binary Map 0.4')
            ax1.set_axis_off()
            
            ax2.imshow((brightness_map_05), cmap='gray')  # hsv is cyclic, like angles
            ax2.set_title('Binary Map 0.5')
            ax2.set_axis_off()
            
            ax3.imshow((brightness_map_06), cmap='gray')  # hsv is cyclic, like angles
            ax3.set_title('Binary Map 0.6')
            ax3.set_axis_off()
            
            # fig.suptitle("{}, x1= {}, x2 = {}".format(self.in_name, x1, x2))
            plt.xlim((1500, 2500))
            plt.ylim((3600, 4096))
            plt.tight_layout()
            plt.show(block=True)
        
        pass
        
        # filtered_image = self.compute_scharr_image_gradient(use_image)
        # filtered_image = self.spatially_filter_image(use_image)
        # binary_image = self.make_RHT_binary_map(filtered_image)
        # self.run_RHT_algorithm(binary_image)

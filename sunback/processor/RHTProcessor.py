import os
from copy import copy
import time
from os import makedirs
from os.path import join, dirname, basename

# from astropy.convolution import convolve, convolve_fft, Box2DKernel, CustomKernel, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans, convolve_fft, Gaussian2DKernel

import cv2
import h5py
import matplotlib.pyplot as plt

import astropy.units as u
import sunpy.data.sample
# import sunpy.map

import sunkit_image.radial as radial
from astropy.io import fits
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage, signal
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
# from aiapy.calibrate import normalize_exposure, register, update_pointing

from sunback.processor.Processor import Processor
import warnings

# from src.utils.RHT.rht import rht
# from src.utils.RHT.rht.convRHT import unsharp_mask

from sunback.utils.stretch_intensity_module import norm_stretch

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.ioff()

do_dprint = False
verb = False

# Make a colormap
colors = [
    (0, 0, 0),
    (144 / 255, 99 / 255, 205 / 255),
]  # first color is black, last is red
cm_purp = LinearSegmentedColormap.from_list("Custom", colors, N=255)


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
    progress_verb = "Analyzing"
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
        self.r_vec_3d = None
        self.edg = None
        self.inv = None
        self.reg = None
        self.from_radial_theta = None
        self.theta_map = None
        self.nan = None
        self.fudge_level = None
        self.window_factors = []
        self.index = 0
        self.W_RHT_1024 = 31
        self.current_w_rht = None
        self.count_fraction_threshold = 0.7586
        self.nan_map = None
        self.r_bar = None
        self.h_xy = None
        self.smol = False
        self.shrink_F = 1
        self.thresholded = None
        # self.tm = time.time()
        self.radial_bin_edges = (
            equally_spaced_bins(inner_value=0.0, nbins=300) * u.R_sun
        )
        self.RHT_out_list = []
        self.rht_cube = None
        self.theta = None
        self.in_name = in_name or "rhe"
        self.params.modified_image = None

        if len(self.params.aftereffects_in_name) > 0:
            self.in_name = self.params.aftereffects_in_name.pop(0)

        print("Evaluating frame: {}".format(self.in_name))

    def do_work(self):
        print("")
        shrink = True
        force = False

        get = "rht"
        if get in self.hdu_name_list_trimmed and not force:
            frames_possessed_trimmed = [
                x[4:-1] for x in self.hdu_name_list if "rht" in x
            ]
            # frames_possessed = [x for x in self.hdu_name_list if 'rht' in x]
            frame_requested = self.in_name.split("(")[0]
            if frame_requested in frames_possessed_trimmed:
                get += "(" + frame_requested + ")"
            self.prep_inputs(shrink=shrink, prnt=False)
            self.params.modified_image, wave, t_rec, center, int_time, name = (
                self.load_this_fits_frame(self.fits_path, get)
            )
            self.modified_image = self.theta_map = self.params.modified_image
            self.make_radius()
            self.doplots()
            return None
        else:
            self.prep_inputs(shrink=shrink, prnt=True)
            self.run_RHT()
            self.doplots()

            return np.transpose(self.params.modified_image)

    def load_last_run(self, h5_files):
        file_name = h5_files[0]
        file_path = os.path.join(self.params.temp_directory(), file_name)
        print(" *   Loading h5 file...", end="")
        with h5py.File(file_path, "r") as f:
            self.rht_cube = np.asarray(f["rht_cube"]).T
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

    def get_angles_from_cube(self, theta, cube, r_bar, thresh):
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

    def compare_angle_methods(self, theta_bar, weighted_theta):
        # tbar[tbar>180] -= 180
        # weight = self.wrap_angles(weighted_theta)
        # tbar = np.abs(self.wrap_angles(tbar))

        # weighted_theta = self.wrap_angles(weighted_theta)
        # theta_bar = self.wrap_angles(theta_bar, bounce=False)

        # rad_tbar = self.change_to_angle_from_radial(theta_bar)
        # rad_weight = self.change_to_angle_from_radial(weighted_theta)

        fig, axes = plt.subplots(1, 3, sharex="all", sharey="all")
        for ax in axes:
            # for aa in ax:
            ax.imshow(np.zeros_like(weighted_theta), cmap="gray")

        axes[0].imshow(weighted_theta, cmap="hsv", vmin=0, vmax=180)
        axes[0].set_title("Weighted")
        axes[1].imshow(theta_bar, cmap="hsv", vmin=0, vmax=180)
        axes[1].set_title("Algorithm")
        diff = np.abs(weighted_theta - theta_bar)
        throw = np.nanmax([np.abs(np.nanmax(diff)), np.abs(np.nanmin(diff))])

        axes[2].imshow(diff, cmap="bwr", vmin=-throw, vmax=throw)
        axes[2].set_title("Diff")

        # axes[1,0].imshow(rad_weight, cmap='hsv', vmin=-90, vmax=90)
        # axes[1,0].set_title("Radial, Weighted")
        # axes[1,1].imshow(rad_tbar, cmap='hsv', vmin=-90, vmax=90)
        # axes[1,1].set_title("Radial, Algorithm")
        # axes[1,2].imshow(np.abs(rad_weight - rad_tbar), cmap='bwr', vmin=-180, vmax=180)
        # axes[1,2].set_title("Radial, Diff")

        plt.show(block=True)

    def get_angles_from_bitmap(self, image, outroot=None, thresh=0.95, w_factor=1.0):
        outroot = outroot or self.outroot
        self.thresh = thresh
        cube, theta_list, theta_bar, r_bar = self.run_RHT_algorithm(
            image, outroot=outroot, w_factor=w_factor
        )

        if self.rht_cube is None:
            self.rht_cube = cube
        else:
            self.rht_cube += cube

        too_low = r_bar < thresh
        r_bar[too_low] = np.nan
        theta_bar[too_low] = np.nan

        # weighted_theta, r_bar = self.get_angles_from_cube(theta_list, cube, r_bar, thresh)

        # self.compare_angle_methods(theta_bar, weighted_theta)
        # self.peek_rht(frame, r_bar, weighted_theta)
        # self.fudge_level = "theta_bar"
        return theta_bar, r_bar

    def peek_rht(self, binary_image, r_bar, theta_image):
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

        axes[0].imshow(binary_image, origin="lower", interpolation="None", cmap="gray")
        axes[0].set_title("Binary")
        axes[1].imshow(
            theta_image,
            origin="lower",
            interpolation="None",
            cmap="hsv",
            vmin=0,
            vmax=180,
        )
        axes[1].imshow(
            binary_image, origin="lower", interpolation="None", cmap="gray", alpha=0.2
        )
        axes[1].set_title("Theta")
        ma = axes[2].imshow(
            r_bar,
            origin="lower",
            interpolation="None",
            cmap="RdYlGn",
            vmin=self.thresh,
            vmax=1.0,
        )
        plt.colorbar(mappable=ma, ax=axes[2])

        axes[2].imshow(
            binary_image, origin="lower", interpolation="None", cmap="gray", alpha=0.2
        )
        checksum = np.nansum(r_bar)
        items = np.sum(np.isfinite(r_bar))
        checkquotient = checksum / items
        found = checksum / len(r_bar.flatten())
        axes[2].set_title(
            "Found= {:0.4},  Avg Conf= {:0.4}".format(found, checkquotient)
        )
        # axes[0].imshow(origin="lower", interpolation="None", cmap="grey")

        # location = (2200, 3900) // self.shrink_F

        from matplotlib.patches import Rectangle, Circle

        fig.suptitle(
            "window size: {}, frac = {:0.4}".format(
                self.current_w_rht, self.count_fraction_threshold
            )
        )

        for ax in axes:
            # ax.add_patch(Rectangle([500, 900], self.current_w_rht, self.current_w_rht, zorder=1000, fill=False, edgecolor='b'))
            ax.add_patch(
                Circle(
                    [500, 950],
                    self.current_w_rht // 2,
                    zorder=1000,
                    fill=False,
                    edgecolor="b",
                    lw=3,
                )
            )
            ax.add_patch(
                Circle(
                    [600, 940],
                    self.current_w_rht // 2,
                    zorder=1000,
                    fill=False,
                    edgecolor="b",
                    lw=3,
                )
            )

        plt.subplots_adjust(
            top=0.959, bottom=0.001, left=0.044, right=0.98, hspace=0.0, wspace=0.18
        )
        axes[0].set_xlim((450, 640))
        axes[0].set_ylim((900, 1024))

        saveloc = r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22\W_RHT\{}_{:0.04}.png"
        fig.set_size_inches((8, 15))
        plt.savefig(saveloc.format(self.index, self.count_fraction_threshold))
        self.index += 1
        plt.close(fig)
        # plt.show(block=True)
        # ax2.add_patch(Rectangle(location, hp_wind, hp_wind, zorder=1000, fill=False, edgecolor='r'))

        # ax3.add_patch(Rectangle(location, hp_wind, hp_wind, zorder=1000, fill=False, edgecolor='r'))

    def combine_runs(self, runs, weighted=False):
        if weighted:
            theta_map, runs = self.combine_runs_weighted(runs)
        else:
            theta_map, runs = self.combine_runs_maxima(runs)
        return theta_map, runs

    def combine_RHT_runs_withnan(self, runs):
        theta_map, runs = self.combine_runs(runs)
        # 1/0
        # Find Nans and nan angles
        # self.count_fraction_threshold = 0.8
        self.count_fraction_threshold = 0.65
        self.window_factors = [1.0, 0.6, 1.4]
        theta_map, runs = self.do_nan_run(theta_map, runs, self.window_factors[0])
        # theta_map, runs = self.do_nan_run(theta_map, runs,  self.window_factors[1])
        theta_map, runs = self.do_nan_run(theta_map, runs, self.window_factors[2])

        # smoothed = theta_map3 #self.sgolay2d(df, 31, 2)

        theta_map[self.radius < 1.03 * (self.limb_radius_from_fit_shrunken)] = np.nan

        # import despike
        # print(" V Aftereffects")
        # print(" *   Despiking...")
        # spikes = despike.spikes(theta_map3)
        # print(" *   Cleaning...")
        # clean_img = despike.clean(theta_map3)

        return theta_map, runs

    def combine_runs_maxima(self, runs):
        # This Doesnt Work
        thetas, r_maps = zip(*runs)
        stack_theta = np.dstack(thetas)
        stack_r = np.dstack(r_maps)
        ones = np.ones_like(thetas[0]) * 0.1
        nans = np.ones_like(ones) * np.nan

        stack_r_buffered = np.dstack((stack_r, ones))
        stack_theta_buffered = np.dstack((stack_theta, nans))

        theta_map = np.zeros_like(thetas[0])
        best_ind = np.nanargmax(stack_r_buffered, axis=2)

        shape = theta_map.shape
        for ii in np.arange(shape[0]):
            for jj in np.arange(shape[1]):
                theta_map[ii, jj] = stack_theta_buffered[ii, jj, best_ind[ii, jj]]

        # nan_array = np.ones_like(stack_r[:,:,0]) * np.nan
        # no_data = np.nansum(stack_r, axis=2)==0
        # theta_map = stack_theta_buffered[best_ind.flatten()]
        # for ii, ind in enumerate(best_ind.flatten()):
        #     theta_map[ii] = stack_theta_buffered
        # theta_map[no_data] = np.nan

        # for index, argmax in enumerate(best_ind):
        #     theta_map[index] = stack_r argmax

        # theta_map = stack_theta[best_ind]
        # # best_ind[no_data] = -1
        #
        # stack_mult = stack_theta * stack_r
        # numerator = np.nansum(stack_mult, axis=2)
        # denominator = np.nansum(stack_r, axis=2)
        # theta_map = np.divide(numerator, denominator)
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

    def do_nan_run(self, theta_map, runs, w_factor=1.0, highfrac=True):
        nan_map = np.isnan(theta_map).astype(np.uint8)

        # kernel=np.ones((3,3),np.uint8)
        # closing_op = cv2.morphologyEx(nan_map,cv2.MORPH_CLOSE,kernel)
        # opening_op = cv2.morphologyEx(nan_map,cv2.MORPH_OPEN,kernel)
        # erosion_op = cv2.erode(nan_map,kernel,iterations=1)
        # dilation_op= cv2.dilate(nan_map,kernel,iterations=1)
        # blurred = cv2.GaussianBlur(nan_map, (0,0), 0.5)
        #
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all", sharey='all')
        # ax1.imshow(nan_map, origin='lower', interpolation="none", cmap="gray")
        # ax2.imshow(blurred, origin='lower', interpolation="none", cmap="gray")
        # ax3.imshow(blurred, origin='lower', interpolation="none", cmap="gray")
        # plt.show(block=True)

        self.nan_map = nan_map
        theta_nan, r_bar_nan = self.get_angles_from_bitmap(nan_map, w_factor=w_factor)
        # r_bar_nan[self.radius <= 1] = 0
        runs.append((theta_nan, r_bar_nan))

        # Combine the Maps Again
        theta_map, runs = self.combine_runs(runs)
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
        coord_theta = self.theta_array * 180 / np.pi
        self.shift_theta = 180 - np.mod(coord_theta, 180).T
        from_radial_theta = theta_map - self.shift_theta

        # plt.imshow(np.zeros_like(theta_map), cmap='gray')
        # plt.imshow(from_radial_theta, cmap='hsv', origin="lower")
        # plt.show(block=True)
        return from_radial_theta

    def make_sobel(self, thresholded_image):
        sobel_64 = cv2.Sobel(thresholded_image, cv2.CV_64F, 1, 0, ksize=1)
        abs_64 = np.absolute(sobel_64)
        sobel_8u = np.uint8(abs_64)
        return sobel_8u

    def make_RHT_cube(self):
        print(" * Making Cube...", flush=True)

        # Get Inputs in Line

        # Make the thresholded images
        thresholded_image = self.segmentation_jing11()
        inv_thresholded_image = self.donut_the_sun(255 - thresholded_image)
        sobel_8u = self.make_sobel(thresholded_image)

        # Run RHT on all of them
        self.count_fraction_threshold = thresh = 0.7586
        self.rht_cube = None
        self.index = 0
        self.reg = self.get_angles_from_bitmap(
            thresholded_image, outroot=os.path.join(self.outroot, "reg"), thresh=thresh
        )
        self.inv = self.get_angles_from_bitmap(
            inv_thresholded_image,
            outroot=os.path.join(self.outroot, "inv"),
            thresh=thresh,
        )
        self.edg = self.get_angles_from_bitmap(
            sobel_8u, outroot=os.path.join(self.outroot, "edg"), thresh=thresh
        )

        # Combine the RHT runs
        runs = [self.reg, self.inv, self.edg]
        theta_map, runs = self.combine_RHT_runs_withnan(runs)
        thetas, r_maps = zip(*runs)

        # Extract out the NANmasked component
        self.nan = (np.nanmean(thetas[3:], axis=0), np.nanmean(r_maps[3:], axis=0))

        # Modify the Angle to be radial based
        # from_radial_theta = self.change_to_angle_from_radial(theta_map)

        # Set the output of this filter
        # self.params.modified_image = theta_map
        self.modified_image = self.params.modified_image = theta_map.T
        self.theta_map = theta_map

        ## Plotting

        # slc = self.params.rez//2 + int(self.r2n(1.1))

        # absiss = self.n2r(np.arange(self.params.rez))
        # theta_curve = np.asarray(theta_map)[:, slc]
        # frmrad_curve = np.asarray(from_radial_theta)[:, slc]

        # plt.scatter(absiss, frmrad_curve)
        # plt.scatter(absiss, theta_curve), plt.show(block=True)

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
        # kernal = 1/5 * np.array([[0, 1, 0],          #Compute the gradient of an frame by 2D convolution with a complex Scharr operator. (Horizontal operator is real, vertical is imaginary.) Use symmetric boundary condition to avoid creating edges at the frame boundaries.
        #                          [1, 1, 1],
        #                          [0, 1, 0]]) # Gx + j*Gy
        #
        # canny_image_dialated = cv2.dilate(canny_image, kernel, iterations=1)
        # mag, direct = self.compute_scharr_image_gradient

        # Output dtype = cv2.CV_8U
        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        self.r_vec_3d = np.dstack((self.reg[1], self.inv[1], self.edg[1]))

    def doplots(self):
        self.from_radial_theta = self.change_to_angle_from_radial(self.modified_image)
        # plt.imshow(self.donut_the_sun(self.theta_map),                   origin="lower", cmap='hsv', interpolation="None", vmin=0, vmax=180)

        # plt.imshow(self.donut_the_sun(self.theta_map), interpolation='none', cmap='PuOr', origin='lower', aspect='auto', vmin=-90, vmax=90)
        # plt.show(block=True)
        if False:
            pass
        # self.projected_angle_plot3_both()
        # self.projected_angle_plot3(True)

        # if False:
        # self.angle_plot2_rainbow(self.r_vec_3d, self.theta_map, self.from_radial_theta, self.nan)
        # self.angle_plot2_biplot(True)

        self.plot_model_angular_data()

        # self.angle_plot2_biplot(False)

        # self.angle_plot2_triplot(self.r_vec_3d, self.theta_map, self.from_radial_theta, self.nan, show=True)

        if False:
            self.big_angle_plot(
                self.reg,
                self.inv,
                self.edg,
                self.nan,
                self.theta_map,
                self.from_radial_theta,
                self.thresholded_image,
                self.inv_thresholded_image,
                self.sobel_8u,
            )

        if False:
            self.plot_angles()

    def plot_model_angular_data(self):
        import sunpy
        import astropy.units as u

        # Load all the maps and set the base map
        all_maps = sunpy.map.Map(self.params.use_image_path())
        map_names = [x.fits_header["extname"] for x in all_maps[1:]]
        map_names.insert(0, "LEV1P0")
        base_map, map_name = all_maps[3], map_names[3]

        # Load the deltas from disk
        ax, ay, steve_delta = self.load_model_angular_data()
        steve_delta = steve_delta + 0
        # steve_delta[steve_delta==91] = np.nan

        _, gilly_angles_interp = self.make_angle_arrays()
        gilly_angles_interp = np.abs(gilly_angles_interp)

        from scipy.interpolate import RectBivariateSpline, interp2d

        # steve_interp = RectBivariateSpline(ax, ay, steve_delta+0)
        steve_interp = interp2d(ax, ay, steve_delta + 0, kind="cubic")
        vx = self.xc[0] / self.limb_radius_from_fit_shrunken
        vy = self.yc[:, 0] / self.limb_radius_from_fit_shrunken
        steve_resampled = steve_interp(vx, vy)

        extent = (np.nanmin(vx), np.nanmax(vx), np.nanmin(vy), np.nanmax(vy))

        # #########################################################################################################
        #
        # # Attempt to make the arrays smart
        # basemap_matched_to_steve = base_map.resample(steve_delta.shape*u.pixel)
        # steve_delta_map = sunpy.map.Map((steve_delta, basemap_matched_to_steve.fits_header))
        #
        # basemap_matched_to_gilly = base_map.resample(gilly_angles_interp.shape*u.pixel)
        # gilly_delta_map = sunpy.map.Map((np.abs(gilly_angles_interp), basemap_matched_to_gilly.fits_header))
        # gilly_delta_matched_to_steve = gilly_delta_map.resample(steve_delta.shape*u.pixel)
        # gilly_coords = sunpy.map.all_coordinates_from_map(gilly_delta_matched_to_steve)
        #
        # steve_resampled = sunpy.map.sample_at_coords(steve_delta_map, gilly_coords)
        nk = 75
        smoothing = 10
        # gilly_filtered = self.lowpass_filt(gilly_angles_interp, nk, smoothing )

        #########################################################################################################
        ### Plotting ###

        # Select Plot items
        gilly_toplot = self.lowpass_filt(gilly_angles_interp, nk, smoothing)

        # steve_toplot = adelta_map.data + 0
        steve_toplot = self.lowpass_filt(steve_resampled, nk, smoothing)
        steve_toplot[np.isnan(gilly_toplot)] = np.nan
        # steve_toplot = np.fliplr(np.flipud(steve_toplot))

        difference = np.abs(gilly_toplot - steve_toplot)

        # raw_toplot =  self.lowpass_filt(self.from_radial_theta, nk, smoothing)

        angle_image = self.projected_angle_plot3_both(vibe="xraw")
        # smooth_angle_image = self.lowpass_filt(angle_image, nk, smoothing)

        self.projected_angle_plot3_oneside(
            angle_image, "rhef", False, do_smooth=smoothing
        )
        self.projected_angle_plot3_oneside(angle_image, "rhef_unfiltered", False)

        self.projected_angle_plot3_oneside(difference, "difference", roll=True)
        self.projected_angle_plot3_oneside(steve_toplot, "model-values", roll=True)
        # self.lowpass_filt(gilly_angles_interp, nk, 10 )

        # ade_coords = sunpy.map.all_coordinates_from_map(adelta_map)

        # gilly_delta_matched_to_steve.plot()
        # fig, axes = plt.subplots(3,1, sharex='all', sharey='all')
        # extents = (np.min(self.xx))
        fig = plt.figure()
        ax0 = fig.add_subplot(
            3,
            1,
            1,
        )  #  projection=gilly_delta_matched_to_steve)
        ax1 = fig.add_subplot(
            3, 1, 2, sharex=ax0, sharey=ax0
        )  # , projection=adelta_map)
        ax2 = fig.add_subplot(
            3, 1, 3, sharex=ax0, sharey=ax0
        )  # , projection=adelta_map)

        axes = (ax0, ax1, ax2)

        textloc = (-0.5, 0)
        ax0.text(
            -0.5, -0.1, "smoothing: {}".format(smoothing)
        )  # np.nansum(np.abs(gilly_toplot)) /np.count_nonzero(np.isfinite(gilly_toplot))  ))
        ax0.text(
            *textloc,
            "Avg Px Value: {:0.4} deg".format(np.nanmean(np.abs(gilly_toplot))),
        )  # np.nansum(np.abs(gilly_toplot)) /np.count_nonzero(np.isfinite(gilly_toplot))  ))
        ax1.text(
            *textloc,
            "Avg Px Value: {:0.4} deg".format(np.nanmean(np.abs(steve_toplot))),
        )  # np.nansum(np.abs(steve_toplot)) /np.count_nonzero(np.isfinite(steve_toplot))  ))
        ax2.text(
            *textloc, "Avg Px Value: {:0.4} deg".format(np.nanmean(np.abs(difference)))
        )  # np.nansum(np.abs(difference   )) /np.count_nonzero(np.isfinite(difference   ))  ))

        ax0.imshow(
            gilly_toplot,
            interpolation="none",
            cmap="plasma",
            origin="lower",
            vmin=0,
            vmax=90,
            extent=extent,
        )
        ax1.imshow(
            steve_toplot,
            interpolation="none",
            cmap="plasma",
            origin="lower",
            vmin=0,
            vmax=90,
            extent=extent,
        )
        img = ax2.imshow(
            difference,
            interpolation="none",
            cmap="plasma",
            origin="lower",
            vmin=0,
            vmax=90,
            extent=extent,
        )

        fig.set_size_inches((6.5, 16))
        cax = plt.axes([0.87, 0.04, 0.05, 0.94])
        cbar = plt.colorbar(img, cax=cax, aspect=50)

        cbar.set_label(r"Magnitude of departure from radial, $\delta_r$ (degrees)")
        ax2.set_xlabel("Distance from Sun Center")

        plt.tight_layout()
        plt.subplots_adjust(right=0.88)

        self.params.this_week_path = r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\{}\{}"
        plt.savefig(
            self.params.this_week_path.format("Meeting 10-4-22", "angle_difference.png")
        )

        import matplotlib as mpl
        # cax, kw = mpl.colorbar.make_axes([ax for ax in axes], fraction=0.1)

        # fig.colorbar(img, ax=axes, orientation='horizontal', location="top", aspect=50)

        # bottom, top = 0.1, 0.9
        # left, right = 0.1, 0.8
        #
        # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        # cbar_ax = fig.add_axes([0.85, bottom, 0.05, top-bottom])
        # fig.colorbar(img, cax=cbar_ax)

        # plt.subplots_adjust(top=0.735,
        #                     bottom=0.11,
        #                     left=0.035,
        #                     right=0.985,
        #                     hspace=0.16,
        #                     wspace=0.135)

        # fig.colorbar(im, cax=cbar_ax)

        # plt.colorbar(img, ax=ax0, aspect=60) #, orientation='horizontal', location="top")

        # self.autoLabelPanels(axes)
        # plt.colorbar(img, ax=axes[0], orientation='horizontal', aspect=60, location="top")
        # for ax in axes:
        #     pass

        # plt.tight_layout()
        plt.close(fig)
        # plt.show(block=True)

        # ax2.imshow(gilly_angles_raw,          interpolation='none', cmap='plasma',  origin='lower')#, vmin=-90, vmax=90)

    def load_model_angular_data(self):
        from scipy.io import readsav

        filepath = r"C:\Users\chgi7364\Dropbox\AB_Interesting_Stuff\Projects\sunback_proj\data\delta2D_CR2212_clongobs_204.sav"
        angle_dict = readsav(filepath)
        nax, nay = angle_dict["nx"], angle_dict["ny"]
        ax = angle_dict["x"]
        ay = angle_dict["y"]
        adelta = angle_dict["delta"]
        return ax, ay, adelta

    def projected_angle_plot(self, r_vec_3D, theta_map, from_radial_theta, nan_r):
        """Vertical Plot"""
        unwrapped_rvec = np.asarray(
            [self.unwrap_polar(item).T for item in r_vec_3D.T]
        ).T
        fig = self.angle_plot(
            unwrapped_rvec,
            self.unwrap_polar(theta_map),
            self.unwrap_polar(from_radial_theta),
            self.unwrap_polar(nan_r),
        )
        plt.ylim((600, theta_map.shape[1]))
        # plt.xlim((512, 1024))
        fig.set_size_inches((10, 14))
        plt.tight_layout()
        plt.savefig(
            "{}\\7_angles.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
            )
        )
        plt.show(block=True)
        # plt.imshow(self.unwrap_polar(reg[0]),        origin="lower", cmap='hsv', interpolation="None"), , plt.show(block=True)

    def projected_angle_plot2(self, r_vec_3D, theta_map, from_radial_theta, nan_r):
        """Wide Plot"""
        unwrapped_rvec = np.asarray(
            [self.unwrap_polar(item).T for item in r_vec_3D.T]
        ).T
        fig = self.angle_plot(
            unwrapped_rvec,
            self.unwrap_polar(theta_map),
            self.unwrap_polar(from_radial_theta),
            self.unwrap_polar(nan_r),
        )
        plt.ylim((610, 780))
        # plt.xlim((512, 1024))
        fig.set_size_inches((18, 10))
        plt.tight_layout()
        plt.savefig(
            "{}\\8_angles.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
            )
        )
        plt.show(block=True)
        # plt.imshow(self.unwrap_polar(reg[0]),        origin="lower", cmap='hsv', interpolation="None"), , plt.show(block=True)

    def wrap_angles(self, unwrapped, bounce=True, abs=False):
        if bounce:
            unwrapped[unwrapped > 90] = 180 - unwrapped[unwrapped > 90]
            unwrapped[unwrapped < -90] = -180 - unwrapped[unwrapped < -90]
        else:
            unwrapped[unwrapped > 90] = unwrapped[unwrapped > 90] - 180
            unwrapped[unwrapped < -90] = unwrapped[unwrapped < -90] + 180

        unwrapped[:] = np.abs(unwrapped[:]) if abs else unwrapped[:]
        return unwrapped

    def roll_half_array(self, arr):
        return np.roll(arr, len(arr) // 2)

    def projected_angle_plot3_oneside(
        self, use_image=None, vibe="both2", do_unwrap=True, do_smooth=None, roll=None
    ):
        """from radial only"""

        use_image = use_image if use_image is not None else self.from_radial_theta

        # Make the figure,
        fig, axes = plt.subplots(
            2, 1, sharex="all", gridspec_kw={"height_ratios": [1, 1]}
        )
        (ax0, ax1) = axes
        ax0.set_title(vibe)
        # for ax in axes:
        # ax0.set_axis_off()
        radius, thetas = self.unwrap_coords()
        radius /= self.limb_radius_from_fit_shrunken

        rad_absiss = radius[:, 0]
        theta_absiss = thetas[self.params.rez // 2]
        rmin, rmax = np.nanmin(radius), np.nanmax(radius)
        tmin, tmax = np.nanmin(theta_absiss), np.nanmax(theta_absiss)
        extents = (0, 360, rmin, rmax)
        fake_theta_ax = np.linspace(0, 360, self.params.rez)
        # Transform the Array
        if do_unwrap:
            unwrapped_from_radial = np.fliplr(self.unwrap_polar(use_image))
            if roll is not None:
                # roll_n = int((140)/360 * unwrapped_from_radial.shape[1])
                # unwrapped_from_radial = np.roll(unwrapped_from_radial, roll_n, axis=1)
                # unwrapped_from_radial = np.fliplr(unwrapped_from_radial)
                pass

        else:
            unwrapped_from_radial = use_image

        # Interpolate the missing components
        interpolated = interpolate_replace_nans(
            unwrapped_from_radial, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft
        )
        masked = interpolated + 0
        # masked = unwrapped_from_radial + 0
        masked[radius > 1.6] = np.nan
        masked[radius < 1.01] = np.nan

        angle_image_wrapped = self.wrap_angles(masked, abs=False)

        if do_smooth is not None:
            angle_image = np.abs(
                self.lowpass_filt((angle_image_wrapped + 0), 100, do_smooth)
            )
        else:
            angle_image = angle_image_wrapped

        # Plot Interpolated Array
        # ax0.set_title('Angle From Radial, {}'.format(64))

        # img = ax0.imshow(self.wrap_angles(masked),          interpolation='none', cmap='PuOr', origin='lower', aspect='auto', vmin=-90, vmax=90) #, extent=extents)
        # img = ax0.pcolormesh(rad_absiss, theta_absiss, angle_image[:-1,:-1]) #, cmap='PuOr', vmin=-90, vmax=90,)
        ax0.imshow(
            np.ones_like(angle_image),
            interpolation="none",
            cmap="gray",
            origin="lower",
            aspect="auto",
            vmin=-90,
            vmax=90,
            extent=extents,
        )
        img = ax0.imshow(
            np.abs(angle_image),
            interpolation="none",
            cmap="plasma",
            origin="lower",
            aspect="auto",
            extent=extents,
        )  # , vmin=-90, vmax=90,
        ax0.set_xlim((0, 360))
        ax0.set_ylim((rmin, rmax))

        ax0.set_ylim((1.025, 1.275))
        # plt.show(block=True)

        # plt.show(block=True)

        if True:
            # Make a colormap
            from matplotlib import cm

            average_rows = 16
            bott, topp = (
                int(1.06 * self.params.rez / rmax),
                int(1.24 * self.params.rez / rmax),
            )
            rows = np.arange(bott, topp, average_rows)
            n_rows = len(rows)
            viridis = cm.get_cmap("viridis", n_rows)

            # Make Averaging Choic
            half_rows = average_rows // 2

            wraplist = []
            # Plot Array Slices
            # print(self.limb_radius_from_header, self.limb_radius_from_fit_shrunken)
            for ind, row_ii in enumerate(rows):
                c = viridis(ind)

                rrr = row_ii / self.params.rez * rmax
                # print(rrr)
                ax0.axhline(rrr, ls="--", c=c, lw=2.5)

                bot, top = row_ii - half_rows, row_ii + half_rows
                band = angle_image[bot:top, :]
                toplot = np.nanmean(band, axis=0)
                # unwrapped = self.wrap_angles(toplot)

                # ax1.plot(savgol_filter(unwrapped, 21, 2), c=c, zorder=1000+ind) # label="Row: {}".format(row_ii))

                cc = list(c)
                cc[-1] = 0.6
                ax1.plot(
                    fake_theta_ax,
                    np.abs(toplot),
                    "o",
                    markerfacecolor=cc,
                    markeredgewidth=0,
                )  # , c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                # ax2.plot(fake_theta_ax, toplot, 'o', markerfacecolor=cc, markeredgewidth=0) #, c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                # ax1.plot(theta_absiss, toplot, 'o', markerfacecolor=cc, markeredgewidth=0) #, c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                # wraplist.append(unwrapped)
                # ax1.plot(unwrapped, c=c, label="Row: {}".format(row_ii))

            # wraparray = np.asarray(wraplist)

            ax0.set_ylabel(r"Height $r/R_\odot$")
            ax1.set_xlabel("Position Angle, with 270 = Solar North")
            ax1.set_ylabel(r"Superradial Magnitude |$\delta_r$|")
            # ax2.set_ylabel(r"Superradial Angle $\delta_r$")
            # ax1.set_title("Slices of Constant Radii")
            # ax1.legend(loc="lower left", frameon=False)
            # toplot = np.nanmean(band, axis=0)
            # unwrapped = np.unwrap_polar(2*toplot)/2
            # ax1.plot(unwrapped, c=c, ls="--")
            # ax1.plot(np.unwrap_polar(toplot), c=c)

            fig.set_size_inches((18, 6))
            # plt.colorbar(img, ax=axes[0], orientation='horizontal', aspect=60, location="top")
            plt.tight_layout()
            # plt.grid(True, which='both')
            plt.minorticks_on()

            ax = ax1
            ax.axhline(0, c="k", lw=2.5)  # , label="datalim")
            ax.axhline(90, c="k", lw=1.5)  # , label="datalim")
            ax.axhline(-90, c="k", lw=1.5)  # , label="datalim")

            ax.axvline(0, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax.axvline(90, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax.axvline(180, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax.axvline(270, c="firebrick", lw=2.0, ls=":")  # , label="datalim")
            ax.axvline(360, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax.grid(True, which="both")

            ax0.axvline(0, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")
            ax0.axvline(90, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")
            ax0.axvline(180, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")
            ax0.axvline(
                270, c="firebrick", lw=2.0, ls=":", zorder=500
            )  # , label="datalim")
            ax0.axvline(360, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")

            ax1.set_ylim((-5, 95))
            # ax2.set_ylim((-95, 95))
        # Make the plot obey the frame
        # ax1.set_aspect('equal', share=True, adjustable="box")

        # ax1.imshow(masked,          interpolation='none', cmap='hsv', origin='lower') #, extent=extents)
        # ax1.pcolormesh(theta_absiss, rad_absiss, masked, cmap='hsv')

        # axes[0].imshow(np.zeros_like(from_radial_theta), origin="lower", cmap='gray', interpolation="None")
        # ax0.imshow(interpolated,    interpolation='none', cmap='hsv', origin='lower')

        # ax1.set_title('Angle From Radial')
        # axes[0].pcolorfast(rad_absiss, theta_absiss, interpolated, cmap='hsv')

        # axes[0].imshow(unwrapped_from_radial,                                                                                 interpolation='none', cmap='hsv', origin='lower')
        # axes[1].imshow(interpolate_replace_nans(unwrapped_from_radial, Gaussian2DKernel(x_stddev=1), convolve=convolve_fft),  interpolation='none', cmap='hsv', origin='lower')
        # axes[3].imshow(interpolate_replace_nans(unwrapped_from_radial, Gaussian2DKernel(x_stddev=3), convolve=convolve_fft),  interpolation='none', cmap='hsv', origin='lower')
        plt.tight_layout()
        # "ABS" if one_sided else "BOTH"
        # plt.show(block=True)
        plt.savefig(
            "{}\\9_slices_{}_{}rows.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22",
                vibe,
                average_rows,
            )
        )
        plt.close(fig)
        # plt.show(block=True)
        #
        # a=1
        #

        # # unwrapped_from_radial
        #
        # # ax2.plot()
        #
        #
        # # from astropy.nddata import block_reduce
        # # r_factor = 4
        # # reduced_from_radial = block_reduce(unwrapped_from_radial, 2, np.nansum)
        # # # upscaled_from_radial = cv2.resize(reduced_from_radial, (1024, 1024))
        # # ax2.set_title('Reduced From Radial')
        # # ax2.imshow(np.zeros_like(reduced_from_radial), origin="lower", cmap='gray', interpolation="None")
        # # ax2.imshow(reduced_from_radial,        origin="lower", cmap='hsv', interpolation="None")
        #
        # # ax2.set_title('Waterfall')
        # # x = np.arange(self.params.rez)
        # # y = np.arange(self.params.rez//r_factor)
        # # X,Y = np.meshgrid(x,y)
        # # self.waterfall_plot(fig,ax, X=X,  Y=Y, Z=unwrapped_from_radial)
        # # ax2.imshow(np.zeros_like(unwrapped_from_radial), origin="lower", cmap='gray', interpolation="None")
        # # # ax3.imshow(unwrapped_from_radial,        origin="lower", cmap='hsv', interpolation="None")
        #
        # # fig = self.angle_plot(unwrapped_rvec, self.unwrap_polar(theta_map),, self.unwrap_polar(nan_r))
        # plt.ylim((610, 780))
        # # plt.xlim((512, 1024))
        # fig.set_size_inches((18,10))
        # plt.tight_layout()
        # plt.savefig("{}\\9_angles.png".format(r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"))
        # # plt.show(block=True)
        # # plt.imshow(self.unwrap_polar(reg[0]),        origin="lower", cmap='hsv', interpolation="None"), , plt.show(block=True)
        #

    def projected_angle_plot3_both(self, use_image=None, vibe="both2"):
        """from radial only"""

        use_image = use_image if use_image is not None else self.from_radial_theta

        # Make the figure,
        fig, axes = plt.subplots(
            3, 1, sharex="all", gridspec_kw={"height_ratios": [2, 1, 2]}
        )
        (ax0, ax1, ax2) = axes
        # for ax in axes:
        # ax0.set_axis_off()
        radius, thetas = self.unwrap_coords()
        radius /= self.limb_radius_from_fit_shrunken

        rad_absiss = radius[:, 0]
        theta_absiss = thetas[self.params.rez // 2]
        rmin, rmax = np.nanmin(radius), np.nanmax(radius)
        tmin, tmax = np.nanmin(theta_absiss), np.nanmax(theta_absiss)
        extents = (0, 360, rmin, rmax)
        fake_theta_ax = np.linspace(0, 360, self.params.rez)
        # Transform the Array
        unwrapped_from_radial = np.fliplr(self.unwrap_polar(use_image))

        # Interpolate the missing components
        interpolated = interpolate_replace_nans(
            unwrapped_from_radial, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft
        )
        masked = interpolated + 0
        # masked = unwrapped_from_radial + 0
        masked[radius > 1.6] = np.nan
        masked[radius < 1.01] = np.nan

        angle_image = self.wrap_angles(masked, abs=False)

        # Plot Interpolated Array
        # ax0.set_title('Angle From Radial, {}'.format(64))

        # img = ax0.imshow(self.wrap_angles(masked),          interpolation='none', cmap='PuOr', origin='lower', aspect='auto', vmin=-90, vmax=90) #, extent=extents)
        # img = ax0.pcolormesh(rad_absiss, theta_absiss, angle_image[:-1,:-1]) #, cmap='PuOr', vmin=-90, vmax=90,)
        ax0.imshow(
            np.ones_like(angle_image),
            interpolation="none",
            cmap="gray",
            origin="lower",
            aspect="auto",
            vmin=-90,
            vmax=90,
            extent=extents,
        )
        img = ax0.imshow(
            np.abs(angle_image),
            interpolation="none",
            cmap="plasma",
            origin="lower",
            aspect="auto",
            extent=extents,
        )  # , vmin=-90, vmax=90,
        ax0.set_xlim((0, 360))
        ax0.set_ylim((rmin, rmax))

        ax0.set_ylim((1.025, 1.275))
        # plt.show(block=True)

        # plt.show(block=True)

        if True:
            # Make a colormap
            from matplotlib import cm

            average_rows = 16
            bott, topp = (
                int(1.06 * self.params.rez / rmax),
                int(1.24 * self.params.rez / rmax),
            )
            rows = np.arange(bott, topp, average_rows)
            n_rows = len(rows)
            viridis = cm.get_cmap("viridis", n_rows)

            # Make Averaging Choic
            half_rows = average_rows // 2

            wraplist = []
            # Plot Array Slices
            # print(self.limb_radius_from_header, self.limb_radius_from_fit_shrunken)
            for ind, row_ii in enumerate(rows):
                c = viridis(ind)

                rrr = row_ii / self.params.rez * rmax
                # print(rrr)
                ax0.axhline(rrr, ls="--", c=c, lw=2.5)

                bot, top = row_ii - half_rows, row_ii + half_rows
                band = angle_image[bot:top, :]
                toplot = np.nanmean(band, axis=0)
                # unwrapped = self.wrap_angles(toplot)

                # ax1.plot(savgol_filter(unwrapped, 21, 2), c=c, zorder=1000+ind) # label="Row: {}".format(row_ii))

                cc = list(c)
                cc[-1] = 0.6
                ax1.plot(
                    fake_theta_ax,
                    np.abs(toplot),
                    "o",
                    markerfacecolor=cc,
                    markeredgewidth=0,
                )  # , c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                ax2.plot(
                    fake_theta_ax, toplot, "o", markerfacecolor=cc, markeredgewidth=0
                )  # , c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                # ax1.plot(theta_absiss, toplot, 'o', markerfacecolor=cc, markeredgewidth=0) #, c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                # wraplist.append(unwrapped)
                # ax1.plot(unwrapped, c=c, label="Row: {}".format(row_ii))

            # wraparray = np.asarray(wraplist)

            ax0.set_ylabel(r"Height $r/R_\odot$")
            ax2.set_xlabel("Position Angle, with 270 = Solar North")
            ax1.set_ylabel(r"Superradial Magnitude |$\delta_r$|")
            ax2.set_ylabel(r"Superradial Angle $\delta_r$")
            # ax1.set_title("Slices of Constant Radii")
            # ax1.legend(loc="lower left", frameon=False)
            # toplot = np.nanmean(band, axis=0)
            # unwrapped = np.unwrap_polar(2*toplot)/2
            # ax1.plot(unwrapped, c=c, ls="--")
            # ax1.plot(np.unwrap_polar(toplot), c=c)

            fig.set_size_inches((18, 8))
            plt.colorbar(
                img, ax=axes[0], orientation="horizontal", aspect=60, location="top"
            )
            plt.tight_layout()
            # plt.grid(True, which='both')
            plt.minorticks_on()

            for ax in (ax1, ax2):
                ax.axhline(0, c="k", lw=2.5)  # , label="datalim")
                ax.axhline(90, c="k", lw=1.5)  # , label="datalim")
                ax.axhline(-90, c="k", lw=1.5)  # , label="datalim")

                ax.axvline(0, c="k", lw=2.0, ls=":")  # , label="datalim")
                ax.axvline(90, c="k", lw=2.0, ls=":")  # , label="datalim")
                ax.axvline(180, c="k", lw=2.0, ls=":")  # , label="datalim")
                ax.axvline(270, c="firebrick", lw=2.0, ls=":")  # , label="datalim")
                ax.axvline(360, c="k", lw=2.0, ls=":")  # , label="datalim")
                ax.grid(True, which="both")

            ax0.axvline(0, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")
            ax0.axvline(90, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")
            ax0.axvline(180, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")
            ax0.axvline(
                270, c="firebrick", lw=2.0, ls=":", zorder=500
            )  # , label="datalim")
            ax0.axvline(360, c="k", lw=2.0, ls=":", zorder=500)  # , label="datalim")

            ax1.set_ylim((-5, 95))
            ax2.set_ylim((-95, 95))
        # Make the plot obey the frame
        # ax1.set_aspect('equal', share=True, adjustable="box")

        # ax1.imshow(masked,          interpolation='none', cmap='hsv', origin='lower') #, extent=extents)
        # ax1.pcolormesh(theta_absiss, rad_absiss, masked, cmap='hsv')

        # axes[0].imshow(np.zeros_like(from_radial_theta), origin="lower", cmap='gray', interpolation="None")
        # ax0.imshow(interpolated,    interpolation='none', cmap='hsv', origin='lower')

        # ax1.set_title('Angle From Radial')
        # axes[0].pcolorfast(rad_absiss, theta_absiss, interpolated, cmap='hsv')

        # axes[0].imshow(unwrapped_from_radial,                                                                                 interpolation='none', cmap='hsv', origin='lower')
        # axes[1].imshow(interpolate_replace_nans(unwrapped_from_radial, Gaussian2DKernel(x_stddev=1), convolve=convolve_fft),  interpolation='none', cmap='hsv', origin='lower')
        # axes[3].imshow(interpolate_replace_nans(unwrapped_from_radial, Gaussian2DKernel(x_stddev=3), convolve=convolve_fft),  interpolation='none', cmap='hsv', origin='lower')
        plt.tight_layout()
        # "ABS" if one_sided else "BOTH"
        # plt.show(block=True)
        plt.savefig(
            "{}\\9_slices_{}_{}rows.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22",
                vibe,
                average_rows,
            )
        )
        plt.close(fig)
        # plt.show(block=True)
        #
        # a=1
        #

        # # unwrapped_from_radial
        #
        # # ax2.plot()
        #
        #
        # # from astropy.nddata import block_reduce
        # # r_factor = 4
        # # reduced_from_radial = block_reduce(unwrapped_from_radial, 2, np.nansum)
        # # # upscaled_from_radial = cv2.resize(reduced_from_radial, (1024, 1024))
        # # ax2.set_title('Reduced From Radial')
        # # ax2.imshow(np.zeros_like(reduced_from_radial), origin="lower", cmap='gray', interpolation="None")
        # # ax2.imshow(reduced_from_radial,        origin="lower", cmap='hsv', interpolation="None")
        #
        # # ax2.set_title('Waterfall')
        # # x = np.arange(self.params.rez)
        # # y = np.arange(self.params.rez//r_factor)
        # # X,Y = np.meshgrid(x,y)
        # # self.waterfall_plot(fig,ax, X=X,  Y=Y, Z=unwrapped_from_radial)
        # # ax2.imshow(np.zeros_like(unwrapped_from_radial), origin="lower", cmap='gray', interpolation="None")
        # # # ax3.imshow(unwrapped_from_radial,        origin="lower", cmap='hsv', interpolation="None")
        #
        # # fig = self.angle_plot(unwrapped_rvec, self.unwrap_polar(theta_map),, self.unwrap_polar(nan_r))
        # plt.ylim((610, 780))
        # # plt.xlim((512, 1024))
        # fig.set_size_inches((18,10))
        # plt.tight_layout()
        # plt.savefig("{}\\9_angles.png".format(r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"))
        # # plt.show(block=True)
        # # plt.imshow(self.unwrap_polar(reg[0]),        origin="lower", cmap='hsv', interpolation="None"), , plt.show(block=True)
        #
        return angle_image

    def projected_angle_plot3(self, one_sided=False):
        """Does either the one sided or double sided unwrapped plot. the both version is preferable, this is depricated."""
        # Make the figure
        fig, axes = plt.subplots(2, 1, sharex=True)  # , sharey="all")
        (ax0, ax1) = axes
        # for ax in axes:
        # ax0.set_axis_off()
        radius, thetas = self.unwrap_coords()
        radius /= self.limb_radius_from_fit_shrunken

        rad_absiss = radius[:, 0]
        theta_absiss = thetas[self.params.rez // 2]
        rmin, rmax = np.nanmin(radius), np.nanmax(radius)
        tmin, tmax = np.nanmin(theta_absiss), np.nanmax(theta_absiss)
        extents = (0, 360, rmin, rmax)
        fake_theta_ax = np.linspace(0, 360, self.params.rez)
        # Transform the Array
        unwrapped_from_radial = np.fliplr(self.unwrap_polar(self.from_radial_theta))

        # Interpolate the missing components
        interpolated = interpolate_replace_nans(
            unwrapped_from_radial, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft
        )
        masked = interpolated + 0
        # masked = unwrapped_from_radial + 0
        masked[radius > 1.6] = np.nan
        masked[radius < 1.01] = np.nan

        angle_image = self.wrap_angles(masked, abs=False)

        # Plot Interpolated Array
        # ax0.set_title('Angle From Radial, {}'.format(64))

        # img = ax0.imshow(self.wrap_angles(masked),          interpolation='none', cmap='PuOr', origin='lower', aspect='auto', vmin=-90, vmax=90) #, extent=extents)
        # img = ax0.pcolormesh(rad_absiss, theta_absiss, angle_image[:-1,:-1]) #, cmap='PuOr', vmin=-90, vmax=90,)
        ax0.imshow(
            np.ones_like(angle_image),
            interpolation="none",
            cmap="gray",
            origin="lower",
            aspect="auto",
            vmin=-90,
            vmax=90,
            extent=extents,
        )
        img = ax0.imshow(
            np.abs(angle_image),
            interpolation="none",
            cmap="plasma",
            origin="lower",
            aspect="auto",
            extent=extents,
        )  # , vmin=-90, vmax=90,
        ax0.set_xlim((0, 360))
        ax0.set_ylim((rmin, rmax))

        ax0.set_ylim((1.025, 1.275))
        # plt.show(block=True)

        # plt.show(block=True)

        if True:
            # Make a colormap
            from matplotlib import cm

            average_rows = 20
            bott, topp = (
                int(1.04 * self.params.rez / rmax),
                int(1.26 * self.params.rez / rmax),
            )
            rows = np.arange(bott, topp, average_rows)
            n_rows = len(rows)
            viridis = cm.get_cmap("viridis", n_rows)

            # Make Averaging Choic
            half_rows = average_rows // 2

            wraplist = []
            # Plot Array Slices
            # print(self.limb_radius_from_header, self.limb_radius_from_fit_shrunken)
            for ind, row_ii in enumerate(rows):
                c = viridis(ind)

                rrr = row_ii / self.params.rez * rmax
                # print(rrr)
                ax0.axhline(rrr, ls="--", c=c, lw=2.5)

                bot, top = row_ii - half_rows, row_ii + half_rows
                band = angle_image[bot:top, :]
                toplot = np.nanmean(band, axis=0)
                # unwrapped = self.wrap_angles(toplot)

                # ax1.plot(savgol_filter(unwrapped, 21, 2), c=c, zorder=1000+ind) # label="Row: {}".format(row_ii))

                cc = list(c)
                cc[-1] = 0.6
                plotty = np.abs(toplot) if one_sided else toplot
                ax1.plot(
                    fake_theta_ax, plotty, "o", markerfacecolor=cc, markeredgewidth=0
                )  # , c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                # ax1.plot(theta_absiss, toplot, 'o', markerfacecolor=cc, markeredgewidth=0) #, c=(0,0,0,0)) # label="Row: {}".format(row_ii))
                # wraplist.append(unwrapped)
                # ax1.plot(unwrapped, c=c, label="Row: {}".format(row_ii))

            # wraparray = np.asarray(wraplist)

            ax1.set_xlabel("Position Angle")
            ax1.set_ylabel("Deviation from radial in degrees")
            # ax1.set_title("Slices of Constant Radii")
            # ax1.legend(loc="lower left", frameon=False)
            # toplot = np.nanmean(band, axis=0)
            # unwrapped = np.unwrap_polar(2*toplot)/2
            # ax1.plot(unwrapped, c=c, ls="--")
            # ax1.plot(np.unwrap_polar(toplot), c=c)

            fig.set_size_inches((18, 10))
            plt.colorbar(
                img, ax=axes[0], orientation="horizontal", aspect=60, location="top"
            )
            plt.tight_layout()
            plt.grid(True, which="both")
            plt.minorticks_on()
            ax1.axhline(0, c="k", lw=2.5)  # , label="datalim")
            ax1.axhline(90, c="k", lw=1.5)  # , label="datalim")
            ax1.axhline(-90, c="k", lw=1.5)  # , label="datalim")

            ax1.axvline(0, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax1.axvline(90, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax1.axvline(180, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax1.axvline(270, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax1.axvline(360, c="k", lw=2.0, ls=":")  # , label="datalim")

            ax0.axvline(0, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax0.axvline(90, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax0.axvline(180, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax0.axvline(270, c="k", lw=2.0, ls=":")  # , label="datalim")
            ax0.axvline(360, c="k", lw=2.0, ls=":")  # , label="datalim")

            ax1.set_ylim((-95, 95))
        # Make the plot obey the frame
        # ax1.set_aspect('equal', share=True, adjustable="box")

        # ax1.imshow(masked,          interpolation='none', cmap='hsv', origin='lower') #, extent=extents)
        # ax1.pcolormesh(theta_absiss, rad_absiss, masked, cmap='hsv')

        # axes[0].imshow(np.zeros_like(from_radial_theta), origin="lower", cmap='gray', interpolation="None")
        # ax0.imshow(interpolated,    interpolation='none', cmap='hsv', origin='lower')

        # ax1.set_title('Angle From Radial')
        # axes[0].pcolorfast(rad_absiss, theta_absiss, interpolated, cmap='hsv')

        # axes[0].imshow(unwrapped_from_radial,                                                                                 interpolation='none', cmap='hsv', origin='lower')
        # axes[1].imshow(interpolate_replace_nans(unwrapped_from_radial, Gaussian2DKernel(x_stddev=1), convolve=convolve_fft),  interpolation='none', cmap='hsv', origin='lower')
        # axes[3].imshow(interpolate_replace_nans(unwrapped_from_radial, Gaussian2DKernel(x_stddev=3), convolve=convolve_fft),  interpolation='none', cmap='hsv', origin='lower')
        plt.tight_layout()
        vibe = "ABS" if one_sided else "BOTH"
        # plt.show(block=True)
        plt.savefig(
            "{}\\9_slices_{}_{}rows_plasma.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22",
                vibe,
                average_rows,
            )
        )

        # plt.show(block=True)
        #
        # a=1
        #

        # # unwrapped_from_radial
        #
        # # ax2.plot()
        #
        #
        # # from astropy.nddata import block_reduce
        # # r_factor = 4
        # # reduced_from_radial = block_reduce(unwrapped_from_radial, 2, np.nansum)
        # # # upscaled_from_radial = cv2.resize(reduced_from_radial, (1024, 1024))
        # # ax2.set_title('Reduced From Radial')
        # # ax2.imshow(np.zeros_like(reduced_from_radial), origin="lower", cmap='gray', interpolation="None")
        # # ax2.imshow(reduced_from_radial,        origin="lower", cmap='hsv', interpolation="None")
        #
        # # ax2.set_title('Waterfall')
        # # x = np.arange(self.params.rez)
        # # y = np.arange(self.params.rez//r_factor)
        # # X,Y = np.meshgrid(x,y)
        # # self.waterfall_plot(fig,ax, X=X,  Y=Y, Z=unwrapped_from_radial)
        # # ax2.imshow(np.zeros_like(unwrapped_from_radial), origin="lower", cmap='gray', interpolation="None")
        # # # ax3.imshow(unwrapped_from_radial,        origin="lower", cmap='hsv', interpolation="None")
        #
        # # fig = self.angle_plot(unwrapped_rvec, self.unwrap_polar(theta_map),, self.unwrap_polar(nan_r))
        # plt.ylim((610, 780))
        # # plt.xlim((512, 1024))
        # fig.set_size_inches((18,10))
        # plt.tight_layout()
        # plt.savefig("{}\\9_angles.png".format(r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"))
        # # plt.show(block=True)
        # # plt.imshow(self.unwrap_polar(reg[0]),        origin="lower", cmap='hsv', interpolation="None"), , plt.show(block=True)
        #

    @staticmethod
    def waterfall_plot(fig, ax, X, Y, Z):
        """
        Make a waterfall plot
        Input:
            fig,ax : matplotlib figure and axes to populate
            Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
            X,Y : n,m array
        """
        # Set normalization to the same values for all plots
        norm = plt.Normalize(Z.min().min(), Z.max().max())
        # Check sizes to loop always over the smallest dimension
        n, m = Z.shape
        if n > m:
            X = X.T
            Y = Y.T
            Z = Z.T
            m, n = n, m

        for j in range(n):
            # reshape the X,Z into pairs
            points = np.array([X[j, :], Z[j, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap="plasma", norm=norm)
            # Set the values used for colormapping
            lc.set_array((Z[j, 1:] + Z[j, :-1]) / 2)
            lc.set_linewidth(
                2
            )  # set linewidth a little larger to see properly the colormap variation
            line = ax.add_collection3d(
                lc, zs=(Y[j, 1:] + Y[j, :-1]) / 2, zdir="y"
            )  # add line to axes

        fig.colorbar(
            lc
        )  # add colorbar, as the normalization is the same for all, it doesent matter which of the lc objects we use

    def unwrap_polar(self, in_array, maxRadius=700):
        out_array = cv2.warpPolar(
            src=in_array,
            dsize=in_array.shape,
            center=self.params.center,
            maxRadius=maxRadius,
            flags=cv2.WARP_POLAR_LINEAR,
        )
        out_array[out_array == 0.0] = np.nan
        out_array = out_array.T
        return out_array

    def unwrap_coords(self, maxRadius=700):
        unwrapped_radius = cv2.warpPolar(
            src=self.radius,
            dsize=self.radius.shape,
            center=self.params.center,
            maxRadius=maxRadius,
            flags=cv2.WARP_POLAR_LINEAR,
        )
        unwrapped_thetas = cv2.warpPolar(
            src=self.theta_array,
            dsize=self.theta_array.shape,
            center=self.params.center,
            maxRadius=maxRadius,
            flags=cv2.WARP_POLAR_LINEAR,
        )
        # unwrapped_radius[unwrapped_radius==0.0] = np.nan
        # unwrapped_thetas[unwrapped_thetas==0.0] = np.nan
        unwrapped_radius = unwrapped_radius.T
        unwrapped_thetas = unwrapped_thetas.T
        return unwrapped_radius, unwrapped_thetas

    # def angle_plot(self, r_vec_3d, theta_map, from_radial_theta, nan_r, show=False):
    #     # Mat the figure
    #     fig, axes = plt.subplots(3, 1, sharex="all", sharey="all")
    #     (ax1, ax2, ax3) = axes
    #     for ax in axes:
    #         ax.set_axis_off()
    #     self.autoLabelPanels(axes)
    #
    #     # Plot
    #     # ax1.set_title('Angle Confidence')
    #     ax1.imshow(r_vec_3d,    origin="lower", interpolation="None")
    #     ax1.imshow(nan_r, origin="lower", interpolation="None", cmap=cm_purp, alpha=0.9)
    #
    #     # ax2.set_title('Angle Map: {}'.format(self.fudge_level))
    #     ax2.imshow(np.zeros_like(theta_map),    origin="lower", cmap='gray', interpolation="None")
    #     ax2.imshow(theta_map,                   origin="lower", cmap='hsv', interpolation="None", vmin=0, vmax=180)
    #
    #     interpolated_from_radial_theta = interpolate_replace_nans(from_radial_theta, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft)
    #     # Interpolate the missing components
    #     masked = from_radial_theta #interpolated_from_radial_theta + 0
    #     radius, thetas = self.unwrap_coords()
    #     radius /= self.limb_radius_from_fit_shrunken
    #     masked[radius > 1.6] = np.nan
    #     masked[radius < 1.03] = np.nan
    #     ax3.set_title('Angle From Radial')
    #     ax3.imshow(np.zeros_like(theta_map), origin="lower", cmap='gray', interpolation="None")
    #     ax3.imshow(masked,        origin="lower", cmap='PuOr', interpolation="None")
    #
    #
    #     # Add the circles showing the rht window
    #     window = self.current_w_rht//2
    #     from matplotlib.patches import Circle
    #     radius = (window + 0)
    #     radius1 = window * self.window_factors[1]
    #     radius2 = window * self.window_factors[2]
    #     ax1.add_patch(Circle((730, 920), radius1, zorder=1000, fill=False, edgecolor='r',      lw=3))
    #     ax1.add_patch(Circle((730, 920), radius2, zorder=1000, fill=False, edgecolor='yellow', lw=3))
    #     ax1.add_patch(Circle((730, 920), radius , zorder=1000, fill=False, edgecolor="orange", lw=3))
    #
    #
    #     # Plot the different zoom levels
    #     fig.set_size_inches((20, 10))
    #     plt.rcParams['figure.dpi'] = 300
    #     if show:
    #         plt.show(block=True)
    #     # self.save3(fig)
    #     return fig

    def make_angle_arrays(self, in_array=None, do_abs=False):
        sparse = self.from_radial_theta if in_array is None else in_array
        interpolated = interpolate_replace_nans(
            sparse, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft
        )
        # interpolated = self.from_radial_theta
        masked_interp = interpolated + 0
        masked_sparse = sparse + 0
        radius = self.radius / self.params.header["R_SUN"] * 4
        masked_interp[radius > 1.6] = np.nan
        masked_interp[radius < 1.03] = np.nan
        masked_sparse[radius > 1.6] = np.nan
        masked_sparse[radius < 1.03] = np.nan
        angles = self.donut_the_sun(self.wrap_angles(masked_sparse, abs=do_abs))
        interp_angles = self.donut_the_sun(self.wrap_angles(masked_interp, abs=do_abs))
        return angles, interp_angles

    def angle_plot2_biplot(self, do_abs=False, show=False):
        """This is the 2 plots of the angle from radial, one interpolated and one not."""
        fig, axes = plt.subplots(1, 2, sharex="all", sharey="all")
        (ax2, ax3) = axes
        for ax in axes:
            ax.set_axis_off()

        angles, interp_angles = self.make_angle_arrays(do_abs=do_abs)

        ax2.imshow(
            np.ones_like(angles) * 0.1,
            origin="lower",
            cmap="gray",
            interpolation="None",
            vmin=0,
            vmax=1,
        )
        ax3.imshow(
            np.ones_like(angles) * 0.1,
            origin="lower",
            cmap="gray",
            interpolation="None",
            vmin=0,
            vmax=1,
        )

        ax2.imshow(
            angles, interpolation="none", cmap="plasma", origin="lower"
        )  # , vmin=-90, vmax=90)
        ax3.imshow(
            interp_angles, interpolation="none", cmap="plasma", origin="lower"
        )  # , vmin=-90, vmax=90)

        # ax2.imshow((self.theta_array * 180 / np.pi), zorder=-10, cmap='PuOr', origin='lower', vmin=-180, vmax=180)
        # ax3.imshow(self.shift_theta, zorder=-10, cmap='PuOr', origin='lower', vmin=-180, vmax=180)

        # ax2.imshow(self.donut_the_sun(angle_image), interpolation='none', cmap='binary_r', origin='lower', vmin=0, vmax=90)

        # interpolated = interpolate_replace_nans(self.from_radial_theta, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft)
        # interpolated = self.from_radial_theta
        # masked = interpolated + 0
        # radius = self.radius
        # masked = unwrapped_from_radial + 0
        # masked[radius > 1.6] = np.nan
        # masked[radius < 1.01] = np.nan
        # angle_image = self.wrap_angles(interpolated, abs=True)
        # ax3.imshow(self.donut_the_sun(angle_image), interpolation='none', cmap='PuOr', origin='lower', vmin=-90, vmax=90)

        # fi2, ax4 = plt.subplots()
        # ax4.imshow(np.ones_like(theta_map),    origin="lower", cmap='copper', interpolation="None", vmin=0, vmax=1)
        # ax4.imshow(self.donut_the_sun(angle_image), interpolation='none', cmap='binary_r', origin='lower', vmin=0, vmax=90)

        # ax1.set_ylim((700, 1000))
        # ax1.set_xlim((600, 900))

        # interpolated_from_radial_theta = interpolate_replace_nans(from_radial_theta, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft)
        # # Interpolate the missing components
        # masked = from_radial_theta #interpolated_from_radial_theta + 0
        # radius, thetas = self.unwrap_coords()
        # radius /= self.limb_radius_from_fit_shrunken
        # masked[radius > 1.6] = np.nan
        # masked[radius < 1.03] = np.nan

        #
        # ax1.imshow(np.zeros_like(theta_map),    origin="lower", cmap='gray', interpolation="None")
        # ax1.imshow(r_vec_3d,    origin="lower", interpolation="None")
        # ax1.imshow(self.donut_the_sun(nan_r), origin="lower", interpolation="None", cmap=cm_purp, alpha=0.9)
        #
        # ax2.set_title('Angle Map: {}'.format(self.fudge_level))
        # ax2.imshow(np.zeros_like(theta_map),    origin="lower", cmap='gray', interpolation="None")
        # ax2.imshow(self.donut_the_sun(theta_map), origin="lower", cmap='hsv', interpolation="None", vmin=0, vmax=180)
        # ax2.imshow(self.donut_the_sun(theta_map), origin="lower", cmap='hsv', interpolation="None", vmin=0, vmax=180)

        # # Add the circles showing the rht window
        # window = self.current_w_rht//2
        # from matplotlib.patches import Circle
        # radius = (window + 0)
        # radius1 = window * self.window_factors[1]
        # radius2 = window * self.window_factors[2]
        # ax2.add_patch(Circle((780, 940), radius1, zorder=1000, fill=False, edgecolor='r',      lw=3))
        # ax2.add_patch(Circle((780, 940), radius2, zorder=1000, fill=False, edgecolor='yellow', lw=3))
        # ax2.add_patch(Circle((780, 940), radius , zorder=1000, fill=False, edgecolor="orange", lw=3))

        self.autoLabelPanels(axes, loc=(0.025, 0.025))
        fig.set_size_inches((20, 10))
        plt.tight_layout()
        plt.rcParams["figure.dpi"] = 300
        # Plot the different zoom levels
        plt.xlim((0, 1024))
        plt.ylim((0, 1024))
        plt.tight_layout()
        plt.savefig(
            "{}\\4_angles_fromradial_{}.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22",
                "double" if not do_abs else "single",
            )
        )

        if show:
            plt.show(block=True)
        else:
            plt.close(fig)
        return fig

    def angle_plot2_triplot(
        self, r_vec_3d, theta_map, from_radial_theta, nan, show=False
    ):
        # This is the 3 plots only version
        nan_r = nan[1]
        # Just the Found Angle and the R_Vec plotted side by side
        fig, axes = plt.subplots(1, 3, sharex="all", sharey="all")
        (ax1, ax2, ax3) = axes
        for ax in axes:
            ax.set_axis_off()
        self.autoLabelPanels(axes, loc=(0.045, 0.05))

        # nan_r[nan_r<0.5]=np.nan

        # Plot
        # ax1.set_title('Angle Confidence')

        r_vec_3d[:, :, 0] = self.donut_the_sun(r_vec_3d[:, :, 0])
        r_vec_3d[:, :, 1] = self.donut_the_sun(r_vec_3d[:, :, 1])
        r_vec_3d[:, :, 2] = self.donut_the_sun(r_vec_3d[:, :, 2])

        ax1.imshow(
            np.zeros_like(theta_map), origin="lower", cmap="gray", interpolation="None"
        )
        ax1.imshow(r_vec_3d, origin="lower", interpolation="None")
        ax1.imshow(
            self.donut_the_sun(nan_r),
            origin="lower",
            interpolation="None",
            cmap=cm_purp,
            alpha=0.9,
        )

        # ax2.set_title('Angle Map: {}'.format(self.fudge_level))
        ax2.imshow(
            np.zeros_like(theta_map), origin="lower", cmap="gray", interpolation="None"
        )
        ax2.imshow(
            self.donut_the_sun(theta_map),
            origin="lower",
            cmap="hsv",
            interpolation="None",
            vmin=0,
            vmax=180,
        )
        # ax2.imshow(self.donut_the_sun(theta_map), origin="lower", cmap='hsv', interpolation="None", vmin=0, vmax=180)

        # interpolated = interpolate_replace_nans(self.from_radial_theta, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft)
        interpolated = self.from_radial_theta
        # masked = interpolated + 0
        # radius = self.radius
        # masked = unwrapped_from_radial + 0
        # masked[radius > 1.6] = np.nan
        # masked[radius < 1.01] = np.nan
        angle_image = self.wrap_angles(interpolated, abs=True)
        # ax3.imshow(self.donut_the_sun(angle_image), interpolation='none', cmap='PuOr', origin='lower', vmin=-90, vmax=90)
        ax3.imshow(
            np.ones_like(theta_map),
            origin="lower",
            cmap="copper",
            interpolation="None",
            vmin=0,
            vmax=1,
        )
        ax3.imshow(
            self.donut_the_sun(angle_image),
            interpolation="none",
            cmap="binary_r",
            origin="lower",
            vmin=0,
            vmax=90,
        )

        fi2, ax4 = plt.subplots()
        ax4.imshow(
            np.ones_like(theta_map),
            origin="lower",
            cmap="copper",
            interpolation="None",
            vmin=0,
            vmax=1,
        )
        ax4.imshow(
            self.donut_the_sun(angle_image),
            interpolation="none",
            cmap="binary_r",
            origin="lower",
            vmin=0,
            vmax=90,
        )

        # ax1.set_ylim((700, 1000))
        # ax1.set_xlim((600, 900))

        # interpolated_from_radial_theta = interpolate_replace_nans(from_radial_theta, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft)
        # # Interpolate the missing components
        # masked = from_radial_theta #interpolated_from_radial_theta + 0
        # radius, thetas = self.unwrap_coords()
        # radius /= self.limb_radius_from_fit_shrunken
        # masked[radius > 1.6] = np.nan
        # masked[radius < 1.03] = np.nan

        # Add the circles showing the rht window
        window = self.current_w_rht // 2
        from matplotlib.patches import Circle

        radius = window + 0
        radius1 = window * self.window_factors[1]
        radius2 = window * self.window_factors[2]
        ax1.add_patch(
            Circle((780, 940), radius1, zorder=1000, fill=False, edgecolor="r", lw=3)
        )
        ax1.add_patch(
            Circle(
                (780, 940), radius2, zorder=1000, fill=False, edgecolor="yellow", lw=3
            )
        )
        ax1.add_patch(
            Circle(
                (780, 940), radius, zorder=1000, fill=False, edgecolor="orange", lw=3
            )
        )

        # Plot the different zoom levels
        fig.set_size_inches((19, 10))
        plt.tight_layout()
        plt.rcParams["figure.dpi"] = 300
        self.autoLabelPanels(axes)
        plt.show(block=True)

        if show:
            plt.xlim((0, 1024))
            plt.ylim((0, 1024))
            plt.tight_layout()
            plt.savefig(
                "{}\\4_angles_allb.png".format(
                    r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
                )
            )
            plt.show(block=True)
            plt.close(fig)
        return fig

    def angle_plot2_rainbow(
        self, r_vec_3d, theta_map, from_radial_theta, nan, show=False
    ):
        """This is the 2 plots of the absolute angle angle, one on black, one on the shirft theta"""

        fig, axes = plt.subplots(1, 2, sharex="all", sharey="all")
        (ax1, ax2) = axes
        for ax in axes:
            ax.set_axis_off()
        self.autoLabelPanels(axes)

        # nan_r[nan_r<0.5]=np.nan

        # Plot
        # ax1.set_title('Angle Confidence')
        # plot_confidence=False
        # if plot_confidence:
        #     nan_r = nan[1]
        #
        #     r_vec_3d[:,:,0] = self.donut_the_sun(r_vec_3d[:,:,0])
        #     r_vec_3d[:,:,1] = self.donut_the_sun(r_vec_3d[:,:,1])
        #     r_vec_3d[:,:,2] = self.donut_the_sun(r_vec_3d[:,:,2])
        #
        #     ax1.imshow(np.zeros_like(theta_map),    origin="lower", cmap='gray', interpolation="None")
        #     ax1.imshow(r_vec_3d,    origin="lower", interpolation="None")
        #     ax1.imshow(self.donut_the_sun(nan_r), origin="lower", interpolation="None", cmap=cm_purp, alpha=0.9)
        #
        #     # Add the circles showing the rht window
        #     window = self.current_w_rht//2
        #     from matplotlib.patches import Circle
        #     radius = (window + 0)
        #     radius1 = window * self.window_factors[1]
        #     radius2 = window * self.window_factors[2]
        #     ax1.add_patch(Circle((780, 940), radius1, zorder=1000, fill=False, edgecolor='r',      lw=3))
        #     ax1.add_patch(Circle((780, 940), radius2, zorder=1000, fill=False, edgecolor='yellow', lw=3))
        #     ax1.add_patch(Circle((780, 940), radius , zorder=1000, fill=False, edgecolor="orange", lw=3))

        # ax1.imshow(self.shift_theta,                   origin="lower", cmap='hsv', interpolation="None", vmin=0, vmax=180)

        # ax2.set_title('Angle Map: {}'.format(self.fudge_level))

        # ax2.imshow(self.donut_the_sun(self.from_radial_theta), interpolation='none', cmap='PuOr', origin='lower', aspect='auto', vmin=-90, vmax=90)

        # ax1.set_ylim((700, 1000))
        # ax1.set_xlim((600, 900))

        # interpolated_from_radial_theta = interpolate_replace_nans(from_radial_theta, Gaussian2DKernel(x_stddev=2), convolve=convolve_fft)
        # # Interpolate the missing components
        # masked = from_radial_theta #interpolated_from_radial_theta + 0
        # radius, thetas = self.unwrap_coords()
        # radius /= self.limb_radius_from_fit_shrunken
        # masked[radius > 1.6] = np.nan
        # masked[radius < 1.03] = np.nan

        ax1.imshow(np.zeros_like(theta_map), cmap="gray")
        ax1.imshow(
            self.donut_the_sun(theta_map + 0),
            origin="lower",
            cmap="hsv",
            interpolation="None",
            vmin=0,
            vmax=180,
        )

        ax2.imshow(np.zeros_like(theta_map), cmap="gray")
        ax2.imshow(
            self.donut_the_sun(self.shift_theta + 0, radius=0.85),
            origin="lower",
            cmap="hsv",
            interpolation="None",
            vmin=0,
            vmax=180,
        )
        ax2.imshow(
            self.donut_the_sun(theta_map + 0),
            origin="lower",
            cmap="hsv",
            interpolation="None",
            vmin=0,
            vmax=180,
        )

        # Plot settings
        fig.set_size_inches((20, 10))
        plt.tight_layout()
        plt.rcParams["figure.dpi"] = 300
        self.autoLabelPanels(axes)
        if show:
            plt.show(block=True)
        else:
            plt.xlim((0, 1024))
            plt.ylim((0, 1024))
            plt.tight_layout()
            plt.savefig(
                "{}\\4_angles_allb_rainbow.png".format(
                    r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
                )
            )
            # plt.show(block=True)
            plt.close(fig)
        return fig

    def save3(self, fig):
        plt.xlim((0, 1024))
        plt.ylim((0, 1024))
        plt.tight_layout()
        plt.savefig(
            "{}\\7_angles_all.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
            )
        )

        plt.xlim((640, 800))
        plt.ylim((870, 1000))
        plt.tight_layout()
        plt.savefig(
            "{}\\8_angles.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
            )
        )

        plt.xlim((700, 765))
        plt.ylim((885, 950))
        plt.tight_layout()
        plt.savefig(
            "{}\\9_angles_zoom.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
            )
        )

        plt.close(fig)

    def big_angle_plot(
        self,
        reg,
        inv,
        edg,
        nan,
        theta_map,
        from_radial_theta,
        thresholded_image,
        inv_thresholded_image,
        sobel_8u,
    ):
        print(" ** Plot")

        # , (ax6, ax7, ax8)
        fig, axes = plt.subplots(3, 5, sharex=True, sharey=True)  # , figsize=(6, 15))

        (
            (ax0, ax1, ax2, ax3, ax4),
            (axw, axx, axy, axz, axa),
            (ax6, ax7, ax8, ax9, ax10),
        ) = axes
        # Row 1 : Binary Images
        ax0.imshow(thresholded_image, cmap="gray", interpolation="None")
        ax0.set_title("Thresholded Image")
        ax0.set_axis_off()

        ax1.imshow(inv_thresholded_image, cmap="gray", interpolation="None")
        ax1.set_title("Inverse( Thresholded frame )")
        ax1.set_axis_off()

        ax2.imshow(sobel_8u, cmap="gray", interpolation="None")
        ax2.set_title("Canny Edges")
        ax2.set_axis_off()

        ax3.imshow(self.nan_map, cmap="gray", interpolation="None")
        ax3.set_title("Remainders")
        ax3.set_axis_off()

        # ax4.imshow(np.ones_like(theta_map), cmap='gray')
        ax4.set_axis_off()
        # ax4.imshow(from_radial_theta, cmap='hsv', interpolation="None")
        # ax4.set_title('From Radial Theta')

        # Row 2: Weighted Theta Maps
        axw.imshow(
            reg[0], cmap="hsv", interpolation="None", vmin=0, vmax=180
        )  # hsv is cyclic, like angles
        # axw.set_title('weighted_theta_reg')
        axw.set_axis_off()

        axx.imshow(
            inv[0], cmap="hsv", interpolation="None", vmin=0, vmax=180
        )  # hsv is cyclic, like angles
        # axx.set_title('weighted_theta_inv')
        axx.set_axis_off()

        axy.imshow(
            edg[0], cmap="hsv", interpolation="None", vmin=0, vmax=180
        )  # hsv is cyclic, like angles
        # axy.set_title('weighted_theta_edg')
        axy.set_axis_off()

        axz.imshow(
            nan[0], cmap="hsv", interpolation="None", vmin=0, vmax=180
        )  # hsv is cyclic, like angles
        # axz.set_title('theta_nan')
        axz.set_axis_off()

        axa.imshow(np.ones_like(theta_map), cmap="gray")
        axa.imshow(theta_map, cmap="hsv", interpolation="None", vmin=0, vmax=180)
        # axa.set_title('theta_all')
        axa.set_axis_off()

        # Row 3: R_bar confidence
        # ax6.imshow(reg[1], cmap='brg', interpolation="None", vmin=thresh, vmax=1.)
        ax6.imshow(
            reg[1],
            origin="lower",
            interpolation="None",
            cmap="RdYlGn",
            vmin=self.thresh,
            vmax=1.0,
        )
        # ax6.set_title('r_bar_reg')
        ax6.set_axis_off()

        ax7.imshow(
            inv[1],
            origin="lower",
            interpolation="None",
            cmap="RdYlGn",
            vmin=self.thresh,
            vmax=1.0,
        )
        # ax7.set_title('r_bar_inv')
        ax7.set_axis_off()

        ax8.imshow(
            edg[1],
            origin="lower",
            interpolation="None",
            cmap="RdYlGn",
            vmin=self.thresh,
            vmax=1.0,
        )
        # ax8.set_title('r_bar_edg')
        ax8.set_axis_off()

        ax9.imshow(
            nan[1],
            origin="lower",
            interpolation="None",
            cmap="RdYlGn",
            vmin=self.thresh,
            vmax=1.0,
        )
        # ax9.set_title('r_bar_nan')
        ax9.set_axis_off()

        # Add the circles showing the rht window
        window = self.current_w_rht // 2 + 0
        from matplotlib.patches import Rectangle, Circle

        for ii, axbox in enumerate(axes):
            for jj, ax in enumerate(axbox):
                radius = window + 0
                if jj == 3:
                    radius1 = window * self.window_factors[1]
                    radius2 = window * self.window_factors[2]
                    ax.add_patch(
                        Circle(
                            (730, 920),
                            radius1,
                            zorder=1000,
                            fill=False,
                            edgecolor="r",
                            lw=3,
                        )
                    )
                    ax.add_patch(
                        Circle(
                            (730, 920),
                            radius2,
                            zorder=1000,
                            fill=False,
                            edgecolor="yellow",
                            lw=3,
                        )
                    )
                if jj == 4 and ii == 0:
                    continue
                ax.add_patch(
                    Circle(
                        (730, 920),
                        radius,
                        zorder=1000,
                        fill=False,
                        edgecolor="orange",
                        lw=3,
                    )
                )

        # Background Thresh Images
        axw.imshow(
            thresholded_image,
            origin="lower",
            interpolation="None",
            cmap="gray",
            alpha=0.2,
        )
        ax6.imshow(
            thresholded_image,
            origin="lower",
            interpolation="None",
            cmap="gray",
            alpha=0.2,
        )
        axx.imshow(
            inv_thresholded_image,
            origin="lower",
            interpolation="None",
            cmap="gray",
            alpha=0.2,
        )
        ax7.imshow(
            inv_thresholded_image,
            origin="lower",
            interpolation="None",
            cmap="gray",
            alpha=0.2,
        )
        axy.imshow(
            sobel_8u, origin="lower", interpolation="None", cmap="gray", alpha=0.2
        )
        ax8.imshow(
            sobel_8u, origin="lower", interpolation="None", cmap="gray", alpha=0.2
        )
        axz.imshow(
            self.nan_map, origin="lower", interpolation="None", cmap="gray", alpha=0.2
        )
        ax9.imshow(
            self.nan_map, origin="lower", interpolation="None", cmap="gray", alpha=0.2
        )

        r_vec_3d = np.dstack((reg[1], inv[1], edg[1]))
        ax10.imshow(r_vec_3d, origin="lower", interpolation="None")
        ax10.imshow(
            nan[1], origin="lower", interpolation="None", cmap=cm_purp, alpha=0.9
        )
        ax10.patch.set(hatch="x", edgecolor="lightgrey")
        # ax10.set_title('Mean Resultant Length')
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
        self.autoLabelPanels(axes)

        fig.set_size_inches((20, 10))
        # fig.suptitle("Building the theta map with the RHT")

        #
        # plt.xlim((0,1024))
        # plt.ylim((0,1024))
        # plt.tight_layout()
        # plt.savefig("{}\\1_angles_all.png".format(r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"))
        #
        plt.xlim((640, 800))
        plt.ylim((870, 1000))
        plt.tight_layout()

        plt.savefig(
            "{}\\2_angles.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
            )
        )

        plt.xlim((700, 765))
        plt.ylim((885, 950))
        plt.tight_layout()
        plt.savefig(
            "{}\\3_angles_zoom.png".format(
                r"C:\Users\chgi7364\Dropbox\All School\CU\Steve Research\Research Notes\Weekly Meetings\2022\Meeting 10-4-22"
            )
        )

        plt.close(fig)
        # plt.show(block=True)

        # plt.close(fig)
        # plt.imshow(self.theta_array*180/np.pi); plt.show(block=True)

    #!python numbers=enable
    @staticmethod
    def sgolay2d(z, window_size, order, derivative=None):
        """ """
        # number of terms in the polynomial expression
        n_terms = (order + 1) * (order + 2) / 2.0

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")

        if window_size**2 < n_terms:
            raise ValueError("order is too high for the window size")

        half_size = window_size // 2

        # exponents of the polynomial.
        # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
        # this line gives a list of two item tuple. Each tuple contains
        # the exponents of the k-th term. First element of tuple is for x
        # second element for y.
        # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
        exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

        # coordinates of points
        ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
        dx = np.repeat(ind, window_size)
        dy = np.tile(ind, [window_size, 1]).reshape(
            window_size**2,
        )

        # build matrix of system of equation
        A = np.empty((window_size**2, len(exps)))
        for i, exp in enumerate(exps):
            A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

        # pad in_array array with appropriate values at the four borders
        new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
        Z = np.zeros((new_shape))
        # top band
        band = z[0, :]
        Z[:half_size, half_size:-half_size] = band - np.abs(
            np.flipud(z[1 : half_size + 1, :]) - band
        )
        # bottom band
        band = z[-1, :]
        Z[-half_size:, half_size:-half_size] = band + np.abs(
            np.flipud(z[-half_size - 1 : -1, :]) - band
        )
        # left band
        band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
        Z[half_size:-half_size, :half_size] = band - np.abs(
            np.fliplr(z[:, 1 : half_size + 1]) - band
        )
        # right band
        band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
        Z[half_size:-half_size, -half_size:] = band + np.abs(
            np.fliplr(z[:, -half_size - 1 : -1]) - band
        )
        # central band
        Z[half_size:-half_size, half_size:-half_size] = z

        # top left corner
        band = z[0, 0]
        Z[:half_size, :half_size] = band - np.abs(
            np.flipud(np.fliplr(z[1 : half_size + 1, 1 : half_size + 1])) - band
        )
        # bottom right corner
        band = z[-1, -1]
        Z[-half_size:, -half_size:] = band + np.abs(
            np.flipud(np.fliplr(z[-half_size - 1 : -1, -half_size - 1 : -1])) - band
        )

        # top right corner
        band = Z[half_size, -half_size:]
        Z[:half_size, -half_size:] = band - np.abs(
            np.flipud(Z[half_size + 1 : 2 * half_size + 1, -half_size:]) - band
        )
        # bottom left corner
        band = Z[-half_size:, half_size].reshape(-1, 1)
        Z[-half_size:, :half_size] = band - np.abs(
            np.fliplr(Z[-half_size:, half_size + 1 : 2 * half_size + 1]) - band
        )
        import scipy
        import scipy.signal

        # solve system and convolve
        if derivative == None:
            m = np.linalg.pinv(A)[0].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, m, mode="valid")
        elif derivative == "col":
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -c, mode="valid")
        elif derivative == "row":
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return scipy.signal.fftconvolve(Z, -r, mode="valid")
        elif derivative == "both":
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return scipy.signal.fftconvolve(
                Z, -r, mode="valid"
            ), scipy.signal.fftconvolve(Z, -c, mode="valid")

    def find_h_xy(self, H_XY=None, fudge=0.275, thresh=True):
        """find the reduced hxy matrix, which is set to 0 below a thresh"""
        H_XY = H_XY if H_XY is not None else self.rht_cube
        h_xy = H_XY + 0
        # self.fudge_level = fudge
        if thresh:
            threshold = np.max(H_XY) - fudge
            h_xy[H_XY < threshold] = 0
        return h_xy

    def find_weighted_sums(self, h_xy=None):
        # theta_map = np.nanmean(np.dstack((weighted_theta_reg, weighted_theta_inv)), axis=2)
        # h_xy = h_xy if h_xy is not None else self.h_xy

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
        theta_bar *= 180 / np.pi
        tb = theta_bar + 0
        theta_bar[tb > 180] -= 180
        theta_bar[tb < 0] += 180

        # Find RBar
        r_bar = np.sqrt(c_bar**2 + s_bar**2)

        stdev_circ = np.sqrt(-2 * np.log(r_bar))

        return theta_bar, r_bar, stdev_circ

    def find_RHT_error(self, H_XY=None):
        h_xy = self.find_h_xy(H_XY=H_XY)
        theta_bar, r_bar, stdev_circ = self.find_weighted_sums(h_xy)
        return theta_bar, r_bar

    def prep_inputs(self, shrink=False, prnt=True):
        # print("   * Conditioning Inputs...")
        if shrink:
            self.resize_image(prnt=prnt)
        self.init_image_frames()
        # if "rhe" in self.in_name:   ######################################################################
        #     self.params.modified_image = norm_stretch(self.params.modified_image)
        mdi = self.mask_out_sun(self.params.modified_image)
        self.params.modified_image = self.vignette(mdi)

    def segmentation_jing11(self, use_image=None, doplot=False):
        """Use a series of filters to segment the frame into a binary map that
        actually matches the fine structure in the corona.
        """
        # print("   * Segmenting Image...")
        use_image = use_image or self.params.modified_image
        use_image_int8 = self.smsh_img_255(use_image)

        #########################
        # Highpass and Lowpass
        if self.smol:
            kern, sigma, blur_f = 21, 7, 1.0
        else:
            kern, sigma, blur_f = 41, 18, 1.0
        highpass_img, lowpass_img = self.highpass_filt(
            use_image_int8, kern, sigma, blur_f
        )

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
            fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(
                2, 3, sharex=True, sharey=True
            )  # , figsize=(6, 15))
            ax0.imshow(use_image_int8, cmap="gray", interpolation="None")
            ax0.set_title("Original")
            ax0.set_axis_off()

            ax1.imshow(lowpass_img, cmap="gray", interpolation="None")
            ax1.set_title("Gaussian Blur, sigma = {}, bf={}".format(sigma, blur_f))
            ax1.set_axis_off()

            ax2.imshow(
                self.smsh_img_255(highpass_img), cmap="gray", interpolation="None"
            )  # hsv is cyclic, like angles
            ax2.set_title("High Pass")
            ax2.set_axis_off()

            ax3.imshow(
                self.smsh_img_255(bandpass_img), cmap="gray", interpolation="None"
            )  # hsv is cyclic, like angles
            ax3.set_title("Bandpass, s1 = {}, s2 = {}".format(sigma, sigma2))
            ax3.set_axis_off()

            # ax4.imshow(canny_image, cmap='gray') # hsv is cyclic, like angles
            # ax6.set_title('Canny Edges: t1={}, t2= {}'.format(t1, t2))
            # ax6.set_axis_off()

            ax4.imshow(
                thresh_img, cmap="gray", interpolation="None"
            )  # hsv is cyclic, like angles
            ax4.set_title("Thresholded, thresh= {:0.8}".format(thresh))
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
        # print("   * Threshold")
        # Threshold the Image
        thresh = mu * np.nanmedian(image)
        th, thresh_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        return thresh_img, th

    def adaptive_threshold_the_image(self, image):
        # print("   * Threshold")
        # Threshold the Image
        th3 = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0
        )
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
        W_RHT = self.W_RHT_1024 * ff * w_factor

        W_RHT = self.ensure_odd(int(W_RHT))
        self.current_w_rht = W_RHT + 0
        # print(W_RHT)
        # W_RHT = 55.

        #############################################################
        # Set the amount of radial smear in the algorithm
        # Boe 2020 says 0.02 to 0.4 Rs
        smear = 2  # np.round(0.02 * r_sun_pixels).astype(np.uint8)
        # smear = 11.

        #############################################################
        frac = self.count_fraction_threshold

        ## Run RHT Algorithm
        print("\n V V Starting RHT V V")
        # source = self.fits_path
        (H_XY, self.theta) = rht.main(
            source=self.fits_path,
            data=binary_image,
            conv=True,
            outroot=outroot_selected,
            wlen=W_RHT,
            smr=smear,
            frac=frac,
        )

        theta_bar, r_bar = self.find_RHT_error(H_XY)

        print(" ^  ^  Success!  ^  ^ ")
        return H_XY, self.theta, theta_bar, r_bar

    ##############################
    # Plotting Angles
    def plot_angles(self, outroot=None):
        outroot_selected = outroot or self.outroot
        print("\n V Plotting angle images")
        # The first dimension of this H_XY is theta
        self.angle_dir = os.path.join(
            outroot_selected, "angles_{}".format(self.in_name)
        )
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

        plt.xlim((x1, x2))
        plt.ylim((y1, y2))
        self.maximizePlot()
        plt.subplots_adjust(
            top=0.934, bottom=0.015, left=0.008, right=0.992, hspace=0.081, wspace=0.0
        )

    ##############################
    # Helper Functions
    def donut_the_sun(self, input_img, radius=None, plug=None):
        return self.mask_out_sun(self.vignette(input_img), radius=radius, plug=plug)

    def highpass_filt(self, image, kern_in=41, sigma_in=18, blur_f=1.0):
        # print("   * Highpass")
        gaussian_blur = self.lowpass_filt(image, kern_in, sigma_in)
        highpass_img = image - blur_f * gaussian_blur
        return highpass_img, gaussian_blur

    def lowpass_filt(self, image, kern_in=11, sigma_in=1.0):
        # print("   * Lowpass")
        # LowPass the Input
        ff = self.shrink_F
        kern2 = kern_in // ff
        sigma2 = sigma_in / ff
        while kern2 < 2 * sigma2:
            kern2 += 1
        kern2 = self.ensure_odd(kern2)

        gaussian_blur = cv2.GaussianBlur(
            src=image, ksize=(kern2, kern2), sigmaX=sigma2, sigmaY=sigma2
        )
        return gaussian_blur

    def smsh_img_255(self, use_img):
        # print(" * Smooshing to 255...")
        thresh = np.nanpercentile(use_img, [0.5, 99.5])
        normed = self.norm_formula(use_img, *thresh)
        normed[normed >= 1] = 1.0
        normed[normed <= 0] = 0.0
        smooshed = np.round(normed * 255).astype(np.uint8)
        return smooshed

    ##############################
    # Depricated
    def compute_scharr_image_gradient(self, use_image):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
        from scipy import signal

        scharr = np.array(
            [
                [-3 - 3j, 0 - 10j, +3 - 3j],
                # Compute the gradient of an frame by 2D convolution with a complex Scharr operator. (Horizontal operator is real, vertical is imaginary.) Use symmetric boundary condition to avoid creating edges at the frame boundaries.
                [-10 + 0j, 0 + 0j, +10 + 0j],
                [-3 + 3j, 0 + 10j, +3 + 3j],
            ]
        )  # Gx + j*Gy

        grad = signal.convolve2d(use_image, scharr, boundary="symm", mode="same")
        magnitude = np.absolute(grad)
        direction = np.angle(grad)
        if True:
            fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(1, 3)  # , figsize=(6, 15))
            ax_orig.imshow(use_image, cmap="gray")
            ax_orig.set_title("Original")
            ax_orig.set_axis_off()
            ax_mag.imshow(np.log10(magnitude), cmap="gray")
            ax_mag.set_title("Gradient magnitude")
            ax_mag.set_axis_off()
            ax_ang.imshow(direction, cmap="hsv")  # hsv is cyclic, like angles
            ax_ang.set_title("Gradient orientation")
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

        from astropy.convolution import (
            convolve,
            convolve_fft,
            Box2DKernel,
            CustomKernel,
        )

        lp_wind = 9  # Options are 3, 5, 7, 9, 11 and different for each frame
        hp_wind = 9

        low_pass_window = Box2DKernel(lp_wind) * (1 / (lp_wind**2))
        # high_pass_window = Box2DKernel(hp_wind)*(-1/(hp_wind**2))
        # high_pass_window[hp_wind//2, hp_wind//2] = 1 - 1/(hp_wind**2)

        # low_pass_window  = np.ones((lp_wind, lp_wind))* (1/(lp_wind**2))
        high_pass_window = np.ones((hp_wind, hp_wind)) * (-1 / (hp_wind**2))
        high_pass_window[hp_wind // 2, hp_wind // 2] = 2 - 1 / (hp_wind**2)

        kernal_1 = low_pass_window
        kernal_2 = high_pass_window

        # lowpass_img = signal.convolve2d(use_image, kernal_1, boundary='symm', mode='same')
        lowpass_img = convolve(use_image, kernal_1)
        # highpass_img = signal.convolve2d(use_image, kernal_2, boundary='symm', mode='same')
        highpass_img = convolve(use_image, kernal_2, nan_treatment="fill")
        grad = lowpass_img - highpass_img

        unsharp = use_image - lowpass_img

        print("Smooshing...")
        smooshed = self.smsh_img_255(lowpass_img)

        print("Canning...")
        import cv2

        output_image = cv2.Canny(smooshed, 20, 60)

        print("Plotting...")

        # input_image = Image.fromarray(use_image)
        # frame = cv2.cvtColor(use_image, cv2.COLOR_BGR2GRAY )
        a = 1
        # output_image= Image.fromarray(use_image)
        # img = cv2.imdecode(use_image, flags=cv2.IMREAD_GRAYSCALE)
        # edged_image = cv2.imdecode(np.zeros_like(use_image), flags=cv2.IMREAD_GRAYSCALE)
        #

        # Plot
        if True:
            fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(
                2, 3, figsize=(16, 12), sharex="all", sharey="all"
            )
            ax0.imshow(np.absolute(use_image), cmap="gray")
            ax0.set_title("Original")
            ax0.set_axis_off()

            ax1.imshow(np.absolute(lowpass_img), cmap="gray")
            ax1.set_title("Lowpass")
            ax1.set_axis_off()
            # import matplotlib.markers as markers
            # marker = markers.MarkerStyle(marker='s', fillstyle='none')

            location = (2200, 3900) // self.shrink_F

            from matplotlib.patches import Rectangle, Circle

            ax1.add_patch(
                Rectangle(
                    location, lp_wind, lp_wind, zorder=1000, fill=False, edgecolor="b"
                )
            )
            ax2.add_patch(
                Rectangle(
                    location, hp_wind, hp_wind, zorder=1000, fill=False, edgecolor="r"
                )
            )

            ax3.add_patch(
                Rectangle(
                    location, lp_wind, lp_wind, zorder=1000, fill=False, edgecolor="b"
                )
            )
            ax3.add_patch(
                Rectangle(
                    location, hp_wind, hp_wind, zorder=1000, fill=False, edgecolor="r"
                )
            )

            ax2.imshow(
                np.log10(np.absolute(highpass_img)), cmap="gray"
            )  # hsv is cyclic, like angles
            ax2.set_title("Highpass")
            ax2.set_axis_off()

            ax3.imshow(np.absolute(grad), cmap="gray")  # hsv is cyclic, like angles
            ax3.set_title("Low - High")
            ax3.set_axis_off()

            ax4.imshow(np.absolute(smooshed), cmap="gray")
            ax4.set_title("uint8")
            ax4.set_axis_off()

            ax5.imshow(np.absolute(output_image), cmap="gray")
            ax5.set_title("Canny Edged")
            ax5.set_axis_off()

            self.adjust_rht_plot(fig, shrink=self.shrink_F)
            plt.show(block=True)

    def DEP_make_RHT_binary_map(self, use_image):
        print("Binary Filtering...")

        brightness_map_04 = use_image >= 0.4
        brightness_map_05 = use_image >= 0.5
        brightness_map_06 = use_image >= 0.6

        if True:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
                2, 2, figsize=(16, 12), sharex="all", sharey="all"
            )
            ax0.imshow(use_image, cmap="gray")
            ax0.set_title("Original")
            ax0.set_axis_off()

            ax1.imshow((brightness_map_04), cmap="gray")
            ax1.set_title("Binary Map 0.4")
            ax1.set_axis_off()

            ax2.imshow((brightness_map_05), cmap="gray")  # hsv is cyclic, like angles
            ax2.set_title("Binary Map 0.5")
            ax2.set_axis_off()

            ax3.imshow((brightness_map_06), cmap="gray")  # hsv is cyclic, like angles
            ax3.set_title("Binary Map 0.6")
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

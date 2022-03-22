from copy import copy
from os import makedirs
from os.path import join, dirname, basename
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import stats
from processor.Processor import Processor
import warnings
import shutil
import os
from astropy import units as u
from science.color_tables import aia_color_table
from time import strftime
import datetime

# from processor.SRNProcessor_Legacy import Legacy_SRN_Kernal as Legacy_Modify

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


class SRNProcessor(Processor):
    """This is the primary code used in the RadialFiltProcessor"""
    
    name = "Default"
    filt_name = '  SRN Radial Base Class'
    description = "Create and Apply the Radial SRN Curves"
    out_name = "SRN"
    
    # do_png = False
    renew_mask = True
    show_plots = True
    can_initialize = True
    image_data = None
    outer_min = None
    inner_min = None
    inner_max = None
    outer_max = None
    avg_min = None
    avg_max = None
    
    radius = None
    rad_flat = None
    bin_rez = None
    found_limb_radius = None  # 1600
    
    rendered_abss = None
    norm_avg_max = None
    norm_avg_min = None
    
    multiple_minimum_curves = []
    multiple_maximum_curves = []
    rendered_min_box = []
    rendered_max_box = []
    radBins_all = []
    
    def __init__(self, fits_path=None, in_name=None, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        """Initialize the main class"""
        super().__init__(params=params, quick=quick, rp=rp)
        # Parse Inputs
        self.in_name = in_name or self.params.master_frame_list_newest
        
        self.RN = None
        self.binfactor = 1
        self.this_index = 0
        self.norm_curve_max_top_name = None
        self.norm_curve_min_top_name = None
        self.make_curves_latch = True  # This Recomputes the curves once
        self.flatten_down = None
        self.flatten_up = None
        self.hCut = None
        self.fit_limb_radius = None
        self.skip_points = None
        self.norm_curve_min_name = None
        self.norm_curve_max_name = None
        self.savgol_filtered_absol_maximum = None
        self.savgol_filtered_absol_minimum = None
        self.abs_min = None
        self.abs_max = None
        self.first = True
        self.lCut = None
        self.smooth_outer_maximum = None
        self.smooth_inner_maximum = None
        self.smooth_inner_minimum = None
        self.smooth_outer_minimum = None
        self.savgol_filtered_frame_maximum = None
        self.savgol_filtered_frame_minimum = None
        self.savgol_filtered_outer_maximum = None
        self.savgol_filtered_inner_maximum = None
        self.savgol_filtered_inner_minimum = None
        self.savgol_filtered_outer_minimum = None
        
        self.there_is_cached_data = False
        self.output_abscissa = None
        self.floor = 0.01
        self.n_keyframes = 0
        self.firstIndex = 0
        self.lastIndex = 0
        
        self.abs_max_scalar = None
        self.abs_min_scalar = None
        self.curve_out_array = None
        self.do_save = None
        self.scalar_in_curve = None
        self.rastered_outer_min = None
        self.rastered_inner_min = None
        self.rastered_inner_max = None
        self.rastered_outer_max = None
        self.scalar_out_curve = None
        self.can_use_keyframes = True
        self.outputs_initialized = False
        self.dont_ignore = True
        self.cut_pixels = None
        self.s_radius = 400
        
        self.fits_path = fits_path
        self.show = show
        self.verb = verb
        self.do_orig = orig
        
        self.mirror_mask = None
        self.grid_mask = None
        self.t_factor = None
        self.tRadius = None
        # self.center = None
        # self.outer_max = None
        # self.outer_min = None
        # self.changed_flat = None
        # self.changed_flat = None
        # self.rez = None
        # self.raw_image = None
        # self.modified_image = None
        self.binInds = None
        self.bin_rez = None
        self.radBins = None
        self.radBins_xy = None
        self.radBins_ind = None
        self.binMax = None
        self.binMin = None
        self.binAbsMax = None
        self.binAbsMin = None
        self.binAbss = None
        self.norm_curve_min = None
        self.norm_curve_max = None
        
        self.vignette_mask = None
        self.frame_maximum = None
        self.frame_minimum = None
        self.frame_abs_max = None
        self.frame_abs_min = None
        self.frame_abss = None
        self.norm_avg_max = None
        self.norm_avg_min = None
    
    # Not Implemented
    def setup(self):
        """Do prep work once before the main algorithm"""
        pass
    
    def do_work(self):
        """Do whatever you want to each image_path in the directory"""
        raise NotImplementedError
    
    def cleanup(self):
        """Runs once after all the images have been modified with do_work"""
        raise NotImplementedError
    
    ###################
    ##   Main Calls  ##
    ###################
    
    def do_fits_function(self, fits_path=None, in_name=None, image=True):
        """Calls the do_work function on a single fits path if indicated"""
        if self.load_fits_image(fits_path, in_name=in_name):
            if (not self.use_keyframes) or (self.fits_path in self.keyframes):
                return self.do_work()  # Do the work on the fits files
        return None
    
    def do_img_function(self):
        """Calls the do_work function on a single fits path if indicated"""
        return self.do_work()  # Do the work on the fits files
    
    ###################
    ## Top-Level ##
    ###################
    
    def image_learn(self):
        """Analyze the input image_path to help make normalization curves"""
        if not self.skip_bad_frame():
            self.init_for_learn()
            self.coronaLearn()
            self.add_to_keyframes()  # Update the running curves
    
    def image_modify(self):
        """Perform the actual normalization on the input array"""
        if not self.skip_bad_frame():
            self.init_for_modify()
            self.coronaNorm()  # Use curves to rescale the in_object
            self.prep_output()
    
    ###################
    ## Keyframes ##
    ###################
    
    def add_to_keyframes(self):
        """Records the current analysis as one of the radial samples"""
        self.update_keyframe_counters()
        if not self.outputs_initialized or self.params.Force_init:
            self.init_running_curves()
        else:
            self.update_running_curves()
        # self.make_save_smoothed_curves()
    
    def coronaLearn(self):
        """Perform the actual analysis"""
        self.bin_radially()  # Create a cloud of intesity values for each radial bin
        self.radial_statistics()  # Find mean and percentiles vs height
    
    def update_keyframe_counters(self, n=1):
        """Keep track of how many items have been added to keyframes"""
        self.n_keyframes += n
        self.lastIndex += n
        self.skipped -= n
        self.skipped = max(self.skipped, 0)
    
    ######################################
    ## Initializing and Converting ##
    ######################################
    def skip_bad_frame(self):
        # TODO Implement Skip logic here
        return False
    
    def init_for_learn(self):
        # self.init_images()
        self.init_radius_array()
        self.init_frame_curves()
        self.init_statistics()
    
    def init_for_modify(self):
        self.init_radius_array()
    
    #     def init_images(self, modified_image=None):
    #         """Get all the variables ready for the normalization"""
    #         dprint("\ninit_images")
    #         if modified_image is not None:
    #             self.modified_image = modified_image
    #         self.rez = self.modified_image.shape[0]
    #         self.modified_image = self.modified_image.astype('float32')
    #         self.modified_image[self.modified_image == 0] = np.nan
    #
    #         if self.raw_image is None:
    #             self.raw_image = self.modified_image + 0
    #
    #         self.raw_flat = self.raw_image.flatten()
    #         self.changed_flat = self.modified_image.flatten()
    
    # def init_radius_array(self, vignette_radius=1.19, s_radius=400, t_factor=1.28, force=False):
    #     """Build an r-coordinate array of shape(in_object)"""
    #     if self.params.rez is None:
    #         self.params.rez = self.params.modified_image.shape[0]
    #     if self.params.center is None:
    #         self.params.center = [self.params.rez / 2, self.params.rez / 2]
    #
    #     self.output_abscissa = np.arange(self.params.rez)
    #     self.find_limb_radius()
    #
    #     if self.radius is None or force or self.params.modified_image.shape[0] != self.params.rez:
    #         dprint("init_radius_array")
    #
    #         xx, yy = np.meshgrid(np.arange(self.params.rez), np.arange(self.params.rez))
    #         xc, yc = xx - self.params.center[0], yy - self.params.center[1]
    #
    #         # self.xxyy =
    #         self.radius = np.sqrt(xc * xc + yc * yc)
    #         self.rad_flat = self.radius.flatten()
    #
    #         self.binfactor = binfactor = 2
    #         self.binInds = np.asarray(binfactor * np.floor(self.rad_flat // binfactor), dtype=np.int32)
    #         # self.make_annular_rings()
    #         # self.binInds = np.digitize(self.rad_flat, self.RN)
    #
    #         self.binXX = xx.flatten()
    #         self.binYY = yy.flatten()
    #         self.binII = np.arange(len(self.rad_flat))
    #         self.vcut = int(vignette_radius * self.params.rez // 2)
    #         self.vrad = self.n2r(self.vcut)
    #         self.vignette_mask = np.asarray(self.radius > self.vcut, dtype=bool)
    #         self.s_radius = s_radius
    #         self.tRadius = self.s_radius * t_factor
    #         del self.radius
    
    def init_frame_curves(self):
        """These are the main frame_level curves"""
        dprint("init_frame_curves")
        self.frame_maximum = np.empty(self.params.rez)
        self.frame_minimum = np.empty(self.params.rez)
        self.frame_abs_max = np.empty(self.params.rez)
        self.frame_abs_min = np.empty(self.params.rez)
        self.frame_abss = np.empty(self.params.rez)
        
        self.frame_maximum.fill(np.nan)
        self.frame_minimum.fill(np.nan)
        self.frame_abs_max.fill(np.nan)
        self.frame_abs_min.fill(np.nan)
        self.frame_abss.fill(np.nan)
    
    def init_statistics(self):
        """Initialize the statistical arrays"""
        dprint("init_statistics")
        
        self.bin_rez = np.max(self.binInds) + 10
        self.radBins = [[] for x in np.arange(self.bin_rez)]
        self.radBins_xy = [[] for x in np.arange(self.bin_rez)]
        self.radBins_ind = [[] for x in np.arange(self.bin_rez)]
        
        self.binMax = np.empty(self.bin_rez)
        self.binMin = np.empty(self.bin_rez)
        self.binAbsMax = np.empty(self.bin_rez)
        self.binAbsMin = np.empty(self.bin_rez)
        self.binAbss = np.arange(self.bin_rez)
        
        self.binMax.fill(np.nan)
        self.binMin.fill(np.nan)
        self.binAbsMax.fill(np.nan)
        self.binAbsMin.fill(np.nan)
    
    def init_running_curves(self):
        """Initialize the curves"""
        need_to_run = (self.frame_minimum is not None and np.sum(np.isfinite(self.frame_minimum)) > 0)
        if need_to_run or self.params.Force_init:
            dprint("init_running_curves")
            
            # The four running extrema
            self.outer_max = self.frame_maximum + 0
            self.inner_max = self.frame_maximum + 0
            self.inner_min = self.frame_minimum + 0
            self.outer_min = self.frame_minimum + 0
            
            # The absolute extrema curves
            self.abs_max = self.frame_abs_max + 0
            self.abs_min = self.frame_abs_min + 0
            
            # Average Curves
            self.avg_min = self.frame_minimum + 0
            self.avg_max = self.frame_maximum + 0
            
            # Scalars
            self.abs_min_scalar = max(np.nanmin(self.outer_min), -10)
            self.abs_max_scalar = np.nanmax(self.outer_max)
            
            self.outputs_initialized = True and self.can_initialize
        return True
    
    def update_running_curves(self):
        """Update the Curves"""
        # # print("\rupdated_running_curves")
        # plt.plot(self.frame_abss, self.abs_max, c='r', label="abs_max", ls=":",)
        # plt.plot(self.frame_abss, self.abs_min, c='k', label="abs_min", ls=":",)
        
        # The four running extrema
        self.outer_max = np.fmax(self.outer_max, self.frame_maximum)
        self.inner_max = np.fmin(self.inner_max, self.frame_maximum)
        
        self.inner_min = np.fmax(self.inner_min, self.frame_minimum)
        self.outer_min = np.fmin(self.outer_min, self.frame_minimum)
        
        # The absolute extrema curves
        self.abs_max = np.fmax(self.abs_max, self.frame_abs_max)
        self.abs_min = np.fmin(self.abs_min, self.frame_abs_min)
        
        # Average Curves
        self.avg_max = self.avg_max + self.frame_maximum
        self.avg_min = self.avg_min + self.frame_minimum
        self.norm_avg_max = self.avg_max / self.n_keyframes
        self.norm_avg_min = self.avg_min / self.n_keyframes
        
        # Scalars
        self.abs_max_scalar = np.fmax(self.abs_max_scalar, np.max(self.outer_max))
        self.abs_min_scalar = np.fmin(self.abs_min_scalar, np.min(self.outer_min))
        
        # plt.plot(self.frame_abss, self.abs_max, c='r')
        # plt.plot(self.frame_abss, self.abs_min, c='k')
        # plt.yscale('symlog')
        # plt.legend()
        # plt.show(block=True)
    
    # TODO Make this sample better, linear isn't really appropriate because its a circle
    # 1+1
    
    def percentilize(self):
        """Another way of looking at the data"""
        self.do_percentile_norm()
        # self.do_percentile_plot()
    
    def do_percentile_norm(self):
        # Make Percentile Image
        from scipy import stats
        # plt.show()
        # plt.figure()
        
        # image_shape = self.params.raw_image2.shape
        # flat_raw = self.params.raw_image2.flatten() + 0
        
        # top_half =
        # bot_half =
        # percentile_image_flat = stats.rankdata(flat_raw, "average")/len(flat_raw)
        # plt.imshow(self.params.quantile_image, origin="lower")
        # plt.show()
        
        # percentile_image_flat = stats.rankdata(flat_raw, "average")/len(flat_raw)
        # self.params.percentile_image = percentile_image_flat.reshape(image_shape)
        
        self.params.percentile_image = self.params.quantile_image
        
        # raw = flat_raw.reshape(image_shape)
        # plt.imshow(self.orig_smasher(raw), vmin=0, vmax=1)
        # plt.imshow(self.params.modified_image,origin='lower', vmin=0, vmax=1)
        # plt.show(block=True)
    
    def do_percentile_plot(self):
        fig, axArray = plt.subplots(2, 3, sharex='row', sharey="row")
        ((axA, axB, axC), (ax1, ax2, ax3)) = (top_axes, bot_axes) = axArray
        self.plot_percentilize_points(bot_axes)
        self.plot_percentilize_images(top_axes)
        plt.show(block=True)
    
    def plot_percentilize_points(self, axes):
        axA, axB, axC = axes
        ## Row 2, Distribution of points
        self.skip_points = 10 if self.params.rez < 3000 else 300
        
        # Gather Points to Display
        flat_raw = self.params.raw_image2.flatten() + 0
        flat_sunback = self.params.modified_image.flatten() + 0
        flat_percentilize = self.params.percentile_image.flatten() + 0
        
        # Take a short subset of the points
        absiss = self.n2r(self.rad_flat[::self.skip_points])
        
        raw_short_points = self.orig_smasher(flat_raw[::self.skip_points])
        sunback_short_points = flat_sunback[::self.skip_points]
        percentile_short_points = flat_percentilize[::self.skip_points]
        
        # Plot Scatter Plots
        blk_alpha = 0.4
        axA.set_title("log10(raw)/2")
        axA.scatter(absiss, raw_short_points, c='k', s=4, alpha=blk_alpha, edgecolors='none')
        
        axB.set_title("Sunback")
        axB.scatter(absiss, sunback_short_points, c='k', s=4, alpha=blk_alpha, edgecolors='none')
        
        axC.set_title("Quantilize")
        axC.scatter(absiss, percentile_short_points, c='k', s=4, alpha=blk_alpha, edgecolors='none')
        
        # ## Plot Formatting
        
        # Horizontal Lines
        axA.axhline(0)
        axB.axhline(0)
        axC.axhline(0)
        
        axA.axhline(1)
        axB.axhline(1)
        axC.axhline(1)
        
        # Plot Limits
        # axA.set_ylim((-2, 300))
        axA.set_ylim((-0.25, 1.25))
        axB.set_ylim((-0.25, 1.25))
        axC.set_ylim((-0.25, 1.25))
        
        axA.set_xlim((-0.05, 1.75))
        axB.set_xlim((-0.05, 1.75))
        axC.set_xlim((-0.05, 1.75))
        
        # Plot Scales
        # axA.set_yscale('symlog')
        
        # Plot Labels
        axA.set_ylabel("Intensity")
        axA.set_xlabel("Distance from Sun Center")
        axB.set_xlabel("Distance from Sun Center")
        axC.set_xlabel("Distance from Sun Center")
    
    def orig_smasher(self, orig):
        return np.log10(orig) / 2
    
    def plot_percentilize_images(self, axes):
        ## Plot Images
        ax1, ax2, ax3 = axes
        
        ax1.imshow(self.orig_smasher(self.params.raw_image2), origin='lower', cmap='gray', vmin=0, vmax=1)
        ax2.imshow(self.params.modified_image, origin='lower', cmap='gray', vmin=0, vmax=1)
        ax3.imshow(self.params.percentile_image, origin='lower', cmap='gray', vmin=0, vmax=1)
    
    def plot_norm_curves(self, show=False, save=True, fig=None, ax=None,
                         extra=False, raw=False, smooth=True):
        """Look at the results of the algorithm"""
        
        if ax is None or fig is None:
            fig, ax = plt.subplots(num="Doing Statistics on Intensity vs Height")
            do_save = True
        else:
            do_save = False
        
        ax.set_title("Intensity as a function of radial distance: AIA_{}".format(self.params.current_wave()))
        ## Plotting ##
        do_all = True
        raw_alpha = 0.85
        grey_alpha = 0.90
        
        rrarr = self.n2r(np.arange(len(self.outer_max)))
        # Plot Raw Curves
        if do_all and self.outer_max is not None:
            if raw: ax.plot(rrarr, self.outer_max, zorder=4, lw=1, label="Top/Bot", alpha=raw_alpha, c='orange')
            if extra: ax.plot(rrarr, self.inner_max, zorder=5, lw=1, label="In Max", alpha=raw_alpha, c='cornflowerblue')
            if extra: ax.plot(rrarr, self.inner_min, zorder=6, lw=1, label="In Min", alpha=raw_alpha, c='cornflowerblue')
            if raw: ax.plot(rrarr, self.outer_min, zorder=3, lw=1, alpha=raw_alpha, c='orange')
        
        # # Plot Final Curves
        # if do_all and self.norm_curve_max is not None:
        #     if True: ax.plot(rrarr, self.norm_curve_max, zorder=4, lw=4, label="Top/Bot", alpha=raw_alpha, c='orange')
        #     if True: ax.plot(rrarr, self.norm_curve_min, zorder=4, lw=4, alpha=raw_alpha, c='orange')
        
        # Plot Current Frame Curves
        if do_all and self.frame_maximum is not None and extra:
            if raw: ax.plot(rrarr, self.frame_maximum, zorder=8, lw=1, c='darkgrey', alpha=grey_alpha)
            if raw: ax.plot(rrarr, self.frame_minimum, zorder=7, lw=1, label="Frame", c='darkgrey', alpha=grey_alpha)
            if smooth: ax.plot(rrarr, self.savgol_filtered_frame_maximum, zorder=10, lw=1, c='darkslategrey', alpha=1)
            if smooth: ax.plot(rrarr, self.savgol_filtered_frame_minimum, zorder=9, lw=1, label="Smooth Frame", c='darkslategrey', alpha=1)
        
        # Plot Absolute Curves
        if do_all and self.abs_max is not None:
            if raw: ax.plot(rrarr, self.abs_max, zorder=1, lw=1, label="Hat/Shoe", c='darkgrey', alpha=grey_alpha)
            if raw: ax.plot(rrarr, self.abs_min, zorder=1, lw=1, c='darkgrey', alpha=grey_alpha)
            if smooth: ax.plot(rrarr, self.smooth_absol_maximum, zorder=200, lw=3, c='cornflowerblue', alpha=1)
            if smooth: ax.plot(rrarr, self.smooth_absol_minimum, zorder=200, lw=3, label="Abs Max/Min", c='cornflowerblue', alpha=1)
        
        RRarr = self.n2r(self.output_abscissa)
        # Plot Filtered Curves
        if self.savgol_filtered_inner_maximum is not None and extra:
            ax.plot(RRarr, self.savgol_filtered_outer_maximum, zorder=103, lw=1, c="m", label="Fltr. Out")
            ax.plot(RRarr, self.savgol_filtered_inner_maximum, zorder=102, lw=1, c='c', label="Fltr. Inn")
            ax.plot(RRarr, self.savgol_filtered_inner_minimum, zorder=102, lw=1, c="c", ls='--')
            ax.plot(RRarr, self.savgol_filtered_outer_minimum, zorder=103, lw=1, c='m', ls='--', )
        
        # Plot Smoothed Curves
        if self.smooth_inner_maximum is not None:
            if smooth: ax.plot(RRarr, self.smooth_outer_maximum, zorder=105, lw=3, c="g", label="Outer Max/Min")
            if smooth: ax.plot(RRarr, self.smooth_inner_maximum, zorder=104, lw=3, c='r', label="Inner Max/Min")
            if smooth: ax.plot(RRarr, self.smooth_inner_minimum, zorder=104, lw=3, c="r")
            if smooth: ax.plot(RRarr, self.smooth_outer_minimum, zorder=105, lw=3, c='g')
            # ax.plot(rrarr, self.savgol_filtered_frame_abs_max, zorder=1, lw=1,                   c='grey', alpha=1)
            # ax.plot(rrarr, self.savgol_filtered_frame_abs_min, zorder=1, lw=1, label="Smo. Abs", c='grey', alpha=1)
        
        # Plot Scatter Points
        self.skip_points = 10 if self.params.rez < 3000 else 50  # TODO Make this sample better, linear isn't appropriate because its a circle
        scat = self.params.raw_image2.flatten()
        blk_alpha = 0.4
        ax.scatter(self.n2r(self.rad_flat[::self.skip_points]), scat[::self.skip_points], c='k', s=4, alpha=blk_alpha, edgecolors='none', label="2. SRN")
        
        # self.touchup_TUNE(self.params.raw_image+0)
        
        ## Plot Formatting
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Distrance from Sun Center")
        ax.set_yscale("symlog")
        ax.set_ylim((-10 ** 1, 10 ** 4))
        ax.set_xlim((0, 1.75))
        ax.legend(loc='lower left')
        fig.set_size_inches(10, 10)
        # plt.show()
        # Plot Saving
        
        if do_save:
            self.force_save_inner_outer(save, fig, ax, show)
        if show:
            plt.show(block=True)
        
        # print('Plot_curves')
    
    def force_save_inner_outer(self, save, fig, ax0, show):
        # Save Path Stuff
        
        if save:
            bs = self.params.base_directory()
            if save == "single":
                folder_name = "norm_curves"
                file_name_1 = '{}_keyframe.png'.format(self.params.current_wave())
                file_name_2 = None
            else:
                folder_name = "analysis"
                fstring = self.file_basename[:-5]
                file_name_1 = 'keyframe_{}.png'.format(fstring)
                file_name_2 = 'zoom_{}.png'.format(fstring)
            
            save_path_1 = join(bs, folder_name, 'radial_hist_pre', file_name_1)
            save_path_2 = join(bs, folder_name, 'radial_hist_pre', 'zoom', file_name_2)
            
            makedirs(dirname(save_path_1), exist_ok=True)
            makedirs(dirname(save_path_2), exist_ok=True)
            fig.set_size_inches((20, 10))
            plt.tight_layout()
            
            while True:
                try:
                    plt.savefig(save_path_1, dpi=150)
                    self.this_index += 1
                    break
                except OSError as e:
                    print("  !!!!!!! Close the Dang Plot!", end='')
                print('.', end='')
            
            plt.xlim((0.9, 1.1))
            plt.ylim((10 ** 1.5, 10 ** 3.5))
            while False:
                try:
                    plt.savefig(save_path_2, dpi=150)
                    break
                except OSError as e:
                    print("  !!!!!!! Close the Dang Plot!", end='')
                print('.', end='')
        
        # Show or not
        if not show:
            plt.close(fig)
        else:
            plt.show(block=True)
    
    ###################################
    ## Raw Normalization Curve Stuff ##
    ###################################
    
    def bin_radially(self):  # TODO Make the save to fits work
        """Bin the intensities by radius """
        do_cache = False
        if do_cache:
            if not self.there_is_cached_data:
                self.do_bin()
                self.save_cached_data(self.radBins)
                self.there_is_cached_data = True
            else:
                self.load_cached_data(self.radBins)
        else:
            self.do_bin()
    
    def do_bin(self, skip=1):  # Bin the intensities by radius
        self.cut_pixels = skip
        
        for binI, dat, xx, yy, ind in zip(self.binInds[::self.cut_pixels],
                                          self.params.modified_image.flatten()[::self.cut_pixels],
                                          self.binXX[::self.cut_pixels], self.binYY[::self.cut_pixels], self.binII[::self.cut_pixels]):
            # for each dat,
            
            self.radBins[binI].append(dat)
            self.radBins_xy[binI].append((xx, yy))
            self.radBins_ind[binI].append(ind)
    
    def make_annular_rings(self, R1=32):
        
        RLast = self.params.rez
        num_bins = np.min((int(np.round((RLast / R1) ** 2)), RLast * 2))
        self.RN = np.zeros(num_bins)
        for N in np.arange(num_bins):
            self.RN[N] = np.sqrt(N) * R1
        
        self.plot_annular_rings()
    
    def plot_annular_rings(self):
        ## Make this do the annular rings thing.
        xy = (self.params.rez // 2, self.params.rez // 2)
        angle = np.linspace(0, 2 * np.pi, 150)
        cos, sin = np.cos(angle), np.sin(angle)
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        for N, rr in enumerate(self.RN):
            
            xx = rr * cos + xy[0]
            yy = rr * sin + xy[1]
            
            cut = 50
            if (not N % cut) or N == 1:
                # print(N, rr)
                ax.plot(xx, yy, lw=2)
            elif N < cut * 2:
                if not N % 5:
                    ax.plot(xx, yy, c='lightgrey', lw=1)
                    pass
        
        rr = self.found_limb_radius
        xx = rr * cos + xy[0]
        yy = rr * sin + xy[1]
        ax.plot(xx, yy, c='k', lw='3', ls=":")
        
        rr = self.params.rez // 2
        xx = rr * cos + xy[0]
        yy = rr * sin + xy[1]
        ax.plot(xx, yy, c='w', lw='2', ls="--", zorder=100000)
        
        to_plot = np.sqrt(self.params.modified_image)
        
        # to_plot[~np.isfinite(to_plot)]=np.min(to_plot)
        
        ax.imshow(to_plot, zorder=10000, alpha=0.7)
        
        ax.set_aspect(1)
        ax.set_xlim([-100, 4196])
        ax.set_ylim([-100, 4196])
        ax.axhline(0)
        ax.axhline(4096)
        ax.axvline(0)
        ax.axvline(4096)
        fig.set_size_inches((8, 8))
        plt.title("Annular Rings of constant area")
        plt.tight_layout()
        plt.show(block=True)
    
    def save_cached_data(self, radBins=None):
        if radBins is not None:
            self.radBins = radBins
        self.save_frame_to_fits_file(fits_path=self.fits_path, frame=np.asarray(self.radBins), out_name='radBins')
        pass
    
    def load_cached_data(self, in_name='radBins'):
        self.load_a_fits_attribute(fits_path=self.fits_path, field='radBins')
        pass
    
    def radial_statistics(self):  # TODO Make this much faster
        """ Find the statistics in each radial bin"""
        self.params.quantile_image = copy(self.params.raw_image.flatten())
        for ii, bin_list in enumerate(self.radBins):
            self.store_bin_array(ii)
        self.finalize_radial_statistics()
    
    def store_bin_array(self, ii):
        """Do statistics on a given bin"""
        
        bin_list = self.radBins[ii]
        keep, bin_array = self.get_bin_items(bin_list)
        coord = self.radBins_ind[ii]
        good_coord = [coord[x] for x in keep]
        
        if len(bin_array) > 0:
            quantileized = stats.rankdata(bin_array, "average") / len(bin_array)
            self.params.quantile_image[good_coord] = quantileized
            
            # a, b, c, d = np.percentile(bin_array, [98.5, 90, 3, 0.5])
            # self.binAbsMax[ii] = a  # np.percentile(bin_array, 99.999)
            # self.binMax[ii] = b  # np.percentile(bin_array, 96)
            # self.binMin[ii] = c  # np.percentile(bin_array, 2)
            # self.binAbsMin[ii] = d  # np.percentile(bin_array, 0.001)
            
            ## TODO make this be percentilized
    
    @staticmethod
    def get_bin_items(bin_list):
        """Retrieve finite values from a bin_list"""
        bin_array = np.asarray(bin_list)
        finite = np.isfinite(bin_array)
        filled = bin_array != 0
        keep = list(np.nonzero(finite & filled)[0])
        finite_out = bin_array[keep]
        return keep, finite_out
    
    def finalize_radial_statistics(self):
        """Clean up the radial statistics to be used"""
        idx = np.isfinite(self.binMax) & np.isfinite(self.binMin)
        n_index = self.binAbss[idx]
        #         import pdb; pdb.set_trace()
        self.frame_abss[n_index] = n_index
        self.frame_maximum[n_index] = self.binMax[idx]
        self.frame_minimum[n_index] = self.binMin[idx]
        self.frame_abs_max[n_index] = self.binAbsMax[idx]
        self.frame_abs_min[n_index] = self.binAbsMin[idx]
        
        self.params.quantile_image = self.params.quantile_image.reshape(self.params.modified_image.shape)
        
        # from utils.stretch_intensity_module import norm_stretch
        # self.params.quantile_image = norm_stretch(self.params.quantile_image, alpha=self.params.alpha)
        
        self.vignette()
    
    def find_limb_radius(self):
        self.load_curves()
        
        # self.found_limb_radius = 400 # self.params.found_limb_radius or 1600
        self.found_limb_radius = self.params.found_limb_radius or 1600
        self.lCut = int(self.found_limb_radius - 0.01 * self.params.rez)
        self.hCut = int(self.found_limb_radius + 0.01 * self.params.rez)
        
        try:
            # abss = self.frame_abss
            use_max = self.outer_max + 0
            use_min = self.outer_min + 0
            
            # outer_mid_abs = abss[self.lCut:self.hCut]
            
            outer_mid_max = self.outer_max[self.lCut:self.hCut]
            inner_mid_max = self.inner_max[self.lCut:self.hCut]
            inner_mid_min = self.inner_min[self.lCut:self.hCut]
            outer_mid_min = self.outer_min[self.lCut:self.hCut]
            
            outer_mid_max_maxInd = np.argmax(outer_mid_max) + self.lCut
            inner_mid_max_maxInd = np.argmax(inner_mid_max) + self.lCut
            inner_mid_min_maxInd = np.argmax(inner_mid_min) + self.lCut
            outer_mid_min_maxInd = np.argmax(outer_mid_min) + self.lCut
            
            self.peak_indList = [outer_mid_max_maxInd, inner_mid_max_maxInd,
                                 inner_mid_min_maxInd, outer_mid_min_maxInd]
            self.fit_limb_radius = int(np.round(np.mean(self.peak_indList), 0))
        except TypeError as e:
            # print("\r        find_limb_radius failed: ", e)
            self.fit_limb_radius = self.found_limb_radius
        
        self.lCut = int(self.fit_limb_radius - 0.01 * self.params.rez)
        self.hCut = int(self.fit_limb_radius + 0.00 * self.params.rez)
    
    ###################################
    ## Smoothed Normalization Curve Stuff ##
    ###################################
    
    def make_save_smoothed_curves(self, banner=False):  ## SNARFLAT Work Here damnit
        """Build the normalization arrays, treating the domain in 3 seperate regions"""
        if self.abs_max is not None:
            # self.save_curves(banner=False)    # TODO This might need to be on
            if banner: print("\r *        Smoothing Curves...", end='')
            self.despike_curves()
            self.triFilter_curves()
            self.monoFilter_curves()
            self.render_smooth_curves()
            if banner: print("Success!")
            
            self.save_curves(banner=banner)
            
            self.select_curves_TUNE()
            # print("Smoothed Curves")
    
    def despike_curves(self):
        # if self.make_curves_latch:
        #     self.abs_max = self.despike(self.abs_max)
        # self.make_curves_latch = False
        
        pass
    
    def triFilter_curves(self):
        self.split_into_three_regions()
        self.filter_three_regions_TUNE()
        self.concatinate_filtered_regions()
    
    def init_smooth_curves(self):
        self.smooth_absol_maximum = self.savgol_filtered_absol_maximum + 0
        self.smooth_outer_maximum = self.savgol_filtered_outer_maximum + 0
        self.smooth_inner_maximum = self.savgol_filtered_inner_maximum + 0
        self.smooth_inner_minimum = self.savgol_filtered_inner_minimum + 0
        self.smooth_outer_minimum = self.savgol_filtered_outer_minimum + 0
        self.smooth_absol_minimum = self.savgol_filtered_absol_minimum + 0
    
    def monoFilter_curves(self):
        self.init_smooth_curves()
        self.filter_all_regions_TUNE()
        self.curve_fit_smooth_curves_TUNE()
        self.flatten_smooth_curves()
    
    def split_into_three_regions(self):
        # Split the domain into three regions
        self.find_limb_radius()
        
        if self.frame_abss is None:
            self.frame_abss = np.arange(self.params.rez)
        
        self.smidge = smidge = 2
        
        # Split outer curves into three regions
        abss = self.frame_abss
        use_max = self.outer_max + 0
        use_min = self.outer_min + 0
        #
        self.outer_low_abs = abss[:self.lCut + smidge]
        self.outer_low_max = use_max[:self.lCut + smidge]
        self.outer_low_min = use_min[:self.lCut + smidge]
        #
        self.outer_mid_abs = abss[self.lCut - smidge:self.hCut + smidge]
        self.outer_mid_max = use_max[self.lCut - smidge:self.hCut + smidge]
        self.outer_mid_min = use_min[self.lCut - smidge:self.hCut + smidge]
        #
        self.outer_high_abs = abss[self.hCut - smidge:]
        self.outer_high_max = use_max[self.hCut - smidge:]
        self.outer_high_min = use_min[self.hCut - smidge:]
        
        # Split inner curves into three regions
        abss = self.frame_abss
        use_max = self.inner_max + 0
        use_min = self.inner_min + 0
        #
        self.inner_low_abs = abss[:self.lCut + smidge]
        self.inner_low_max = use_max[:self.lCut + smidge]
        self.inner_low_min = use_min[:self.lCut + smidge]
        #
        self.inner_mid_abs = abss[self.lCut - smidge:self.hCut + smidge]
        self.inner_mid_max = use_max[self.lCut - smidge:self.hCut + smidge]
        self.inner_mid_min = use_min[self.lCut - smidge:self.hCut + smidge]
        #
        self.inner_high_abs = abss[self.hCut - smidge:]
        self.inner_high_max = use_max[self.hCut - smidge:]
        self.inner_high_min = use_min[self.hCut - smidge:]
        
        # Split absolute curves into three regions
        abss = self.frame_abss
        use_max = self.abs_max + 0
        use_min = self.abs_min + 0
        #
        self.absolute_low_abs = abss[:self.lCut + smidge]
        self.absolute_low_max = use_max[:self.lCut + smidge]
        self.absolute_low_min = use_min[:self.lCut + smidge]
        #
        self.absolute_mid_abs = abss[self.lCut - smidge:self.hCut + smidge]
        self.absolute_mid_max = use_max[self.lCut - smidge:self.hCut + smidge]
        self.absolute_mid_min = use_min[self.lCut - smidge:self.hCut + smidge]
        #
        self.absolute_high_abs = abss[self.hCut - smidge:]
        self.absolute_high_max = use_max[self.hCut - smidge:]
        self.absolute_high_min = use_min[self.hCut - smidge:]
    
    def concatinate_filtered_regions(self):
        # Concatinate filtered curves
        smidge = self.smidge
        self.output_abscissa = np.hstack((self.inner_low_abs[:self.lCut], self.inner_mid_abs[smidge:-smidge], self.inner_high_abs[smidge:]))
        self.savgol_filtered_absol_maximum = np.hstack(
                (self.absolute_low_max[:self.lCut], self.absolute_mid_max[smidge:-smidge], self.absolute_high_max[smidge:]))
        self.savgol_filtered_outer_maximum = np.hstack((self.outer_low_max[:self.lCut], self.outer_mid_max[smidge:-smidge], self.outer_high_max[smidge:]))
        self.savgol_filtered_inner_maximum = np.hstack((self.inner_low_max[:self.lCut], self.inner_mid_max[smidge:-smidge], self.inner_high_max[smidge:]))
        self.savgol_filtered_inner_minimum = np.hstack((self.inner_low_min[:self.lCut], self.inner_mid_min[smidge:-smidge], self.inner_high_min[smidge:]))
        self.savgol_filtered_outer_minimum = np.hstack((self.outer_low_min[:self.lCut], self.outer_mid_min[smidge:-smidge], self.outer_high_min[smidge:]))
        self.savgol_filtered_absol_minimum = np.hstack(
                (self.absolute_low_min[:self.lCut], self.absolute_mid_min[smidge:-smidge], self.absolute_high_min[smidge:]))
    
    def flatten_smooth_curves(self, flatten_down_ind=None, flatten_up_ind=None):
        # Flatten out the low edge
        flatten_inner_ind = flatten_down_ind or 200
        self.smooth_absol_maximum[0:flatten_inner_ind] = self.savgol_filtered_absol_maximum[flatten_inner_ind]
        self.smooth_outer_maximum[0:flatten_inner_ind] = self.savgol_filtered_outer_maximum[flatten_inner_ind]
        self.smooth_inner_maximum[0:flatten_inner_ind] = self.savgol_filtered_inner_maximum[flatten_inner_ind]
        self.smooth_inner_minimum[0:flatten_inner_ind] = self.savgol_filtered_inner_minimum[flatten_inner_ind]
        self.smooth_outer_minimum[0:flatten_inner_ind] = self.savgol_filtered_outer_minimum[flatten_inner_ind]
        self.smooth_absol_minimum[0:flatten_inner_ind] = self.savgol_filtered_absol_minimum[flatten_inner_ind]
        
        # Flatten out the high edge
        flatten_outer_ind = flatten_up_ind or int(self.r2n(1.7))
        self.smooth_absol_maximum[flatten_outer_ind:] = self.savgol_filtered_absol_maximum[flatten_outer_ind]
        self.smooth_outer_maximum[flatten_outer_ind:] = self.savgol_filtered_outer_maximum[flatten_outer_ind]
        self.smooth_inner_maximum[flatten_outer_ind:] = self.savgol_filtered_inner_maximum[flatten_outer_ind]
        self.smooth_inner_minimum[flatten_outer_ind:] = self.savgol_filtered_inner_minimum[flatten_outer_ind]
        self.smooth_outer_minimum[flatten_outer_ind:] = self.savgol_filtered_outer_minimum[flatten_outer_ind]
        self.smooth_absol_minimum[flatten_outer_ind:] = self.savgol_filtered_absol_minimum[flatten_outer_ind]
        
        self.flatten_up = self.n2r(flatten_outer_ind)
        self.flatten_down = self.n2r(flatten_inner_ind)
    
    def render_smooth_curves(self):
        self.norm_curve_absol_max = np.squeeze(self.smooth_absol_maximum[self.binInds])
        self.norm_curve_outer_max = np.squeeze(self.smooth_outer_maximum[self.binInds])
        self.norm_curve_inner_max = np.squeeze(self.smooth_inner_maximum[self.binInds])
        self.norm_curve_inner_min = np.squeeze(self.smooth_inner_minimum[self.binInds])
        self.norm_curve_outer_min = np.squeeze(self.smooth_outer_minimum[self.binInds])
        self.norm_curve_absol_min = np.squeeze(self.smooth_absol_minimum[self.binInds])
        
        # plt.plot(self.savgol_filtered_absol_maximum)
        # plt.show()
        # self.prep_save_outs()
        # Filter
        # flatten_inner_ind=200
        
        # if self.abs_max is not None:
        #     self.savgol_filtered_absol_maximum = self.abs_max + 0
        #     self.savgol_filtered_absol_minimum = self.abs_min + 0
        #
        # for i in range(an):
        #     try:
        #         self.savgol_filtered_absol_maximum = savgol_filter(self.savgol_filtered_absol_maximum, aWindow, rank, mode=mode)
        #         self.savgol_filtered_absol_minimum = savgol_filter(self.savgol_filtered_absol_minimum, aWindow, rank, mode=mode)
        #     except np.linalg.LinAlgError as e:
        #         print("\n filter:three:regions::")
        #         print(e)
        
        #
        # if self.abs_max is not None:
        #     filtered_abs_max = savgol_filter(self.abs_max, 21, 3, mode='nearest')
        #     filtered_abs_min = savgol_filter(self.abs_min, 21, 3, mode='nearest')
        #     self.savgol_filtered_absol_maximum = filtered_abs_max
        #     self.savgol_filtered_absol_minimum = filtered_abs_min
        #
        #
        # if self.frame_minimum is not None:
        #     self.savgol_filtered_frame_minimum = self.frame_minimum + 0
        #     self.savgol_filtered_frame_maximum = self.frame_maximum + 0
        #
        #     self.savgol_filtered_frame_abs_max = self.frame_abs_max + 0
        #     self.savgol_filtered_frame_abs_min = self.frame_abs_min + 0
        #
        #     for i in range(maxn):
        #         try:
        #             # Bonus Extrema Filtering!
        #             if self.frame_minimum is not None:
        #                 self.savgol_filtered_frame_minimum = savgol_filter(self.savgol_filtered_frame_minimum, maxWindow, rank, mode=mode)
        #                 self.savgol_filtered_frame_maximum = savgol_filter(self.savgol_filtered_frame_maximum, maxWindow, rank, mode=mode)
        #                 self.savgol_filtered_frame_abs_max = savgol_filter(self.savgol_filtered_frame_abs_max, maxWindow, rank, mode=mode)
        #                 self.savgol_filtered_frame_abs_min = savgol_filter(self.savgol_filtered_frame_abs_min, maxWindow, rank, mode=mode)
        #         except np.linalg.LinAlgError as e:
        #             print("\n filter:three:regions::")
        #             print(e)
        #
    
    ###################################
    ##   Places to TUNE the model    ##
    ###################################
    
    def filter_three_regions_TUNE(self):
        ### Filter the regions separately
        mode = 'nearest'
        
        # Savgol windows
        lWindow = 201  # 4 * self.extra_rez + 1
        ln = 1  # 6
        mWindow = 11  # 4 * self.extra_rez + 1
        mn = 3  # 3
        hWindow = 51  # 30 * self.extra_rez + 1
        hn = 3  # 2
        
        rank = 2
        
        # Filter Low
        for i in range(ln):
            
            try:
                self.absolute_low_max = savgol_filter(self.absolute_low_max, lWindow, rank, mode=mode)
                self.outer_low_max = savgol_filter(self.outer_low_max, lWindow, rank, mode=mode)
                self.inner_low_max = savgol_filter(self.inner_low_max, lWindow, rank, mode=mode)
                self.inner_low_min = savgol_filter(self.inner_low_min, lWindow, rank, mode=mode)
                self.outer_low_min = savgol_filter(self.outer_low_min, lWindow, rank, mode=mode)
                self.absolute_low_min = savgol_filter(self.absolute_low_min, lWindow, rank, mode=mode)
            
            except np.linalg.LinAlgError as e:
                print("\n filter:three:regions::")
                print(e)
        
        # Filter Mid
        for i in range(mn):
            try:
                self.absolute_mid_max = savgol_filter(self.absolute_mid_max, mWindow, rank, mode=mode)
                self.outer_mid_max = savgol_filter(self.outer_mid_max, mWindow, rank, mode=mode)
                self.inner_mid_max = savgol_filter(self.inner_mid_max, mWindow, rank, mode=mode)
                self.inner_mid_min = savgol_filter(self.inner_mid_min, mWindow, rank, mode=mode)
                self.outer_mid_min = savgol_filter(self.outer_mid_min, mWindow, rank, mode=mode)
                self.absolute_mid_min = savgol_filter(self.absolute_mid_min, mWindow, rank, mode=mode)
            
            except np.linalg.LinAlgError as e:
                print("\n filter:three:regions::")
                print(e)
        
        # Filter High
        for i in range(hn):
            try:
                self.absolute_high_max = savgol_filter(self.absolute_high_max, hWindow, rank, mode=mode)
                self.outer_high_max = savgol_filter(self.outer_high_max, hWindow, rank, mode=mode)
                self.inner_high_max = savgol_filter(self.inner_high_max, hWindow, rank, mode=mode)
                self.inner_high_min = savgol_filter(self.inner_high_min, hWindow, rank, mode=mode)
                self.outer_high_min = savgol_filter(self.outer_high_min, hWindow, rank, mode=mode)
                self.absolute_high_min = savgol_filter(self.absolute_high_min, hWindow, rank, mode=mode)
            
            except np.linalg.LinAlgError as e:
                print("\n filter:three:regions::")
                print(e)
    
    def filter_all_regions_TUNE(self):
        mode = 'nearest'
        aWindow = 21  # 30 * self.extra_rez + 1
        an = 1
        rank = 2
        
        # Filter All
        for i in range(an):
            try:
                self.smooth_absol_maximum = savgol_filter(self.smooth_absol_maximum, aWindow, rank, mode=mode)
                self.smooth_outer_maximum = savgol_filter(self.smooth_outer_maximum, aWindow, rank, mode=mode)
                self.smooth_inner_maximum = savgol_filter(self.smooth_inner_maximum, aWindow, rank, mode=mode)
                self.smooth_inner_minimum = savgol_filter(self.smooth_inner_minimum, aWindow, rank, mode=mode)
                self.smooth_outer_minimum = savgol_filter(self.smooth_outer_minimum, aWindow, rank, mode=mode)
                self.smooth_absol_minimum = savgol_filter(self.smooth_absol_minimum, aWindow, rank, mode=mode)
            except np.linalg.LinAlgError as e:
                print("\n filter:all:regions::")
                print(e)
    
    def curve_fit_smooth_curves_TUNE(self):
        # Fit the lowest region with a polynomial to make it much smoother
        # degree = 5
        # p = np.polyfit(self.low_abs, self.low_max_filt, degree)
        # self.low_max_fit = np.polyval(p, self.low_abs)  # * 1.1
        # p = np.polyfit(self.low_abs, self.low_min_filt, degree)
        # self.low_min_fit = np.polyval(p, self.low_abs)
        pass
    
    def select_curves_TUNE(self):
        ## These are the options
        
        # self.norm_curve_absol_max
        # self.norm_curve_outer_max
        # self.norm_curve_inner_max
        # self.norm_curve_inner_min
        # self.norm_curve_outer_min
        # self.norm_curve_absol_min
        
        # Select Bottom Norms
        
        self.show_norm = False
        
        self.norm_curve_max_bottom_name = "norm_curve_outer_max"
        self.norm_curve_min_bottom_name = "norm_curve_absol_min"
        self.norm_curve_max = getattr(self, self.norm_curve_max_bottom_name)
        self.norm_curve_min = getattr(self, self.norm_curve_min_bottom_name)
        
        # Select Top Norms
        self.norm_curve_max_top_name = "norm_curve_inner_max"
        self.norm_curve_min_top_name = "norm_curve_absol_min"
        norm_curve_max_top = getattr(self, self.norm_curve_max_top_name)
        norm_curve_min_top = getattr(self, self.norm_curve_min_top_name)
        
        # Merge the two
        low = self.fit_limb_radius
        self.norm_curve_max[low:] = norm_curve_max_top[low:]
        self.norm_curve_min[low:] = norm_curve_min_top[low:]
    
    @staticmethod
    def norm_formula(image, the_min, the_max):
        """Standard Normalization Formula"""
        
        image_flat = image.flatten()
        
        diff = np.subtract(the_max, the_min)
        np.subtract(image_flat, the_min, out=image_flat)
        np.divide(image_flat, diff, out=image_flat)
        
        image = image_flat.reshape(image.shape)
        
        return image
    
    def touchup_TUNE(self, img):
        img *= 10.
        np.power(img, 1 / 3, out=img)
        img /= 3.5
        # img += 0.1
        
        # img[img > 1.] = np.power(img[img > 1.], 1/2)
        
        # img *= 1.5
        # img -= 0.75
        
        # img[img < 0.] = 0.
        # img[img == 0.] = np.nan
        img[~np.isfinite(img)] = np.nan
        return img
    
    def coronagraph_touchup_TUNE(self):
        """Deal with pixel outliers. Lots of adjustable parameters in here"""
        self.touchup_TUNE(self.params.raw_image)
        self.touchup_TUNE(self.params.modified_image)
        # neg = self.modified_image<0
        # neg_pts = self.modified_image[neg]
        # minn = np.abs(np.min(neg_pts))
        # normed = neg_pts + min(neg_pts)
        
        # self.modified_image += minn
        
        # self.params.modified_image = np.power(self.params.modified_image, 1 / 3)
        # self.params.modified_image -= 0.15
        # self.params.modified_image /= 1.5
        
        # self.modified_image = np.power(self.modified_image, 1/4)
        # self.modified_image -= minn
        
        ## Deal with too hot things ##
        # self.vmax = 2
        # self.vmax_plot = 0.95  # np.max(changed_flat) #this is in the header of the imageprocessor now
        # hotpowr = 1 / 2
        # hot = self.modified_image > self.vmax
        # self.modified_image[hot] = self.modified_image[hot] ** hotpowr
        
        # ## Deal with too cold things ##
        # self.vmin = 0.3
        # self.vmin_plot = -0.05  # np.min(changed_flat)# 0.3# -0.03 #this is in the header of the imageprocessor now
        # coldpowr = 1 / 2
        # cold = self.changed_flat < self.vmin
        # self.changed_flat[cold] = -((np.abs(self.changed_flat[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin
        
        ## Some Final Normalization ##      TODO: I think this might be breaking things!
        # self.changed_flat = self.normalize(self.changed_flat, high=99.99, low=1)
        pass
    
    #######################################
    ## Image Reduction Helper Algorithms ##
    #######################################
    
    @staticmethod
    def rolling_window(data, block):
        shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
        strides = data.strides + (data.strides[-1],)
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    def despike(self, arr, n1=2.5, n2=40, block=25):
        # Condition the Input
        data = arr.copy()
        data[data == -1] = np.NaN
        offset = np.nanmin(data)
        data -= offset
        roll = self.rolling_window(data, block)
        roll = np.ma.masked_invalid(roll)
        std = n1 * roll.std(axis=1)
        mean = roll.mean(axis=1)
        # Use the last value to fill-up.
        std = np.r_[std, np.tile(std[-1], block - 1)]
        mean = np.r_[mean, np.tile(mean[-1], block - 1)]
        mask = (np.abs(data - mean.filled(fill_value=np.NaN)) >
                std.filled(fill_value=np.NaN))
        data[mask] = np.NaN
        # Pass two: recompute the mean and std without the flagged values from pass
        # one now removing the flagged data.
        roll = self.rolling_window(data, block)
        roll = np.ma.masked_invalid(roll)
        std = n2 * roll.std(axis=1)
        mean = roll.mean(axis=1)
        # Use the last value to fill-up.
        std = np.r_[std, np.tile(std[-1], block - 1)]
        mean = np.r_[mean, np.tile(mean[-1], block - 1)]
        mask = (np.abs(arr - mean.filled(fill_value=np.NaN)) >
                std.filled(fill_value=np.NaN))
        arr[mask] = mean[mask]
        return arr + offset
    
    def coronaNorm(self):
        """Normalize the in_object using the radial percentile curves"""
        # Make Curves
        self.make_save_smoothed_curves()
        
        # Normalize Them
        self.execute_norm()
        
        # Deal with some outliers
        self.coronagraph_touchup_TUNE()
        
        # Vignette
        self.vignette()  # Truncate the in_object above given radius
    
    def execute_norm(self):
        """Apply the Normalization to the Image Array"""
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # Standard Normalization Formula
                self.params.modified_image = self.norm_formula(self.params.modified_image, self.norm_curve_min, self.norm_curve_max)
                # self.params.modified_image = self.params.quantile_image
            
            except RuntimeWarning as e:
                print(e)
        return
    
    @staticmethod
    def norm_formula(image, the_min, the_max):
        """Standard Normalization Formula"""
        image_flat = image.flatten()
        diff = np.subtract(the_max, the_min)
        np.subtract(image_flat, the_min, out=image_flat)
        np.divide(image_flat, diff, out=image_flat)
        image = image_flat.reshape(image.shape)
        return image
    
    def prep_output(self):
        self.mask_output()
        self.mirror_output()
    
    def mask_output(self, do_mask=None):
        """Allows you to only show sub-sections of the in_object as reduced images"""
        if not do_mask:
            return False
        
        self.grid_mask = self.get_mask(self.params.modified_image, force=True)
        
        if self.grid_mask is not None:
            self.params.modified_image[self.grid_mask] = self.params.raw_image[self.grid_mask]
    
    def mirror_output(self, do_mirror=None):
        # Allows you to mirror horizontally, with only one half rfeduced
        if not do_mirror:
            return False
        
        self.mirror_mask = self.get_mask(self.params.modified_image, force=True)
        
        newDat = self.params.modified_image[self.mirror_mask]
        xx, yy = self.mirror_mask.shape[0], int(self.mirror_mask.shape[1] / 2)
        grid = newDat.reshape(xx, yy)
        flipped = np.fliplr(grid)
        
        if self.mirror_mask is not None:
            self.params.modified_image[~self.mirror_mask] = flipped.flatten()  # np.flip(newDat)
    
    def get_mask(self, output_frame, force=None):
        """ Generates a mask that defines which portion of the in_object will be modified"""
        if force is not None:
            self.renew_mask = force
        if not self.renew_mask:
            return self.grid_mask
        
        corona_mask = np.full_like(output_frame, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)
        
        mode = 'y'
        
        if type(mode) in [float, int]:
            mask_num = mode
        elif 'y' in mode:
            mask_num = 1
        elif 'n' in mode:
            mask_num = 2
        else:
            if 'r' in mode:
                if len(mode) < 2:
                    mode += 'a'
            
            if 'a' in mode:
                top = 8
                btm = 1
            elif 'h' in mode:
                top = 6
                btm = 3
            elif 'd' in mode:
                top = 8
                btm = 7
            elif 'w' in mode:
                top = 2
                btm = 1
            else:
                print('Unrecognized Mode')
                top = 8
                btm = 1
            
            ii = 0
            while True:
                mask_num = np.random.randint(btm, top + 1)
                if mask_num not in self.mask_num:
                    self.mask_num.append(mask_num)
                    break
                ii += 1
                if ii > 10:
                    self.mask_num = []
        
        if mask_num == 1:
            corona_mask[:, :] = True
        
        if mask_num == 2:
            corona_mask[:, :] = False
        
        if mask_num == 3:
            corona_mask[half:, :] = True
        
        if mask_num == 4:
            corona_mask[:half, :] = True
        
        if mask_num == 5:
            corona_mask[:, half:] = True
        
        if mask_num == 6:
            corona_mask[:, :half] = True
        
        if mask_num == 7:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
        
        if mask_num == 8:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
            corona_mask = np.invert(corona_mask)
        
        return corona_mask
    
    def vignette(self):
        """Truncate the in_object above a certain radis"""
        return
        self.params.modified_image[self.vignette_mask] = np.nan
        self.params.quantile_image[self.vignette_mask] = np.nan
        self.params.raw_image[self.vignette_mask] = np.nan
    
    ########################
    ## Plotting Stuff ##
    ########################
    
    def peek_norm(self, do=False, show=False, save=True):
        """This plot is in radius and has a scatter plot
            overlaid with the norm curves as determined elsewhere"""
        vprint(" *    Plotting Analysis...     ", end='')
        the_alpha = 0.05
        
        # Init the Figure
        fig, (ax0) = plt.subplots(1, 1, sharex="all", num="Radial Statistics")
        
        skip = 1000
        self.skip_points = 100 if self.params.rez < 3000 else skip
        
        ########################
        ##  Plot 0: Absolute  ##
        ########################
        self.plot_norm_curves(fig=fig, ax=ax0, save=False, smooth=True, extra=False, raw=False)
        
        # Plot Scattered Points from the raw image_path in midnightblue
        orig_abs = self.params.raw_image.flatten()
        ax0.scatter(self.n2r(self.rad_flat[::self.skip_points]), orig_abs[::self.skip_points],
                    alpha=the_alpha, edgecolors='none', c='midnightblue', s=3)
        
        # Vertical Lines
        # ax0.axvline(1, lw=3)
        if self.lCut is not None:
            ax0.axvline(self.n2r(self.lCut), ls=":")
            ax0.axvline(self.n2r(self.hCut), ls=":")
        
        # for line in self.peak_indList:
        #     ax0.axvline(self.n2r(line), c='orange')
        
        ########################
        ## Plot 1: Normalized ##
        ########################
        
        # # Plot Scattered Points from the raw image_path in midnightblue
        # ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), orig_abs[::self.skip_points], zorder=-1,
        #             alpha=the_alpha, edgecolors='none', c='midnightblue', s=3, label="1. t_int")
        #
        # # Plot Scattered Points from the raw image_path but rooted, in red
        # self.touchup_TUNE(self.params.raw_image)
        # scat2 = self.params.raw_image.flatten()
        # ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), scat2[::self.skip_points],
        #             alpha=the_alpha, edgecolors='none', c='r', s=3, zorder=0, label="2. ROOT")
        #
        # # Plot Scattered Points from the final modified image_path, in black
        # self.touchup_TUNE(self.params.modified_image)
        # points = np.array(self.params.modified_image.flatten(), dtype=np.float32)
        # ax1.scatter(self.n2r(self.rad_flat[::skip]), points[::skip], c='k', s=3, alpha=the_alpha, edgecolors='none', label="3. SRN")
        #
        # # Extra Lines
        # ax1.axhline(2, c='lightgrey', ls=':', zorder=-1)
        # ax1.axhline(1, c='k', ls=':', zorder=-1)
        # ax1.axhline(0, c='k', ls=':', zorder=-1)
        
        ## Plot 0 Formatting
        ax0.set_title("Various Norm Curves in Absolute Scale")
        
        full = True
        if full:
            ax0.set_ylim((-10 ** 1, 10 ** 4))
            ax0.set_xlim((0, 1.85))
        else:
            ax0.set_ylim((0, 1000))
            ax0.set_xlim((0.85, 1.15))
        
        ax0.axvline(self.vrad, ls=':', c='lightgrey', label='Vignette')
        
        if self.flatten_up:
            ax0.axvline(self.flatten_up, ls=':', c='grey', label='Flattening')
            ax0.axvline(self.flatten_down, ls=':', c='grey')
        
        ax0.annotate("Top Curve L:\n{}".format(self.norm_curve_max_bottom_name), (0.025, 0.3),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Bot Curve L:\n{}".format(self.norm_curve_min_bottom_name), (0.025, 0.2),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        
        ax0.annotate("Top Curve R:\n{}".format(self.norm_curve_max_top_name), (0.725, 0.7),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Bot Curve R:\n{}".format(self.norm_curve_min_top_name), (0.725, 0.6),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        
        # ax0.legend()
        # import matplotlib as mpl
        
        ax0.set_yscale('symlog')
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        # ## Plot 1 Formatting
        # ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        # ax1.set_ylabel(r"Normalized Intensity")
        # ax1.set_title("")
        # ax1.set_yscale("symlog")
        # ax1.set_ylim((-0.5, 20))
        # ax1.legend(markerscale=4., handletextpad=0.2, borderaxespad=0.3, scatteryoffsets=[0.55])
        #
        # import matplotlib as mpl
        # ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: int(x) if x >= 1 else x))
        # fig.set_size_inches(7, 11)
        #         fig.set_size_inches(7, 14)
        
        plt.tight_layout()
        plt.show(block=True)
        # 1/0
        # return True
        # self.force_save_radial_figures(save, fig, ax0, show)
        
        vprint("Success!")
        if not do:
            return
        if self.first:
            self.first = False
            return
        # import pdb; pdb.set_trace()
        # self.output_abscissa
        # dprint("plot_full_normalization")
        
        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))
        # ax.axvline(self.tRadius, c='r')
        # raw_touch = self.params.raw_image+0
        # self.touchup_TUNE(raw_touch)
    
    def plot_full_normalization(self, do=False, show=False, save=True):
        """This plot is in radius and has a scatter plot
               overlaid with the norm curves as determined elsewhere"""
        
        vprint(" *    Plotting Analysis...     ", end='')
        blu_alpha = 0.15
        red_alpha = 0.15
        blk_alpha = 0.4
        # Init the Figure
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex="all", num="Radial Statistics")
        
        skip = 1000
        self.skip_points = 100 if self.params.rez < 3000 else skip
        blk_skip = 100
        
        ########################
        ##  Plot 0: Absolute  ##
        ########################
        self.plot_norm_curves(fig=fig, ax=ax0, save=False)
        
        # Vertical Lines
        ax0.axvline(1)
        if self.lCut is not None:
            ax0.axvline(self.n2r(self.lCut), ls=":")
            ax0.axvline(self.n2r(self.hCut), ls=":")
        
        # Plot Scattered Points from the raw image_path in midnightblue
        
        orig_abs = self.params.raw_image.flatten()
        # ax0.scatter(self.n2r(self.rad_flat[::self.skip_points]), orig_abs[::self.skip_points],
        #             alpha=blu_alpha, edgecolors='none', c='midnightblue', s=3)
        
        ########################
        ## Plot 1: Normalized ##
        ########################
        
        # Plot Scattered Points from the raw image_path in midnightblue
        do_raw_scatter = False
        if do_raw_scatter:
            ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), orig_abs[::self.skip_points], zorder=-1,
                        alpha=blu_alpha, edgecolors='none', c='midnightblue', s=3, label="1. t_int")
        
        # Plot Scattered Points from the raw image_path but rooted, in red
        do_red_points = False
        if do_red_points:
            scat2 = self.params.raw_image.flatten()
            ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), scat2[::self.skip_points],
                        alpha=red_alpha, edgecolors='none', c='r', s=3, zorder=0, label="1. INT+ROOT")
        
        # Plot Scattered Points from the final modified image_path, in black
        points = np.array(self.params.modified_image.flatten(), dtype=np.float32)
        ax1.scatter(self.n2r(self.rad_flat[::blk_skip]), points[::blk_skip], c='k', s=3, alpha=blk_alpha, edgecolors='none', label="2. SRN")
        
        # Extra Lines
        ax1.axhline(2, c='lightgrey', ls=':', zorder=-1)
        ax1.axhline(1, c='k', ls=':', zorder=-1)
        ax1.axhline(0, c='k', ls=':', zorder=-1)
        
        ## Plot 0 Formatting
        ax0.set_title("Various Norm Curves in Absolute Scale")
        ax0.set_ylim((-10 ** 1, 10 ** 4))
        ax0.set_xlim((0, 1.85))
        
        ax0.axvline(self.vrad, ls=':', c='lightgrey')
        ax0.annotate("Top Curve L:\n{}".format(self.norm_curve_max_bottom_name), (0.025, 0.3),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Bot Curve L:\n{}".format(self.norm_curve_min_bottom_name), (0.025, 0.2),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Top Curve R:\n{}".format(self.norm_curve_max_top_name), (0.65, 0.9),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Bot Curve R:\n{}".format(self.norm_curve_min_top_name), (0.65, 0.8),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        
        # ax0.legend()
        # import matplotlib as mpl
        
        ax0.set_yscale('symlog')
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        ## Plot 1 Formatting
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax1.set_title("")
        ax1.set_yscale("symlog")
        ax1.set_ylim((-0.5, 2))
        ax1.legend(markerscale=4., handletextpad=0.2, borderaxespad=0.3, scatteryoffsets=[0.55])
        
        import matplotlib as mpl
        ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: int(x) if x >= 1 else x))
        fig.set_size_inches(11, 9)
        #         fig.set_size_inches(7, 14)
        
        plt.tight_layout()
        # plt.show(block=True)
        # 1/0
        # return True
        self.force_save_radial_figures(save, fig, ax0, show)
        
        vprint("Success!")
        if not do:
            return
        if self.first:
            self.first = False
            return
        # import pdb; pdb.set_trace()
        # self.output_abscissa
        # dprint("plot_full_normalization")
        
        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))
        # ax.axvline(self.tRadius, c='r')
        # raw_touch = self.params.raw_image+0
        # self.touchup_TUNE(raw_touch)
    
    def plot_full_normalization_server(self, do=False, show=False, save=True):
        """This plot is in radius and has a scatter plot
                overlaid with the norm curves as determined elsewhere"""
        the_alpha = 0.5
        # Init the Figure
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex="all", num="Radial Statistics")
        fig0 = fig1 = fig
        
        #         skip = 100
        #         self.skip_points = 10 if self.params.rez < 3000 else skip
        skip = self.skip_points = 1
        #         ########################
        #         ##  Plot 0: Absolute  ##
        #         ########################
        #         self.plot_norm_curves(fig=fig1, ax=ax0, save=False)
        
        # Vertical Lines
        ax0.axvline(1)
        if self.lCut is not None:
            ax0.axvline(self.n2r(self.lCut), ls=":")
            ax0.axvline(self.n2r(self.hCut), ls=":")
        
        # Plot Scattered Points from the raw image_path in midnightblue
        flat_raw = self.params.raw_image.flatten()
        ax0.scatter(self.n2r(self.rad_flat[::self.skip_points]), flat_raw[::self.skip_points],
                    alpha=the_alpha, edgecolors='none', c='midnightblue', s=3)
        
        ########################
        ## Plot 1: Normalized ##
        ########################
        
        # Plot Scattered Points from the raw image_path in midnightblue
        #         ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), flat_raw[::self.skip_points], zorder=-1,
        #                     alpha=the_alpha, edgecolors='none', c='midnightblue', s=3, label="1. t_int")
        
        # Plot Scattered Points from the raw image_path but rooted, in red
        flat_raw = self.params.raw_image.flatten()
        touched_raw = self.touchup_TUNE(self.params.raw_image + 0)
        #         ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), touched_raw[::self.skip_points],
        #                     alpha=the_alpha, edgecolors='none', c='r', s=3, zorder=0, label="2. ROOT")
        
        # Plot Scattered Points from the final modified image_path, in black
        self.touchup_TUNE(self.params.modified_image)
        flat_modified_image = np.array(self.params.modified_image.flatten(), dtype=np.float32)
        ax1.scatter(self.n2r(self.rad_flat[::skip]), flat_modified_image[::skip], c='k', s=3, alpha=the_alpha, edgecolors='none', label="3. SRN")
        #         points = np.array(self.params.modified_image.flatten(), dtype=np.float32)
        #         ax1.scatter(self.n2r(self.rad_flat[::skip]), points[::skip], c='k', s=3, alpha=the_alpha, edgecolors='none', label="")
        
        # Extra Lines
        ax1.axhline(2, c='lightgrey', ls=':', zorder=-1)
        ax1.axhline(1, c='k', ls=':', zorder=-1)
        ax1.axhline(0, c='k', ls=':', zorder=-1)
        #         ax1.axvline(0.5)
        
        ## Plot 0 Formatting
        ax0.set_title("Various Norm Curves in Absolute Scale")
        ax0.set_ylim((-10 ** 0, 10 ** 2.2))
        ax0.set_xlim((0, 1.85))
        
        ax0.axvline(self.vrad, ls=':', c='lightgrey')
        ax0.annotate("Top Curve:\n{}".format(self.norm_curve_max_name), (0.025, 0.3),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Bot Curve:\n{}".format(self.norm_curve_min_name), (0.025, 0.2),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        # ax0.legend()
        import matplotlib as mpl
        
        ax0.set_yscale('symlog')
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        ## Plot 1 Formatting
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax1.set_title("")
        ax1.set_yscale("symlog")
        ax1.set_ylim((-0.5, 1.5))
        ax1.legend(markerscale=4., handletextpad=0.2, borderaxespad=0.3, scatteryoffsets=[0.55])
        ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: int(x) if x >= 2 else "{:0.1f}".format(x)))
        fig0.set_size_inches(10, 5)
        plt.tight_layout()
        plt.show()
        
        return True
    
    def plot_full_normalization_orig(self, do=False, show=False, save=True):
        """This plot is in radius and has a scatter plot
            overlaid with the norm curves as determined elsewhere"""
        the_alpha = 0.05
        
        # Init the Figure
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex="all", num="Radial Statistics")
        
        skip = 100
        self.skip_points = 10 if self.params.rez < 3000 else skip
        
        ########################
        ##  Plot 0: Absolute  ##
        ########################
        self.plot_norm_curves(fig=fig, ax=ax0, save=False)
        
        # Vertical Lines
        ax0.axvline(1)
        if self.lCut is not None:
            ax0.axvline(self.n2r(self.lCut), ls=":")
            ax0.axvline(self.n2r(self.hCut), ls=":")
        
        # Plot Scattered Points from the raw image_path in midnightblue
        orig_abs = self.params.raw_image.flatten()
        ax0.scatter(self.n2r(self.rad_flat[::self.skip_points]), orig_abs[::self.skip_points],
                    alpha=the_alpha, edgecolors='none', c='midnightblue', s=3)
        
        ########################
        ## Plot 1: Normalized ##
        ########################
        
        # Plot Scattered Points from the raw image_path in midnightblue
        ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), orig_abs[::self.skip_points], zorder=-1,
                    alpha=the_alpha, edgecolors='none', c='midnightblue', s=3, label="1. t_int")
        
        # Plot Scattered Points from the raw image_path but rooted, in red
        self.touchup_TUNE(self.params.raw_image)
        scat2 = self.params.raw_image.flatten()
        ax1.scatter(self.n2r(self.rad_flat[::self.skip_points]), scat2[::self.skip_points],
                    alpha=the_alpha, edgecolors='none', c='r', s=3, zorder=0, label="2. ROOT")
        
        # Plot Scattered Points from the final modified image_path, in black
        self.touchup_TUNE(self.params.modified_image)
        points = np.array(self.params.modified_image.flatten(), dtype=np.float32)
        ax1.scatter(self.n2r(self.rad_flat[::skip]), points[::skip], c='k', s=3, alpha=the_alpha, edgecolors='none', label="3. SRN")
        
        # Extra Lines
        ax1.axhline(2, c='lightgrey', ls=':', zorder=-1)
        ax1.axhline(1, c='k', ls=':', zorder=-1)
        ax1.axhline(0, c='k', ls=':', zorder=-1)
        
        ## Plot 0 Formatting
        ax0.set_title("Various Norm Curves in Absolute Scale")
        ax0.set_ylim((-10 ** 0, 10 ** 2.2))
        ax0.set_xlim((0, 1.85))
        
        ax0.axvline(self.vrad, ls=':', c='lightgrey')
        ax0.annotate("Top Curve L:\n{}".format(self.norm_curve_max_bottom_name), (0.025, 0.3),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Bot Curve L:\n{}".format(self.norm_curve_min_bottom_name), (0.025, 0.2),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Top Curve R:\n{}".format(self.norm_curve_max_top_name), (0.65, 0.9),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        ax0.annotate("Bot Curve R:\n{}".format(self.norm_curve_min_top_name), (0.65, 0.8),
                     xycoords='axes fraction', fontsize='medium', color='k')  # , horizontalalignment='center')
        
        # ax0.legend()
        import matplotlib as mpl
        
        ax0.set_yscale('symlog')
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        ## Plot 1 Formatting
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax1.set_title("")
        ax1.set_yscale("symlog")
        ax1.set_ylim((-0.5, 20))
        ax1.legend(markerscale=4., handletextpad=0.2, borderaxespad=0.3, scatteryoffsets=[0.55])
        
        ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: int(x) if x >= 1 else x))
        fig.set_size_inches(7, 11)
        #         fig.set_size_inches(7, 14)
        
        plt.tight_layout()
        plt.show()
        return True
    
    #         self.force_save_radial_figures(save, fig, ax0, show)
    #         if not do:
    #             return
    #         if self.first:
    #             self.first = False
    #             return
    #         import pdb; pdb.set_trace()
    # self.output_abscissa
    #         dprint("plot_full_normalization")
    
    # locs = np.arange(self.rez)[::int(self.rez/5)]
    # ax1.set_xticks(locs)
    # ax1.set_xticklabels(self.n2r(locs))
    # ax.axvline(self.tRadius, c='r')
    # raw_touch = self.params.raw_image+0
    # self.touchup_TUNE(raw_touch)
    
    def force_save_radial_figures(self, save, fig, ax0, show):
        first = True
        while True:
            try:
                self.save_radial_figures(save, fig, ax0, show)
                # if not first: print("  Thanks, good job.\n")
                break
            except OSError as e:
                if first:
                    print("\n\n", e)
                    print("  !!!!!!! Close the Dang Plot!", end='')
                    first = False
                print('.', end='')
    
    def save_radial_figures(self, do=False, fig=None, ax=None, show=False):
        
        if do:
            save_path_1, save_path_2 = self.params.get_pre_radial_fig_paths()
            
            plt.savefig(save_path_1)
            
            if show:
                plt.show(block=True)
            
            ax.set_xlim((0.9, 1.1))
            plt.savefig(save_path_2)
        
        if not show:
            plt.close(fig)
    
    def get_points(self, index):
        ## Scatter Plot
        skip = 100
        return None
        plotY = self.radBins_all[index]
        
        xBox = []
        yBox = []
        for ii, bin in enumerate(plotY):
            for item in bin:
                xBox.append(self.n2r(ii))
                yBox.append(item)
        
        out = np.array((xBox, yBox))
        return out
    
    ########################
    ## Utilities ##
    ########################
    ## Static Methods ##
    def n2r(self, n):
        """Convert index to solar radius"""
        if not self.fit_limb_radius:
            self.find_limb_radius()
        if n is None:
            n = 0
        r = n / self.fit_limb_radius
        return r
    
    def r2n(self, r):
        """Convert index to solar radius"""
        if not self.fit_limb_radius:
            self.find_limb_radius()
        n = r * self.fit_limb_radius
        return n
    
    @staticmethod
    def normalize(image, high=98, low=15):
        """Normalize the Array"""
        if low is None:
            lowP = 0
        else:
            lowP = np.nanpercentile(image, low)
        highP = np.nanpercentile(image, high)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                out = (image - lowP) / (highP - lowP)
            except RuntimeWarning as e:
                out = image
        return out
    
    @staticmethod
    def fill_end(use):
        iii = -1
        val = use[iii]
        while np.isnan(val):
            iii -= 1
            val = use[iii]
        use[iii:] = val
        return use
    
    @staticmethod
    def fill_start(use):
        iii = 0
        val = use[iii]
        while np.isnan(val):
            iii += 1
            val = use[iii]
        use[:iii] = val
        return use


class SRNSingleShotProcessor(SRNProcessor):
    out_name = 'SRN'
    name = filt_name = 'SRN Single Shot Processor'
    description = "Create and Apply the Radial SRN Curves"
    progress_verb = 'Processing'
    finished_verb = "Modified"
    show_plots = True
    
    def __init__(self, fits_path=None, in_name=None, orig=False,
                 show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path=fits_path, in_name=in_name, orig=orig, show=show, verb=verb, quick=quick, rp=rp, params=params)

        self.in_name = in_name or self.params.master_frame_list_newest
        self.first = True
        self.go_ahead = True
        self.params.current_wave('rainbow')
        self.params.Force_init = True
        self.can_use_keyframes = False
        self.can_initialize = False
        self.frame_list = []
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.params.use_cdf:
            return self.run_cdf()
        else:
            return self.run_single()
    
    def run_single(self):
        """Run the program on a single loaded frame"""
        self.image_learn()  # Analyze the input to help make normalization curves
        self.image_modify()  # Actually Normalize This Image
        self.image_plot()
        self.first = False
        self.plot_full_normalization(save=True, show=False, do=True)
        print(" ^ Success!\n")
        return self.params.modified_image
    
    def run_cdf(self, do_plot=False):
        """Run the program on a single loaded frame"""
        
        for index in range(self.params.n_frames):
            #             print("Frame {}".format(index))
            self.params.cdf_fetcher.select_frame(get_ind=index)
            self.image_learn()  # Analyze the input to help make normalization curves
            self.image_modify()  # Actually Normalize This Image
            self.image_store_cdf()
            if do_plot:
                self.params.cdf_fetcher.peek_selection()
                self.image_plot()
            self.first = False
        self.image_save_cdf()
        print(" ^ Success!\n")
    
    #             return self.params.modified_image
    #         import pdb; pdb.set_trace()
    #         self.params.old_fetchers
    
    #         the_fetcher = [x for x in self.params.old_fetchers if type(x) is LocalCdfFetcher][0]
    
    #         for frame in the_fetcher.select_frame(gen=True):
    #             print(frame)
    
    #         pass
    
    def image_store_cdf(self, do_plot=False):
        
        wave = self.params.image_data[0] + 0
        frame = self.touchup_TUNE(self.params.modified_image) + 0
        
        self.frame_list.append((frame, wave))
        
        #         print("           Storing Frame {}...".format(wave), pointing_end='')
        if do_plot:
            plt.imshow(frame, origin='lower')
            plt.title(wave)
            plt.show()
    
    #         print("Not Yet Implemented")
    
    def image_save_cdf(self):
        self.params.cdf_fetcher.save_cdf(self.params.new_img_path, self.frame_list, self.params.confirm_save)
    
    def image_plot(self, save=False, show=True, do=True):
        # self.plot_norm_curves(save=False, show=True, extra=True)
        self.plot_full_normalization(save=save, show=show, do=do)
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        # print("Save/load!")
        self.save_curves(banner=False)
        self.load_curves()
        pass


class SRNSingleShotProcessor_Legacy(SRNSingleShotProcessor):
    name = filt_name = "Legacy SRN"
    description = "Single Shot SRN, Legacy Style"
    progress_verb = 'LSRNing'
    finished_verb = "LSRN Filtered"
    out_name = 'LSRN'
    show_plots = True
    
    def run_cdf(self):
        raise NotImplementedError
    
    def run_single(self):
        """modifies and uploads the image"""
        wave, t_rec = self.params.header['WAVELNTH'], self.params.header['T_OBS']
        data = self.params.raw_image
        image_meta = str(wave), str(wave), t_rec, data.shape
        self.params.modified_image = Legacy_SRN_Kernal(data, image_meta).get()
        return self.params.modified_image


class Legacy_SRN_Kernal:
    
    def __init__(self, data, image_data):
        """Initialize a new parameter object or use the provided one"""
        self.image_data = image_data
        self.name = "AWS Trial"
        self.renew_mask = True
        self.original = data
        self.changed = self.image_modify(data + 0)
        # self.plot_and_save()
    
    def get(self):
        return self.changed
        
    def image_modify(self, data):
        """Perform the image normalization on the input array"""
        
        data = self.radial_analyze(data, False)
        
        data = self.vignette(data)
        # data = self.absqrt(data)
        data = self.coronagraph(data)
        
        plotStats = False
        if plotStats:
            self.plot_stats()
        
        dat = data.astype('float32')
        # dat2 = self.renormalize(dat)
        # half = int(dat.shape[0]/2)
        # dat[:, :half] = dat2[:, :half]
        # dat[:, half:] = dat2[:, half:]
        # return dat
        
        return dat
    
    def radial_analyze(self, data, plotStats=False):
        self.offset = np.min(data)
        
        data -= self.offset
        
        self.make_radius(data)
        self.sort_radially(data)
        self.bin_radially()
        self.fit_curves()
        return data
    
    def make_radius(self, data):
        
        self.rez = data.shape[0]
        centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - centerPt, yy - centerPt
        
        self.extra_rez = 2
        
        self.sRadius = 400 * self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
        self.rez *= self.extra_rez
    
    def sort_radially(self, data):
        # Create arrays sorted by radius
        self.rad_flat = self.radius.flatten()
        self.dat_flat = data.flatten()
        inds = np.argsort(self.rad_flat)
        self.rad_sorted = self.rad_flat[inds]
        self.dat_sort = self.dat_flat[inds]
    
    def bin_radially(self):
        
        # Bin the intensities by radius
        self.radBins = [[] for x in np.arange(self.rez)]
        binInds = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)
        for ii, binI in enumerate(binInds):
            self.radBins[binI].append(self.dat_sort[ii])
        
        # Find the statistics by bin
        self.binMax = np.zeros(self.rez)
        self.binMin = np.zeros(self.rez)
        self.binMid = np.zeros(self.rez)
        self.binMed = np.zeros(self.rez)
        self.radAbss = np.arange(self.rez)
        
        for ii, it in enumerate(self.radBins):
            item = np.asarray(it)
            
            idx = np.isfinite(item)
            finite = item[idx]
            idx2 = np.nonzero(finite - self.offset)
            subItems = finite[idx2]
            
            if len(subItems) > 0:
                self.binMax[ii] = np.percentile(subItems, 75)  # np.nanmax(subItems)
                self.binMin[ii] = np.percentile(subItems, 2)  # np.min(subItems)
                self.binMid[ii] = np.mean(subItems)
                self.binMed[ii] = np.median(subItems)
            else:
                self.binMax[ii] = np.nan
                self.binMin[ii] = np.nan
                self.binMid[ii] = np.nan
                self.binMed[ii] = np.nan
        
        # Remove NANs
        idx = np.isfinite(self.binMax) & np.isfinite(self.binMin)
        self.binMax = self.binMax[idx]
        self.binMin = self.binMin[idx]
        self.binMid = self.binMid[idx]
        self.binMed = self.binMed[idx]
        self.radAbss = self.radAbss[idx]
    
    def fit_curves(self):
        # Input Stuff
        self.highCut = 0.8 * self.rez
        
        # Locate the Limb
        theMin = int(0.35 * self.rez)
        theMax = int(0.45 * self.rez)
        near_limb = np.arange(theMin, theMax)
        
        r1 = self.radAbss[np.argmax(self.binMid[near_limb]) + theMin]
        r2 = self.radAbss[np.argmax(self.binMax[near_limb]) + theMin]
        r3 = self.radAbss[np.argmax(self.binMed[near_limb]) + theMin]
        
        self.limb_radii = int(np.mean([r1, r2, r3]))
        # print(self.limb_radii)
        self.lCut = int(self.limb_radii - 0.01 * self.rez)
        self.hCut = int(self.limb_radii + 0.01 * self.rez)
        
        # Split into three regions
        self.low_abs = self.radAbss[:self.lCut]
        self.low_max = self.binMax[:self.lCut]
        self.low_min = self.binMin[:self.lCut]
        
        self.mid_abs = self.radAbss[self.lCut:self.hCut]
        self.mid_max = self.binMax[self.lCut:self.hCut]
        self.mid_min = self.binMin[self.lCut:self.hCut]
        
        self.high_abs = self.radAbss[self.hCut:]
        self.high_max = self.binMax[self.hCut:]
        self.high_min = self.binMin[self.hCut:]
        
        doPlot = False
        if doPlot:
            # plt.axvline(r1, c='g')
            # plt.axvline(r2, c='g')
            # plt.axvline(r3, c='g')
            
            plt.plot(self.radAbss, self.binMax, label="Max")
            plt.plot(self.radAbss, self.binMin, label="Min")
            plt.plot(self.radAbss, self.binMid, label="Mid")
            plt.plot(self.radAbss, self.binMed, label="Med")
            
            plt.axvline(theMin)
            plt.axvline(theMax)
            
            plt.axvline(self.limb_radii)
            plt.axvline(self.lCut, ls=':')
            plt.axvline(self.hCut, ls=':')
            plt.xlim([self.lCut, self.hCut])
            plt.legend()
            plt.show()
        
        # Filter the regions separately
        
        lWindow = 7  # 4 * self.extra_rez + 1
        mWindow = 7  # 4 * self.extra_rez + 1
        hWindow = 7  # 30 * self.extra_rez + 1
        fWindow = 7  # int(3 * self.extra_rez) + 1
        rank = 3
        
        # print(self.count_nan(self.throw_nan(self.low_max)))
        mode = 'nearest'
        low_max_filt = savgol_filter(self.low_max, lWindow, rank, mode=mode)
        #
        
        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank, mode=mode)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        
        high_max_filt = savgol_filter(self.high_max, hWindow, rank, mode=mode)
        
        low_min_filt = savgol_filter(self.low_min, lWindow, rank, mode=mode)
        mid_min_filt = savgol_filter(self.mid_min, mWindow, rank, mode=mode)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank, mode=mode)
        
        #
        # Fit the low curves
        degree = 5
        p = np.polyfit(self.low_abs, low_max_filt, degree)
        low_max_fit = np.polyval(p, self.low_abs)  # * 1.1
        p = np.polyfit(self.low_abs, low_min_filt, degree)
        low_min_fit = np.polyval(p, self.low_abs)
        
        ind = 10
        low_max_fit[0:ind] = low_max_fit[ind]
        low_min_fit[0:ind] = low_min_fit[ind]
        
        doPlot = False
        if doPlot:
            plt.plot(self.low_abs, low_max_filt, lw=4)
            plt.plot(self.mid_abs, mid_max_filt, lw=4)
            plt.plot(self.high_abs, high_max_filt, lw=4)
            
            plt.plot(self.radAbss, self.binMax, label="Max")
            
            plt.plot(self.low_abs, low_min_filt, lw=4)
            plt.plot(self.mid_abs, mid_min_filt, lw=4)
            plt.plot(self.high_abs, high_min_filt, lw=4)
            
            plt.plot(self.radAbss, self.binMin, label="Min")
            
            plt.plot(self.low_abs, low_min_fit, c='k')
            plt.plot(self.low_abs, low_max_fit, c='k')
            
            # plt.plot(self.radAbss, self.binMid, label="Mid")
            # plt.plot(self.radAbss, self.binMed, label="Med")
            
            # plt.xlim([0.6*theMin,theMax*1.5])
            
            plt.legend()
            plt.show()
        
        # Build output curves
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax0 = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin0 = np.hstack((low_min_fit, mid_min_filt, high_min_filt))
        
        # Filter again to smooth boundaraies
        self.fakeMax0 = self.fill_end(self.fill_start(savgol_filter(self.fakeMax0, fWindow, rank)))
        self.fakeMin0 = self.fill_end(self.fill_start(savgol_filter(self.fakeMin0, fWindow, rank)))
        
        # Put the nans back in
        self.fakeMax = np.empty(self.rez)
        self.fakeMax.fill(np.nan)
        self.fakeMin = np.empty(self.rez)
        self.fakeMin.fill(np.nan)
        
        self.fakeMax[self.fakeAbss] = self.fakeMax0
        self.fakeMin[self.fakeAbss] = self.fakeMin0
        # plt.plot(np.arange(self.rez), self.fakeMax)
        # plt.plot(np.arange(self.rez), self.fakeMin)
        # plt.show()
        
        # # Locate the Noise Floor
        # noiseMin = 550 * self.extra_rez - self.hCut
        # near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        # self.diff_max_abs = self.high_abs[near_noise]
        # self.diff_max = np.diff(high_max_filt)[near_noise]
        # self.diff_max += np.abs(np.nanmin(self.diff_max))
        # self.diff_max /= np.nanmean(self.diff_max) / 100
        # self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        # self.noise_radii = 565 * self.extra_rez
    
    def vignette(self, data):
        
        mask = self.radius > (int(1.1 * self.rez // 2))  # (3.5 * self.noise_radii)
        data[mask] = np.nan
        return data
    
    def coronaNorm(self, data):
        data[data == 0] = np.nan
        
        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        flat_data = data.flatten()
        
        the_min = self.fakeMin[radius_bin]
        # plt.plot(self.fakeMin)
        # plt.show()
        # import pdb; pdb.set_trace()
        # the_min = np.asarray([self.fakeMin[r] for r in radius_bin])
        # print('d3', the_min)
        the_max = self.fakeMax[radius_bin]
        # the_max = np.asarray([self.fakeMax[r] for r in radius_bin])
        
        # the_max = self.fakeMax[radius_bin]
        top = bottom = dat_corona = np.ones_like(flat_data)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                
                top = np.subtract(flat_data, the_min)
                bottom = np.subtract(the_max, the_min)
                
                dat_corona = np.divide(top, bottom)
            except RuntimeWarning as e:
                pass
        
        return dat_corona
    
    def coronagraph(self, data):
        
        dat_corona = self.coronaNorm(data)
        
        # sys.stderr = original
        
        # Deal with too hot things
        self.vmax = 0.95
        self.vmax_plot = 0.85  # np.max(dat_corona)
        hotpowr = 1 / 1.5
        
        hot = dat_corona > self.vmax
        # dat_corona[hot] = dat_corona[hot] ** hotpowr
        
        # Deal with too cold things
        self.vmin = 0.3
        self.vmin_plot = -0.05  # np.min(dat_corona)# 0.3# -0.03
        coldpowr = 1 / 2
        
        cold = dat_corona < self.vmin
        dat_corona[cold] = -((np.abs(dat_corona[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin
        
        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)
        
        if self.renew_mask:
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False
        dat_corona_square = np.sign(dat_corona_square) * np.power(np.abs(dat_corona_square), (1 / 5))
        data = self.normalize(data, high=100, low=0)
        dat_corona_square = self.normalize(dat_corona_square, high=100, low=1)
        
        #
        do_mirror = False
        if do_mirror:
            # Do stuff
            xx, yy = self.corona_mask.shape[0], int(self.corona_mask.shape[1] / 2)
            #
            newDat = data[self.corona_mask]
            grid = newDat.reshape(xx, yy)
            # if self.
            flipped = np.fliplr(grid)
            data[~self.corona_mask] = flipped.flatten()  # np.flip(newDat)
        
        data[self.corona_mask] = dat_corona_square[self.corona_mask]
        # print(data.dtype)
        #
        # inds = np.argsort(self.rad_flat)
        # rad_sorted = self.rad_flat[inds]
        # dat_sort = dat_corona[inds]
        #
        # plt.figure()
        # # plt.yscale('log')
        # plt.scatter(rad_sorted[::30], dat_sort[::30], c='k')
        # plt.show()
        
        # data = data / np.mean(data)
        
        # data = data**(1/2)
        # data = np.log(data)
        
        # data = self.normalize(data, high=85, low=5)
        
        return data
    
    def get_mask(self, dat_out):
        
        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)
        
        mode = 'y'
        
        if type(mode) in [float, int]:
            mask_num = mode
        elif 'y' in mode:
            mask_num = 1
        elif 'n' in mode:
            mask_num = 2
        else:
            if 'r' in mode:
                if len(mode) < 2:
                    mode += 'a'
            
            if 'a' in mode:
                top = 8
                btm = 1
            elif 'h' in mode:
                top = 6
                btm = 3
            elif 'd' in mode:
                top = 8
                btm = 7
            elif 'w' in mode:
                top = 2
                btm = 1
            else:
                print('Unrecognized Mode')
                top = 8
                btm = 1
            
            ii = 0
            while True:
                mask_num = np.random.randint(btm, top + 1)
                if mask_num not in self.mask_num:
                    self.mask_num.append(mask_num)
                    break
                ii += 1
                if ii > 10:
                    self.mask_num = []
        
        if mask_num == 1:
            corona_mask[:, :] = True
        
        if mask_num == 2:
            corona_mask[:, :] = False
        
        if mask_num == 3:
            corona_mask[half:, :] = True
        
        if mask_num == 4:
            corona_mask[:half, :] = True
        
        if mask_num == 5:
            corona_mask[:, half:] = True
        
        if mask_num == 6:
            corona_mask[:, :half] = True
        
        if mask_num == 7:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
        
        if mask_num == 8:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
            corona_mask = np.invert(corona_mask)
        
        return corona_mask
    
    def plot_stats(self):
        
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex="True")
        ax0.scatter(self.n2r(self.rad_sorted[::30]), self.dat_sort[::30], c='k', s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        # ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=':')
        ax0.axvline(self.n2r(self.hCut), ls=':')
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))
        
        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_mean, 'r:')
        
        ax0.plot(self.n2r(self.low_abs), self.low_max, 'm', label="Percentile")
        ax0.plot(self.n2r(self.low_abs), self.low_min, 'm')
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')
        
        ax0.plot(self.n2r(self.high_abs), self.high_max, 'c', label="Percentile")
        ax0.plot(self.n2r(self.high_abs), self.high_min, 'c')
        
        ax0.plot(self.n2r(self.mid_abs), self.mid_max, 'y', label="Percentile")
        ax0.plot(self.n2r(self.mid_abs), self.mid_min, 'y')
        # plt.plot(self.high_abs, self.high_min_fit, 'r')
        # plt.plot(self.high_abs, self.high_max_fit, 'r')
        
        # try:
        #     ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        # except:
        #     ax0.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax0.plot(self.n2r(self.radAbss), self.fakeMin, 'g')
        
        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMid, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        # ax0.set_xlim((0, self.n2r(self.highCut)))
        ax0.legend()
        fig.set_size_inches((8, 12))
        ax0.set_yscale('log')
        
        ax1.scatter(self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c='k', s=2)
        ax1.set_ylim((-0.25, 2))
        
        ax1.axhline(self.vmax, c='r', label='Confinement')
        ax1.axhline(self.vmin, c='r')
        ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        ax1.axhline(self.vmin_plot, c='orange')
        
        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))
        
        ax1.legend()
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        plt.tight_layout()
        doPlot = False
        if doPlot:  # self.params.is_debug():
            file_name = '{}_Radial.png'.format(self.name)
            # print("Saving {}".format(file_name))
            # save_path = join(r"data\images\radial", file_name)
            # plt.savefig(save_path)
            
            file_name = '{}_Radial_zoom.png'.format(self.name)
            ax0.set_xlim((0.9, 1.1))
            # save_path = join(r"data\images\radial", file_name)
            # plt.savefig(save_path)
            # plt.show()
            plt.close(fig)
        else:
            plt.show()
    
    def n2r(self, n):
        return n / self.limb_radii
    
    def fill_end(self, use):
        iii = -1
        val = use[iii]
        while np.isnan(val):
            iii -= 1
            val = use[iii]
        use[iii:] = val
        return use
    
    def fill_start(self, use):
        iii = 0
        val = use[iii]
        while np.isnan(val):
            iii += 1
            val = use[iii]
        use[:iii] = val
        return use
    
    @staticmethod
    def normalize(data, high=98, low=15):
        if low is None:
            lowP = 0
        else:
            lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                out = (data - lowP) / (highP - lowP)
            except RuntimeWarning as e:
                out = data
        return out
    
    def plot_and_save(self):
        
        self.render()
        
        self.export_files()
    
    def render(self):
        """Generate the plots"""
        data = self.changed
        original_data = self.original
        
        full_name, save_path, time_string, ii = self.image_data
        time_string2 = self.clean_time_string(time_string)
        name, wave = self.clean_name_string(full_name)
        
        self.figbox = []
        for processed in [False, True]:
            
            # Create the Figure
            fig, ax = plt.subplots()
            self.blankAxis(ax)
            fig.set_facecolor("k")
            
            self.inches = 10
            fig.set_size_inches((self.inches, self.inches))
            
            if 'hmi' in name.casefold():
                inst = ""
                plt.imshow(data, origin='upper', interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05
            
            else:
                
                # from color_tables import aia_wave_dict
                # aia_wave_dict(wave)
                
                inst = '  AIA'
                cmap = 'sdoaia{}'.format(wave)
                cmap = aia_color_table(int(wave) * u.angstrom)
                if processed:
                    plt.imshow(data, cmap=cmap, origin='lower', interpolation=None, vmin=self.vmin_plot, vmax=self.vmax_plot)
                else:
                    toprint = self.normalize(self.absqrt(original_data))
                    # plt.imshow(toprint, cmap='sdoaia{}'.format(wave), origin='lower', interpolation=None) #,  vmin=self.vmin_plot, vmax=self.vmax_plot)
                    
                    plt.imshow(self.absqrt(original_data), cmap=cmap, origin='lower', interpolation=None)  # ,  vmin=self.vmin_plot, vmax=self.vmax_plot)
                
                plt.tight_layout(pad=0)
                height = 0.95
            
            # Annotate with Text
            buffer = '' if len(name) == 3 else '  '
            buffer2 = '    ' if len(name) == 2 else ''
            
            title = "{}    {} {}, {}{}".format(buffer2, inst, wave, time_string2, buffer)
            ax.annotate(title, (0.15, height + 0.02), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')
            # title2 = "{} {}, {}".format(inst, name, time_string2)
            # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
            the_time = strftime("%Z %I:%M%p")
            if the_time[0] == '0':
                the_time = the_time[1:]
            ax.annotate(the_time, (0.15, height), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')
            
            # Format the Plot and Save
            self.blankAxis(ax)
            # plt.show()
            self.figbox.append([fig, ax, processed])
            # plt.show()
    
    def export(self):
        full_name, save_path, time_string, ii = self.image_data
        pixels = self.changed.shape[0]
        dpi = pixels / self.inches
        try:
            self.img_box = []
            for fig, ax, processed in self.figbox:
                # middle = '' if processed else "_orig"
                #
                # new_path = save_path[:-5] + middle + ".png"
                # name = self.clean_name_string(full_name)
                # directory = "renders/"
                # path = directory + new_path.rsplit('/')[1]
                # os.makedirs(directory, exist_ok=True)
                # plt.close(fig)
                # self.newPath = path
                
                # Image from plot
                ax.axis('off')
                fig.tight_layout(pad=0)
                # To remove the huge white borders
                ax.margins(0)
                ax.set_facecolor('k')
                
                fig.canvas.draw()
                
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                self.img_box.append(image_from_plot)
                # fig.savefig(path, facecolor='black', edgecolor='black', dpi=dpi)
                # print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))
        except Exception as e:
            raise e
        finally:
            for fig, ax, processed in self.figbox:
                plt.close(fig)
    
    def export_files(self):
        full_name, save_path, time_string, ii = self.image_data
        pixels = self.changed.shape[0]
        dpi = pixels / self.inches
        self.pathBox = []
        try:
            for fig, ax, processed in self.figbox:
                middle = '' if processed else "_orig"
                
                name, wave = self.clean_name_string(full_name)
                new_path = save_path[:-5] + name + middle + ".png"
                directory = "renders/"
                path = directory + new_path
                os.makedirs(directory, exist_ok=True)
                fig.savefig(path, facecolor='black', edgecolor='black', dpi=dpi)
                # print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))
                self.pathBox.append(path)
        except Exception as e:
            raise e
        finally:
            for fig, ax, processed in self.figbox:
                plt.close(fig)
            if False:
                self.save_concatinated()
    
    def save_concatinated(self):
        name = self.pathBox[1][:-4] + "_cat.png"
        fmtString = "ffmpeg -i {} -i {} -y -filter_complex hstack {} -hide_banner -loglevel warning"
        os.system(fmtString.format(self.pathBox[1], self.pathBox[0], name))
    
    # def export_files2(self):
    #     full_name, save_path, time_string, ii = self.image_data
    #     pixels = self.changed.shape[0]
    #     dpi = pixels / self.inches
    #     paths = []
    #     try:
    #         for fig, ax, processed in self.figbox:
    #             middle = '' if processed else "_orig"
    #
    #             new_path = save_path[:-5] + middle + ".png"
    #             name = self.clean_name_string(full_name)
    #             directory = "renders/"
    #             path = directory + new_path.rsplit('/')[1]
    #             os.makedirs(directory, exist_ok=True)
    #             self.newPath = path
    #             fig.savefig(path, facecolor='black', edgecolor='black', dpi=dpi)
    #             print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))
    #             paths.append(path)
    #
    #     except Exception as e:
    #         raise e
    #     finally:
    #         for fig, ax, processed in self.figbox:
    #             plt.close(fig)
    
    def get_figs(self):
        return self.figbox
    
    def get_imgs(self):
        return self.img_box
    
    def get_paths(self):
        return self.pathBox
    
    @staticmethod
    def blankAxis(ax):
        ax.patch.set_alpha(0)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', which='both',
                       top=False, bottom=False, left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    @staticmethod
    def clean_name_string(full_name):
        digits = ''.join(i for i in full_name if i.isdigit())
        # Make the name strings
        name = digits + ''
        digits = "{:04d}".format(int(name))
        # while name[0] == '0':
        #     name = name[1:]
        return digits, name
    
    @staticmethod
    def clean_time_string(time_string):
        # Make the name strings
        
        cleaned = datetime.datetime.strptime(time_string[:-4], "%Y-%m-%dT%H:%M:%S")
        # cleaned += timedelta(hours=-7)
        
        # tz = timezone(timedelta(hours=-1))
        # import pdb; pdb.set_trace()
        # cleaned = time_string.replace(tzinfo=timezone.utc).astimezone(tz=None)
        # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime("%I:%M%p, %b-%d, %Y")
        # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=tz).strftime("%I:%M%p, %b-%d, %Y")
        # cleaned = Time(time_string).datetime.strftime("%I:%M%p, %b-%d, %Y")
        # print("----------->", cleaned)
        # import pdb; pdb.set_trace()
        return cleaned.strftime("%m-%d-%Y %I:%M%p")
        # name = full_name + ''
        # while name[0] == '0':
        #     name = name[1:]
        # return name
    
    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))



#
# class SRNpreProcessor_Legacy(SRNProcessor):
#     """Analyzes the whole dataset and builds curves"""
#     out_name = None
#     name = filt_name = 'SRN Pre-Processor'
#     description = "Create the Radial SRN Curves"
#     progress_verb = 'Analyzing'
#     finished_verb = "Analyzed"
#     show_plots = True
#
#     def __init__(self, fits_path=None, in_name="t_int", orig=False,
#                  show=False, verb=False, quick=False, rp=None, params=None):
#         super().__init__(fits_path=fits_path, in_name=in_name, orig=orig, show=show, verb=verb, quick=quick, rp=rp, params=params)
#         self.first = True
#         self.go_ahead = True
#
#     def setup(self):
#         self.load()
#         self.print_keyframes()
#         self.skipped = 0
#
#     def do_work(self):
#         """Analyze the Image, Normalize it, Plot"""
#         if self.should_run():
#             self.image_learn()
#             # self.plot_norm_curves(save=True)
#         # self.out_name = "quantile"
#         return self.params.quantile_image
#
#     def cleanup(self):
#         """Runs after all the images have been modified with do_work"""
#         if self.should_run():
#             self.skipped -= 1
#             self.make_save_smoothed_curves(banner=False)  # Build smooth curves based on the statistics
#         self.render_pre_hist_video()
#         # print("Curves Saved!")
#
#     def render_pre_hist_video(self):
#         fps = 8
#         os.makedirs(self.params.base_directory(), exist_ok=True)
#         print("Rendering pre-processor video...", end='')
#         path1 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\a-pre-hist.avi")
#         self.write_video_in_directory(fullpath=path1, fps=fps, key_string="inner", destroy=False, pop=2)
#
#         # path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist_pre\\{}_inner_outer_{}.avi".format(self.params.current_wave(), time()))
#         # path2 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\zoom\\{}_zoom_{}.avi".format(self.params.current_wave(), time()))
#
#         # directory1 = os.path.dirname(path1)
#         # name1 = os.path.basename(path1)
#         # directory2 = os.path.dirname(path2)
#         # name2 = os.path.basename(path2)
#
#         # self.write_video_in_directory(fullpath=path2, fps=fps, key_string="zoom" , destroy=False)
#
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         print("Success!")
#
#     def should_run(self):
#         """Decide of the processor should run on this file"""
#         self.can_use_keyframes = True
#         not_dark = self.header["IMG_TYPE"] == "LIGHT"
#         not_weak = self.header["EXPTIME"] >= 1.0
#         set_to_make = self.params.remake_norm_curves() or self.reprocess_mode()
#         not_made_yet = not os.path.exists(self.params.curve_path()) or self.outer_min is None
#         frame_is_not_loaded = self.params.raw_image is None
#         self.go_ahead = not_weak & not_dark and (set_to_make or not_made_yet or frame_is_not_loaded)
#         return self.go_ahead
#
#     # def delete_temp_folder(self, folder):
#     #     if os.path.isdir(folder):
#     #         shutil.rmtree(folder)
#     #
#     # def delete_temp_folder_items(self, folder):
#     #     for root, dirs, files in os.walk(folder):
#     #         for file in files:
#     #             self.force_delete(file, root)
#
#     @staticmethod
#     def force_delete(file, root='', do=True):
#         if do:
#             if not os.path.isdir(file):
#                 os.remove(os.path.join(root, file))
#             else:
#                 shutil.rmtree(file)
#
#
# class SRNradialFiltProcessor_Legacy(SRNProcessor):
#     """Uses radial curves to normalize images"""
#     name = out_name = 'SRN'
#     filt_name = 'SRN Radial Filter'
#     description = "Filter the Images Radially with SRN"
#     progress_verb = 'Filtering'
#     progress_unit = 'Images'
#     finished_verb = "Filtered"
#
#     def __init__(self, fits_path=None, in_name=-1, orig=False,
#                  show=False, verb=False, quick=False, rp=None, params=None):
#         super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
#         self.show_norm = False
#         self.first = True
#         self.go_ahead = True
#         self.can_use_keyframes = False
#
#     def setup(self):
#         self.super_flush()
#         self.load_curves()
#
#     def do_work(self):
#         self.image_modify()
#         # self.peek_norm()
#         self.show_norm = False
#         self.plot_full_normalization(True, show=self.show_norm, save=True)
#         self.percentilize()
#         return self.params.modified_image
#
#     def cleanup(self):
#         """Runs after all the images have been modified with do_work"""
#         self.render_post_hist_video()
#         print(" ^ Filter Applied Successfully", flush=True)
#
#     def render_post_hist_video(self):
#         print("Rendering post-processor video...", end='')
#         fps = 8
#         os.makedirs(self.params.base_directory(), exist_ok=True)
#         path1 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_post\\b-post-hist.avi")
#         self.write_video_in_directory(fullpath=path1, fps=fps, destroy=False, pop=2)
#
#         # path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist_pre\\{}_inner_outer_{}.avi".format(self.params.current_wave(), time()))
#         # path2 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\zoom\\{}_zoom_{}.avi".format(self.params.current_wave(), time()))
#
#         # directory1 = os.path.dirname(path1)
#         # name1 = os.path.basename(path1)
#         # directory2 = os.path.dirname(path2)
#         # name2 = os.path.basename(path2)
#
#         # self.write_video_in_directory(fullpath=path2, fps=fps, key_string="zoom" , destroy=False)
#
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         print("Success!")



# # ~~~~~
#
# class SRNpreProcessor(SRNProcessor):
#     """Analyzes the whole dataset and builds curves"""
#     out_name = None
#     name = filt_name = 'SRN Pre-Processor'
#     description = "Create the Radial SRN Curves"
#     progress_verb = 'Analyzing'
#     finished_verb = "Analyzed"
#     show_plots = True
#
#     def __init__(self, fits_path=None, in_name="t_int", orig=False,
#                  show=False, verb=False, quick=False, rp=None, params=None):
#         super().__init__(fits_path=fits_path, in_name=in_name, orig=orig, show=show, verb=verb, quick=quick, rp=rp, params=params)
#         self.first = True
#         self.go_ahead = True
#
#     def setup(self):
#         self.load()
#         self.print_keyframes()
#         self.skipped = 0
#
#     def do_work(self):
#         """Analyze the Image, Normalize it, Plot"""
#         if self.should_run():
#             self.image_learn()
#             # self.plot_norm_curves(save=True)
#         # self.out_name = "quantile"
#         return self.params.quantile_image
#
#     def cleanup(self):
#         """Runs after all the images have been modified with do_work"""
#         if self.should_run():
#             self.skipped -= 1
#             self.make_save_smoothed_curves(banner=False)  # Build smooth curves based on the statistics
#         self.render_pre_hist_video()
#         # print("Curves Saved!")
#
#     def render_pre_hist_video(self):
#         fps = 8
#         os.makedirs(self.params.base_directory(), exist_ok=True)
#         print("Rendering pre-processor video...", end='')
#         path1 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\a-pre-hist.avi")
#         self.write_video_in_directory(fullpath=path1, fps=fps, key_string="inner", destroy=False, pop=2)
#
#         # path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist_pre\\{}_inner_outer_{}.avi".format(self.params.current_wave(), time()))
#         # path2 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\zoom\\{}_zoom_{}.avi".format(self.params.current_wave(), time()))
#
#         # directory1 = os.path.dirname(path1)
#         # name1 = os.path.basename(path1)
#         # directory2 = os.path.dirname(path2)
#         # name2 = os.path.basename(path2)
#
#         # self.write_video_in_directory(fullpath=path2, fps=fps, key_string="zoom" , destroy=False)
#
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         print("Success!")
#
#     def should_run(self):
#         """Decide of the processor should run on this file"""
#         self.can_use_keyframes = True
#         not_dark = self.header["IMG_TYPE"] == "LIGHT"
#         not_weak = self.header["EXPTIME"] >= 1.0
#         set_to_make = self.params.remake_norm_curves() or self.reprocess_mode()
#         not_made_yet = not os.path.exists(self.params.curve_path()) or self.outer_min is None
#         frame_is_not_loaded = self.params.raw_image is None
#         self.go_ahead = not_weak & not_dark and (set_to_make or not_made_yet or frame_is_not_loaded)
#         return self.go_ahead
#
#     # def delete_temp_folder(self, folder):
#     #     if os.path.isdir(folder):
#     #         shutil.rmtree(folder)
#     #
#     # def delete_temp_folder_items(self, folder):
#     #     for root, dirs, files in os.walk(folder):
#     #         for file in files:
#     #             self.force_delete(file, root)
#
#     @staticmethod
#     def force_delete(file, root='', do=True):
#         if do:
#             if not os.path.isdir(file):
#                 os.remove(os.path.join(root, file))
#             else:
#                 shutil.rmtree(file)
#
#
# class SRNradialFiltProcessor(SRNProcessor):
#     """Uses radial curves to normalize images"""
#     name = out_name = 'SRN'
#     filt_name = 'SRN Radial Filter'
#     description = "Filter the Images Radially with SRN"
#     progress_verb = 'Filtering'
#     progress_unit = 'Images'
#     finished_verb = "Filtered"
#
#     def __init__(self, fits_path=None, in_name=-1, orig=False,
#                  show=False, verb=False, quick=False, rp=None, params=None):
#         super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
#         self.show_norm = False
#         self.first = True
#         self.go_ahead = True
#         self.can_use_keyframes = False
#
#     def setup(self):
#         self.super_flush()
#         self.load_curves()
#
#     def do_work(self):
#         self.image_modify()
#         # self.peek_norm()
#         self.show_norm = False
#         self.plot_full_normalization(True, show=self.show_norm, save=True)
#         self.percentilize()
#         return self.params.modified_image
#
#     def cleanup(self):
#         """Runs after all the images have been modified with do_work"""
#         self.render_post_hist_video()
#         print(" ^ Filter Applied Successfully", flush=True)
#
#     def render_post_hist_video(self):
#         print("Rendering post-processor video...", end='')
#         fps = 8
#         os.makedirs(self.params.base_directory(), exist_ok=True)
#         path1 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_post\\b-post-hist.avi")
#         self.write_video_in_directory(fullpath=path1, fps=fps, destroy=False, pop=2)
#
#         # path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist_pre\\{}_inner_outer_{}.avi".format(self.params.current_wave(), time()))
#         # path2 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\zoom\\{}_zoom_{}.avi".format(self.params.current_wave(), time()))
#
#         # directory1 = os.path.dirname(path1)
#         # name1 = os.path.basename(path1)
#         # directory2 = os.path.dirname(path2)
#         # name2 = os.path.basename(path2)
#
#         # self.write_video_in_directory(fullpath=path2, fps=fps, key_string="zoom" , destroy=False)
#
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         # self.delete_temp_folder_items(os.path.dirname(path1))
#         print("Success!")

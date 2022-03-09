import os
import shutil
from copy import copy
from os import makedirs
from os.path import join, dirname, basename
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import stats

from processor.Processor import Processor

import warnings

from science.color_tables import aia_color_table
import astropy.units as u

warnings.filterwarnings("ignore")
# import matplotlib as mpl

# try:
#     mpl.use("qt5agg")
# except ImportError as e:
#     print(e)
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


class QRNProcessor(Processor):
    """This class holds the code for the Quantile Radial Norm Processor"""
    name = filt_name = "Quantile Norm"
    description = "Apply the single-shot Quantile Norm to images"
    progress_verb = 'Filtering'
    finished_verb = "Radially Filtered"
    out_name = "QRN"
    
    # Flags
    show_plots = True
    do_png = False
    renew_mask = True
    can_initialize = True
    
    # Parse Inputs
    def __init__(self, fits_path=None, in_name=None, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp)

        self.shrink_factor = 1
        self.radius = None
        # Ingest
        self.fits_path = fits_path
        self.in_name = in_name or self.params.master_frame_list_newest
        self.show = show
        self.verb = verb
        self.do_orig = orig
        
        # Parameters
        self.make_curves_latch = True  # This Recomputes the curves once
        self.floor = 0.01
        self.s_radius = 400
        self.can_use_keyframes = False

        ### Initializations ###
        
        # Flags
        self.first = True
        self.go_ahead = True
        self.first = True
        self.there_is_cached_data = False
        self.do_save = None
        self.outputs_initialized = False
        self.dont_ignore = True
        
        # Data
        self.image_data = None
        self.outer_min = None
        self.inner_min = None
        self.inner_max = None
        self.outer_max = None
        self.avg_min = None
        self.avg_max = None
        self.rad_flat = None
        self.bin_rez = None
        self.found_limb_radius = None  # 1600
        self.rendered_abss = None
        self.norm_avg_max = None
        self.norm_avg_min = None
        self.multiple_minimum_curves = []
        self.multiple_maximum_curves = []
        self.rendered_min_box = []
        self.rendered_max_box = []
        self.radBins_all = []
        self.RN = None
        self.norm_curve_max_top_name = None
        self.norm_curve_min_top_name = None
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
        self.binfactor = 1
        self.this_index = 0
        self.n_keyframes = 0
        self.firstIndex = 0
        self.lastIndex = 0
        self.output_abscissa = None
        self.abs_max_scalar = None
        self.abs_min_scalar = None
        self.curve_out_array = None
        self.scalar_in_curve = None
        self.rastered_outer_min = None
        self.rastered_inner_min = None
        self.rastered_inner_max = None
        self.rastered_outer_max = None
        self.scalar_out_curve = None
        
        self.mirror_mask = None
        self.grid_mask = None
        self.t_factor = None
        self.tRadius = None

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
        self.cut_pixels = None
    
    def setup(self):
        self.load()
        self.print_keyframes()
        self.skipped = 0
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.should_run():
            self.image_learn()
        return self.params.modified_image
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        if self.should_run():
            self.skipped -= 1
            self.skipped = max((self.skipped, 0))
   
    def should_run(self):
        """Decide of the processor should run on this file"""
        if not self.header:
            print("No header Loaded")
            return False
        self.can_use_keyframes = True
        not_dark = self.header["IMG_TYPE"] == "LIGHT"
        not_weak = self.header["EXPTIME"] >= 0.9
        set_to_make = self.params.remake_norm_curves() or self.reprocess_mode()
        not_made_yet = not os.path.exists(self.params.curve_path()) or self.outer_min is None
        frame_is_not_loaded = self.params.raw_image is None
        self.go_ahead = not_weak & not_dark and (set_to_make or not_made_yet or frame_is_not_loaded)
        if not self.go_ahead:
            problem = "Weak" if not not_weak else "dark" if not not_dark else "incomprehensible"
            print("Skipping because the file is {}".format(problem))
        return self.go_ahead
    
    # def delete_temp_folder(self, folder):
    #     if os.path.isdir(folder):
    #         shutil.rmtree(folder)
    #
    # def delete_temp_folder_items(self, folder):
    #     for root, dirs, files in os.walk(folder):
    #         for file in files:
    #             self.force_delete(file, root)
    
    @staticmethod
    def force_delete(file, root='', do=True):
        if do:
            if not os.path.isdir(file):
                os.remove(os.path.join(root, file))
            else:
                shutil.rmtree(file)
    
    ###################
    ##   Main Calls  ##
    ###################
    
    # def do_img_function(self):
    #     """Calls the do_work function on a single fits path if indicated"""
    #     raise NotImplementedError
    #     # return self.do_work()  # Do the work on the fits files
    
    ###################
    ## Top-Level ##
    ###################
    
    def image_learn(self):
        """Analyze the input image_path to help make normalization curves"""
        if not self.skip_bad_frame():
            self.init_for_learn()
            self.coronaLearn()
            self.add_to_keyframes()  # Update the running curves
    
    
    ###################
    ## Keyframes ##
    ###################
    
    def add_to_keyframes(self):
        """Records the current analysis as one of the radial samples"""
        self.update_keyframe_counters()

    
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
        self.init_modified_image()
        self.init_radius_array()
        # self.init_frame_curves()
        self.init_statistics()
        
    def init_modified_image(self): ## TODO POTENTIAL BREAK POINT
        # mod = self.params.modified_image
        # if mod is None or not mod:
        self.params.modified_image = np.zeros_like(self.params.raw_image)
            
    def init_radius_array(self, vignette_radius=1.19, s_radius=400, t_factor=1.28, force=False):
        """Build an r-coordinate array of shape(in_object)"""
        
        self.params.center = [self.header["X0_MP"], self.header["Y0_MP"]]
        self.found_limb_radius = self.fit_limb_radius = self.header["R_SUN"]
        self.params.rez = self.header["NAXIS1"]
        
        nn = 1
        while self.found_limb_radius > self.params.rez/2:
            nn *= 2
            self.found_limb_radius = self.fit_limb_radius = self.header["R_SUN"] / nn
            self.params.center = [self.header["X0_MP"]/nn, self.header["Y0_MP"]/nn]
        
        self.shrink_factor = nn
        self.output_abscissa = np.arange(self.params.rez)
        
        xx, yy = np.meshgrid(np.arange(self.params.rez), np.arange(self.params.rez))
        xc, yc = xx - self.params.center[0], yy - self.params.center[1]
        self.radius = np.sqrt(xc * xc + yc * yc)
        self.rad_flat = self.radius.flatten()
        
        
        self.binfactor = binfactor = 2
        self.binInds = np.asarray(binfactor * np.floor(self.rad_flat // binfactor), dtype=np.int32)
        # self.make_annular_rings()
        # self.binInds = np.digitize(self.rad_flat, self.RN)
        
        self.binXX = xx.flatten()
        self.binYY = yy.flatten()
        self.binII = np.arange(len(self.rad_flat))
        self.vcut = int(vignette_radius * self.params.rez // 2)
        self.vrad = self.n2r(self.vcut)
        self.vignette_mask = np.asarray(self.radius > self.vcut, dtype=bool)
        self.s_radius = s_radius
        self.tRadius = self.s_radius * t_factor
 
    def init_statistics(self):
        """Initialize the statistical arrays"""
        dprint("init_statistics")
        
        self.bin_rez = np.max(self.binInds) + 10
        self.radBins = [[] for x in np.arange(self.bin_rez)]
        self.radBins_xy = [[] for x in np.arange(self.bin_rez)]
        self.radBins_ind = [[] for x in np.arange(self.bin_rez)]
    
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
        flat_raw = self.params.raw_image2.Fflatten() + 0
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
                                          self.params.raw_image.flatten()[::self.cut_pixels],
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
        # self.params.modified_image = copy(self.params.raw_image.flatten())
        self.params.mod_flat =self.params.modified_image.flatten()
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
            self.params.mod_flat[good_coord] = quantileized

    
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
        use_shape = self.params.raw_image.shape
        self.params.modified_image = self.params.mod_flat.reshape(use_shape)
        
        # from utils.stretch_intensity_module import norm_stretch
        # self.params.modified_image = norm_stretch(self.params.modified_image, alpha=self.params.alpha)

        # plt.imshow(self.params.modified_image)
        # plt.show(block=True)
    

    
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
    

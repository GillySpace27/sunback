import os
from os import makedirs
from os.path import join, dirname
import numpy as np
from scipy.signal import savgol_filter
from processor.Processor import Processor

import warnings

warnings.filterwarnings("ignore")
import matplotlib as mpl

try:
    mpl.use("qt5agg")
except ImportError as e:
    print(e)
import matplotlib.pyplot as plt

plt.ioff()

do_dprint = False


def dprint(txt, **kwargs):
    if do_dprint:
        print(txt, **kwargs)


class SRNProcessor(Processor):
    """This is the primary code used in the RadialFiltProcessor"""
    
    name = "Default"
    filt_name = '  SRN Radial Base Class'
    description = "Create and Apply the Radial SRN Curves"
    out_name = None
    do_png = False
    renew_mask = True
    show_plots = True
    can_initialize= True
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
    
    def __init__(self, fits_path=None, in_name=-1, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp)
        # Parse Inputs

        self.smoothed_abs_max = None
        self.smoothed_abs_min = None
        self.abs_min = None
        self.abs_max = None
        self.first = True
        self.lCut = None
        self.smooth_outer_maximum = None
        self.smooth_inner_maximum = None
        self.smooth_inner_minimum = None
        self.smooth_outer_minimum = None
        self.smoothed_frame_maximum = None
        self.smoothed_frame_minimum = None
        self.filtered_outer_maximum = None
        self.filtered_inner_maximum = None
        self.filtered_inner_minimum = None
        self.filtered_outer_minimum = None
        
        self.there_is_cached_data = False
        self.output_abscissa = None
        self.floor = 0.01
        self.n_keyframes = 0
        self.firstIndex = 0
        self.lastIndex = 0
        
        self.absolute_max = None
        self.absolute_min = None
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
        self.out_name = "SRN"
        self.s_radius = 400
        
        self.in_name = in_name
        self.fits_path = fits_path
        self.show = show
        self.verb = verb
        self.do_orig = orig
        
        self.mirror_mask = None
        self.grid_mask = None
        self.t_factor = None
        self.tRadius = None
        self.center = None
        # self.outer_max = None
        # self.outer_min = None
        self.changed_flat = None
        self.changed_flat = None
        self.rez = None
        self.original = None
        self.changed = None
        self.binInds = None
        self.bin_rez = None
        self.radBins = None
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
    
    ###################
    ## Structure ##
    ###################
    
    def setup(self):
        """Do prep work once before the main algorithm"""
        raise NotImplementedError
    
    def do_work(self):
        """Do whatever you want to each image in the directory"""
        raise NotImplementedError
    
    def cleanup(self):
        """Runs once after all the images have been modified with do_work"""
        raise NotImplementedError
    
    def do_fits_function(self, fits_path=None, in_name=None, image=True):
        """Calls the do_work function on a single fits path if indicated"""
        if self.load_fits_image(fits_path):
            if (not self.use_keyframes) or (self.fits_path in self.keyframes):
                return self.do_work()  # Do the work on the fits files
        return None

    def do_img_function(self):
        """Calls the do_work function on a single fits path if indicated"""
        return self.do_work()  # Do the work on the fits files
    
    ###################
    ## Top-Level ##
    ###################
    
    def skip_bad_frame(self):
        # self.header
        return False
    
    def image_learn(self):
        """Analyze the input image to help make normalization curves"""
        if not self.skip_bad_frame():
            self.init_for_learn()
            self.bin_radially()  # Create a cloud of intesity values for each radial bin
            self.radial_statistics()  # Find mean and percentiles vs height
            self.add_to_keyframes()  # Update the running curves
            self.make_smoothed_curves()
    
    def image_modify(self):
        """Perform the actual normalization on the input array"""
        self.init_for_modify()
        self.coronaNorm()  # Use curves to rescale the in_object
        self.coronagraph_touchup()  # Deal with some outliers
        self.prep_output()
        self.vignette()  # Truncate the in_object above given radius
    
    ###################
    ## Keyframes ##
    ###################
    
    def add_to_keyframes(self):
        """Records the current analysis as one of the radial samples"""
        self.update_keyframe_counters()
        # self.remove_offset()
        if not self.outputs_initialized or self.params.Force_init:
            self.init_running_curves()
        else:
            self.update_running_curves()
    
    def update_keyframe_counters(self, n=1):
        """Keep track of how many items have been added to keyframes"""
        self.n_keyframes += n
        self.lastIndex += n
        self.skipped -= n
        self.skipped = max(self.skipped, 0)
    
    ######################################
    ## Initializeing and Converting ##
    ######################################
    
    def init_for_learn(self):
        self.init_images()
        self.init_frame_curves()
        self.init_radius_array()
        self.init_statistics()
    
    def init_for_modify(self):
        self.init_images()
        self.init_radius_array()

    
    def init_radius_array(self, vignette_radius=1.2, s_radius=400, t_factor=1.28, force=False):
        """Build an r-coordinate array of shape(in_object)"""
        
        if self.rez is None:
            self.rez = self.changed.shape[0]
        self.output_abscissa = np.arange(self.rez)
        
        if self.radius is None or force or self.changed.shape[0] != self.rez:
            dprint("init_radius_array")
            
            xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
            xc, yc = xx - self.center[0], yy - self.center[1]
            
            self.radius = np.sqrt(xc * xc + yc * yc)
            self.rad_flat = self.radius.flatten()
            self.binInds = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
            
            self.vignette_mask = self.radius > (int(vignette_radius * self.rez // 2))
            self.s_radius = s_radius
            self.tRadius = self.s_radius * t_factor
    
    def init_images(self, changed=None):
        """Get all the variables ready for the normalization"""
        dprint("\ninit_images")
        if self.params.changed is not None:
            self.changed = self.params.changed
        
        if changed is not None:
            self.changed = changed
        
        self.rez = self.changed.shape[0]
        self.changed = self.changed.astype('float32')
        self.changed[self.changed == 0] = np.nan
        
        if self.original is None:
            self.original = self.changed + 0
        
        self.original_flat = self.original.flatten()
        self.changed_flat = self.changed.flatten()
    
    def init_frame_curves(self):
        """These are the main frame_level curves"""
        dprint("init_frame_curves")
        self.frame_maximum = np.empty(self.rez)
        self.frame_minimum = np.empty(self.rez)
        self.frame_abs_max = np.empty(self.rez)
        self.frame_abs_min = np.empty(self.rez)
        self.frame_abss = np.empty(self.rez)
        
        self.frame_maximum.fill(np.nan)
        self.frame_minimum.fill(np.nan)
        self.frame_abs_max.fill(np.nan)
        self.frame_abs_min.fill(np.nan)
        self.frame_abss.fill(np.nan)
        
        if self.center is None:
            self.center = [self.rez / 2, self.rez / 2]
    
    def init_statistics(self):
        """Initialize the statistical arrays"""
        dprint("init_statistics")
        
        self.bin_rez = np.max(self.binInds) + 10
        self.radBins = [[] for x in np.arange(self.bin_rez)]
        
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
        need_to_run = (self.frame_minimum is not None)
        need_to_run = need_to_run and np.sum(np.isfinite(self.frame_minimum)) > 0
        if need_to_run or self.params.Force_init:
            dprint("init_running_curves")
            self.outer_min = self.frame_minimum + 0
            self.avg_min = self.frame_minimum + 0
            self.inner_min = self.frame_minimum + 0
            self.inner_max = self.frame_maximum + 0
            self.avg_max = self.frame_maximum + 0
            self.outer_max = self.frame_maximum + 0
            self.abs_max = self.frame_abs_max + 0
            self.abs_min = self.frame_abs_min + 0
            
            self.absolute_min = max(np.nanmin(self.outer_min), -10)
            self.absolute_max = np.nanmax(self.outer_max)
            # self.check_if_same("init_running_curves")
            self.outputs_initialized = True and self.can_initialize # TODO I changed this in case something is broken
        return True
    
    def update_running_curves(self):
        """Update the Curves"""
        dprint("update_running_curves")
        
        self.outer_min = np.fmin(self.outer_min, self.frame_minimum)
        self.inner_min = np.fmax(self.inner_min, self.frame_minimum)
        self.inner_max = np.fmin(self.inner_max, self.frame_maximum)
        self.outer_max = np.fmax(self.outer_max, self.frame_maximum)
        self.abs_max   = np.fmax(self.abs_max,   self.frame_abs_max)
        self.abs_min   = np.fmin(self.abs_min,   self.frame_abs_min)
        
        self.absolute_max = np.fmax(self.absolute_max, np.max(self.outer_max))
        self.absolute_min = np.fmin(self.absolute_min, np.max(self.outer_min))

        self.avg_min = self.avg_min + self.frame_minimum
        self.avg_max = self.avg_max + self.frame_maximum
        self.norm_avg_min = self.avg_min / self.n_keyframes
        self.norm_avg_max = self.avg_max / self.n_keyframes
    

    
    # def raster_extrema_curves(self):
    #     """Raster out the min/max curves from the rendered version"""
    #     if self.norm_curve_min is None and self.outer_min is not None:
    #         self.do_raster()
    #         return True
    #     return False
    
    def do_raster(self):
        """make the normalization curves that reduce the data to the smaller range"""
        # Roadrunner
        # self.rastered_outer_max = np.squeeze(self.outer_max[self.binInds])
        # self.rastered_inner_max = np.squeeze(self.inner_max[self.binInds])
        # self.rastered_inner_min = np.squeeze(self.inner_min[self.binInds])
        # self.rastered_outer_min = np.squeeze(self.outer_min[self.binInds])
        
        # self.norm_curve_max = np.squeeze(self.smooth_inner_maximum[self.binInds])
        # self.norm_curve_min = np.squeeze(self.smooth_inner_minimum[self.binInds])
        pass
        
    def remove_offset(self):
        """Make sure everything is positive"""
        self.changed -= self.absolute_min
        self.original -= self.absolute_min
        self.outer_max -= self.absolute_min
        self.inner_max -= self.absolute_min
        self.inner_min -= self.absolute_min
        self.outer_min -= self.absolute_min
    
    def plot_inner_outer(self, show=False, save=True, fig=None, ax=None,
                         extra=False, raw=False, smooth=True):
        """Look at the results of the algorithm"""
        if ax is None or fig is None:
            fig, ax = plt.subplots()
            do_save = True
        else:
            do_save = False
            
        plt.ioff()
        ax.set_title("Intensity as a function of radial distance: AIA_{}".format(self.params.current_wave()))
        ## Plotting ##
        do_all = True
        raw_alpha = 0.85
        grey_alpha = 0.90
        
        rrarr = self.n2r(np.arange(len(self.outer_max)))
        # Plot Raw Curves
        if do_all and self.outer_max is not None:
            if raw: ax.plot(rrarr, self.outer_max, zorder=4, lw=1, label="Top/Bot", alpha=raw_alpha, c='orange')
            if extra: ax.plot(rrarr, self.inner_max, zorder=5, lw=1, label="In Max",  alpha=raw_alpha, c='cornflowerblue')
            if extra: ax.plot(rrarr, self.inner_min, zorder=6, lw=1, label="In Min",  alpha=raw_alpha, c='cornflowerblue')
            if raw: ax.plot(rrarr, self.outer_min, zorder=3, lw=1, alpha=raw_alpha, c='orange')
    
        # Plot Current Frame Curves
        if do_all and self.frame_maximum is not None and extra:
            if raw: ax.plot(rrarr, self.frame_maximum,zorder=8, lw=1,                     c='darkgrey', alpha=grey_alpha)
            if raw: ax.plot(rrarr, self.frame_minimum,zorder=7, lw=1, label="Frame",      c='darkgrey', alpha=grey_alpha)
            if smooth: ax.plot(rrarr, self.smoothed_frame_maximum, zorder=10,lw=1,                      c='darkslategrey', alpha=1)
            if smooth: ax.plot(rrarr, self.smoothed_frame_minimum, zorder=9, lw=1, label="Smooth Frame",  c='darkslategrey', alpha=1)

        # Plot Absolute Curves
        if do_all and self.abs_max is not None:
            if raw: ax.plot(rrarr, self.abs_max, zorder=1, lw=1, label="Hat/Shoe",   c='darkgrey', alpha=grey_alpha)
            if raw: ax.plot(rrarr, self.abs_min, zorder=1, lw=1,   c='darkgrey', alpha=grey_alpha)
            if smooth: ax.plot(rrarr, self.smoothed_abs_max, zorder=200, lw=4,                          c='cornflowerblue', alpha=1)
            if smooth: ax.plot(rrarr, self.smoothed_abs_min, zorder=200, lw=4, label="Smooth Hat/Shoe",  c='cornflowerblue', alpha=1)
            
        RRarr = self.n2r(self.output_abscissa)
        # Plot Filtered Curves
        if self.filtered_inner_maximum is not None and extra:
            ax.plot(RRarr, self.filtered_outer_maximum, zorder=103,lw =1, c="m", label="Fltr. Out")
            ax.plot(RRarr, self.filtered_inner_maximum, zorder=102,lw =1, c='c', label="Fltr. Inn")
            ax.plot(RRarr, self.filtered_inner_minimum, zorder=102,lw =1, c="c", ls='--')
            ax.plot(RRarr, self.filtered_outer_minimum, zorder=103,lw =1, c='m', ls='--', )

        # Plot Smoothed Curves
        if self.smooth_inner_maximum is not None:
            if smooth: ax.plot(RRarr, self.smooth_outer_maximum, zorder=105, lw=4, c="g", label="Smooth Top/Bot")
            if smooth: ax.plot(RRarr, self.smooth_inner_maximum, zorder=104, lw=4, c='r', label="Smooth Shoulder/Knee")
            if smooth: ax.plot(RRarr, self.smooth_inner_minimum, zorder=104, lw=4, c="r")
            if smooth: ax.plot(RRarr, self.smooth_outer_minimum, zorder=105, lw=4, c='g')
            # ax.plot(rrarr, self.smoothed_frame_abs_max, zorder=1, lw=1,                   c='grey', alpha=1)
            # ax.plot(rrarr, self.smoothed_frame_abs_min, zorder=1, lw=1, label="Smo. Abs", c='grey', alpha=1)
    
        # Vertical Lines
        ax.axvline(1)
        if self.lCut is not None:
            ax.axvline(self.n2r(self.lCut), ls=":")
            ax.axvline(self.n2r(self.hCut), ls=":")
    
        # Plot Scatter Points
        self.skip_points = 10 if self.rez < 3000 else 500  # TODO Make this sample better, linear isn't appropriate because its a circle
        ax.scatter(self.n2r(self.rad_flat[::self.skip_points]), self.original.flatten()[::self.skip_points], c='k', s=2)
    
        ## Plot Formatting
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Distrance from Sun Center")
        ax.set_yscale("symlog")
        ax.set_ylim((-10 ** 1, 10 ** 4))
        ax.legend(loc='lower left')
        # plt.show()
        # Plot Saving
        if do_save:
            self.force_save_inner_outer(save, fig, ax, show)
        else:
            plt.show(block=True)
    
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
                
            save_path_1 = join(bs, folder_name, 'radial_hist', file_name_1)
            save_path_2 = join(bs, folder_name, 'radial_hist', 'zoom', file_name_2)
            
            makedirs(dirname(save_path_1), exist_ok=True)
            # makedirs(dirname(save_path_2), exist_ok=True)
            fig.set_size_inches((20,10))
            plt.tight_layout()

            while True:
                try:
                    plt.savefig(save_path_1, dpi=150)
                    break
                except OSError as e:
                    print("  !!!!!!! Close the Dang Plot!", end='')
                print('.', end='')
                
            plt.xlim((0.9, 1.1))
            plt.ylim((10**1.5, 10**3.5))
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
    ## Normalization Curve Stuff ##
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
    
    def do_bin(self, skip=30):  # Bin the intensities by radius
        self.cut_pixels = skip
        for binI, dat in zip(self.binInds[::self.cut_pixels], self.original_flat[::self.cut_pixels]):
            self.radBins[binI].append(dat)
    
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
        for ii, bin_list in enumerate(self.radBins):
            self.store_bin_array(ii, bin_list)
        self.finalize_radial_statistics()
    
    def store_bin_array(self, ii, bin_list):
        """Do statistics on a given bin"""
        bin_array = self.get_bin_items(bin_list)
        if len(bin_array) > 0:
            
            a,b,c,d = np.percentile(bin_array, [99, 95, 2, 0.5])
            
            self.binAbsMax[ii]  = a     #np.percentile(bin_array, 99.999)
            self.binMax[ii]     = b    #np.percentile(bin_array, 96)
            self.binMin[ii]     = c    #np.percentile(bin_array, 2)
            self.binAbsMin[ii]  = d     #np.percentile(bin_array, 0.001)
    
    @staticmethod
    def get_bin_items(bin_list):
        """Retrieve finite values from a bin_list"""
        bin_array = np.asarray(bin_list)
        finite = bin_array[np.isfinite(bin_array)]
        subItems = finite[np.nonzero(finite)]
        return subItems
    
    def finalize_radial_statistics(self):
        """Clean up the radial statistics to be used"""
        idx = np.isfinite(self.binMax) & np.isfinite(self.binMin)
        n_index = self.binAbss[idx]
        self.frame_abss[n_index] = n_index
        self.frame_maximum[n_index] = self.binMax[idx]
        self.frame_minimum[n_index] = self.binMin[idx]
        self.frame_abs_max[n_index] = self.binAbsMax[idx]
        self.frame_abs_min[n_index] = self.binAbsMin[idx]
    

        
    def make_smoothed_curves(self):
        """Build the normalization arrays, treating the domain in 3 seperate regions"""
        
        self.split_into_three_regions()
        self.filter_three_regions()
        self.concatinate_filtered_regions()
        self.render_smooth_curves()
        self.prep_save_outs()
    
    def split_into_three_regions(self):
        # Split the domain into three regions
        if not self.found_limb_radius:
            self.found_limb_radius = self.params.found_limb_radius
        self.lCut = int(self.found_limb_radius - 0.01 * self.rez)
        self.hCut = int(self.found_limb_radius + 0.01 * self.rez)
        
        if self.frame_abss is None:
            self.frame_abss = np.arange(self.rez)
        abss = self.frame_abss
        use_max = self.outer_max + 0
        use_min = self.outer_min + 0
        
        # Split into three regions
        self.outer_low_abs = abss[:self.lCut]
        self.outer_low_max = use_max[:self.lCut]
        self.outer_low_min = use_min[:self.lCut]
        
        self.outer_mid_abs = abss[self.lCut:self.hCut]
        self.outer_mid_max = use_max[self.lCut:self.hCut]
        self.outer_mid_min = use_min[self.lCut:self.hCut]
        
        self.outer_high_abs = abss[self.hCut:]
        self.outer_high_max = use_max[self.hCut:]
        self.outer_high_min = use_min[self.hCut:]
        
        abss = self.frame_abss
        use_max = self.inner_max + 0
        use_min = self.inner_min + 0
        
        # Split into three regions
        self.inner_low_abs = abss[:self.lCut]
        self.inner_low_max = use_max[:self.lCut]
        self.inner_low_min = use_min[:self.lCut]
        
        self.inner_mid_abs = abss[self.lCut:self.hCut]
        self.inner_mid_max = use_max[self.lCut:self.hCut]
        self.inner_mid_min = use_min[self.lCut:self.hCut]
        
        self.inner_high_abs = abss[self.hCut:]
        self.inner_high_max = use_max[self.hCut:]
        self.inner_high_min = use_min[self.hCut:]
        
        
        
    
    def filter_three_regions(self):
        ### Filter the regions separately
        mode = 'nearest'
        
        # Savgol windows
        lWindow = 31  # 4 * self.extra_rez + 1
        ln = 6
        mWindow = 11  # 4 * self.extra_rez + 1
        mn = 3
        hWindow = 31  # 30 * self.extra_rez + 1
        hn = 2
        
        maxWindow = 31
        maxn = 6
        
        
        rank = 2
        
        # Max Curve
        for i in range(ln):
            try:
                self.outer_low_max  = savgol_filter(self.outer_low_max,  lWindow, rank, mode=mode)
                self.inner_low_max  = savgol_filter(self.inner_low_max,  lWindow, rank, mode=mode)
                self.inner_low_min  = savgol_filter(self.inner_low_min,  lWindow, rank, mode=mode)
                self.outer_low_min  = savgol_filter(self.outer_low_min,  lWindow, rank, mode=mode)
            except np.linalg.LinAlgError as e:
                print("\n filter:three:regions::")
                print(e)
        
        for i in range(mn):
            try:
                self.outer_mid_max  = savgol_filter(self.outer_mid_max,  mWindow, rank, mode=mode)
                self.inner_mid_max  = savgol_filter(self.inner_mid_max,  mWindow, rank, mode=mode)
                self.inner_mid_min  = savgol_filter(self.inner_mid_min,  mWindow, rank, mode=mode)
                self.outer_mid_min  = savgol_filter(self.outer_mid_min,  mWindow, rank, mode=mode)
            except np.linalg.LinAlgError as e:
                print("\n filter:three:regions::")
                print(e)
        for i in range(hn):
            try:
                self.outer_high_max = savgol_filter(self.outer_high_max, hWindow, rank, mode=mode)
                self.inner_high_max = savgol_filter(self.inner_high_max, hWindow, rank, mode=mode)
                self.inner_high_min = savgol_filter(self.inner_high_min, hWindow, rank, mode=mode)
                self.outer_high_min = savgol_filter(self.outer_high_min, hWindow, rank, mode=mode)
            except np.linalg.LinAlgError as e:
                print("\n filter:three:regions::")
                print(e)
                
                
        if self.frame_minimum is not None:
            self.smoothed_frame_minimum = self.frame_minimum+0
            self.smoothed_frame_maximum = self.frame_maximum+0
            self.smoothed_frame_abs_max = self.frame_abs_max+0
            self.smoothed_frame_abs_min = self.frame_abs_min+0
            
            self.smoothed_abs_max = self.abs_max+0
            self.smoothed_abs_min = self.abs_min+0
            
        for i in range(maxn):
            try:
                # Bonus Extrema Filtering!
                if self.frame_minimum is not None:
                    # print("running")
                    self.smoothed_frame_minimum = savgol_filter(self.smoothed_frame_minimum, maxWindow, rank, mode=mode)
                    self.smoothed_frame_maximum = savgol_filter(self.smoothed_frame_maximum, maxWindow, rank, mode=mode)
                    self.smoothed_frame_abs_max = savgol_filter(self.smoothed_frame_abs_max, maxWindow, rank, mode=mode)
                    self.smoothed_frame_abs_min = savgol_filter(self.smoothed_frame_abs_min, maxWindow, rank, mode=mode)
                    self.smoothed_abs_max       = savgol_filter(self.smoothed_abs_max,       maxWindow, rank, mode=mode)
                    self.smoothed_abs_min       = savgol_filter(self.smoothed_abs_min,       maxWindow, rank, mode=mode)
            except np.linalg.LinAlgError as e:
                print("\n filter:three:regions::")
                print(e)
        
    def concatinate_filtered_regions(self):
        # Concatinate filtered curves
        self.output_abscissa        = np.hstack((self.inner_low_abs, self.inner_mid_abs, self.inner_high_abs))
        self.filtered_outer_maximum = np.hstack((self.outer_low_max, self.outer_mid_max, self.outer_high_max))
        self.filtered_inner_maximum = np.hstack((self.inner_low_max, self.inner_mid_max, self.inner_high_max))
        self.filtered_inner_minimum = np.hstack((self.inner_low_min, self.inner_mid_min, self.inner_high_min))
        self.filtered_outer_minimum = np.hstack((self.outer_low_min, self.outer_mid_min, self.outer_high_min))
    
    def render_smooth_curves(self):
        
        # Fit the lowest region with a polynomial to make it much smoother
        flatten_inner_ind = 200

        # degree = 5
        # p = np.polyfit(self.low_abs, self.low_max_filt, degree)
        # self.low_max_fit = np.polyval(p, self.low_abs)  # * 1.1
        # p = np.polyfit(self.low_abs, self.low_min_filt, degree)
        # self.low_min_fit = np.polyval(p, self.low_abs)
        

        
        self.smooth_outer_maximum = self.filtered_outer_maximum + 0
        self.smooth_inner_maximum = self.filtered_inner_maximum + 0
        self.smooth_inner_minimum = self.filtered_inner_minimum + 0
        self.smooth_outer_minimum = self.filtered_outer_minimum + 0

        # Flatten out the edges
        self.smooth_outer_maximum[0:flatten_inner_ind] = self.filtered_outer_maximum[flatten_inner_ind]
        self.smooth_inner_maximum[0:flatten_inner_ind] = self.filtered_inner_maximum[flatten_inner_ind]
        self.smooth_inner_minimum[0:flatten_inner_ind] = self.filtered_inner_minimum[flatten_inner_ind]
        self.smooth_outer_minimum[0:flatten_inner_ind] = self.filtered_outer_minimum[flatten_inner_ind]

        self.norm_curve_outer_max = np.squeeze(self.smooth_outer_maximum[self.binInds])
        self.norm_curve_inner_max = np.squeeze(self.smooth_inner_maximum[self.binInds])
        self.norm_curve_inner_min = np.squeeze(self.smooth_inner_minimum[self.binInds])
        self.norm_curve_outer_min = np.squeeze(self.smooth_outer_minimum[self.binInds])
        
    def render_extrema_curves(self, flatten_inner_ind=200):
        if self.abs_max is not None:
            filtered_abs_max     = savgol_filter(self.abs_max, 21, 3, mode='nearest')
            filtered_abs_min     = savgol_filter(self.abs_min, 21, 3, mode='nearest')
            self.smoothed_abs_max     = filtered_abs_max
            self.smoothed_abs_min     = filtered_abs_min
            self.smoothed_abs_max[0:flatten_inner_ind]     = self.smoothed_abs_max[flatten_inner_ind]
            self.smoothed_abs_min[0:flatten_inner_ind]     = self.smoothed_abs_min[flatten_inner_ind]
            self.norm_smoothed_abs_max = np.squeeze(self.smoothed_abs_max[self.binInds])
            self.norm_smoothed_abs_min = np.squeeze(self.smoothed_abs_min[self.binInds])
            
    ####################################
    ## Image Reduction Algorithms ##
    ####################################
    
    def coronaNorm(self, changed=None):
        """Normalize the in_object using the radial percentile curves"""
        # Collect Arrays
        self.init_images(changed)

        # Make Curves
        self.make_smoothed_curves()
        
        # Normalize Them
        self.execute_norm()
    
    def execute_norm(self):
        """Apply the Normalization to the Image Array"""
        # self.norm_curve_outer_max
        # self.norm_curve_inner_max
        # self.norm_curve_inner_min
        # self.norm_curve_outer_min

        self.render_extrema_curves()
        self.norm_curve_max = self.norm_curve_inner_max
        # self.norm_curve_min = self.norm_curve_outer_min
        self.norm_curve_min = self.norm_smoothed_abs_min
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # Standard Normalization Formula
                del self.changed_flat
                del self.original_flat
                self.norm_formula(self.changed, self.norm_curve_min, self.norm_curve_max)
            except RuntimeWarning as e:
                print(e)  
        return  
    
#                 self.changed_flat = self.norm_formula(self.changed, self.norm_curve_min, self.norm_curve_max)
#                 self.changed = self.changed_flat.reshape(the_shape).astype('float32') # # purple
#                         import pdb; pdb.set_trace()
#         del flat_image
#         del self.original
#         del self.original_flat
#         del the_min
#         del the_max
        
#         pass

    
    def coronagraph_touchup(self):
        """Deal with pixel outliers. Lots of adjustable parameters in here"""
        
        # neg = self.changed<0
        # neg_pts = self.changed[neg]
        # minn = np.abs(np.min(neg_pts))
        # normed = neg_pts + min(neg_pts)
        
        
        # self.changed += minn
        self.changed = np.power(self.changed, 1/3)
        self.changed /= 1.5
        self.changed -= 0.15
        
        # self.changed = np.power(self.changed, 1/4)
        # self.changed -= minn
        
        ## Deal with too hot things ##
        # self.vmax = 2
        # self.vmax_plot = 0.95  # np.max(changed_flat) #this is in the header of the imageprocessor now
        # hotpowr = 1 / 2
        # hot = self.changed > self.vmax
        # self.changed[hot] = self.changed[hot] ** hotpowr
        
        
        
        # ## Deal with too cold things ##
        # self.vmin = 0.3
        # self.vmin_plot = -0.05  # np.min(changed_flat)# 0.3# -0.03 #this is in the header of the imageprocessor now
        # coldpowr = 1 / 2
        # cold = self.changed_flat < self.vmin
        # self.changed_flat[cold] = -((np.abs(self.changed_flat[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin
        
        ## Some Final Normalization ##      TODO: I think this might be breaking things!
        # self.changed_flat = self.normalize(self.changed_flat, high=99.99, low=1)

        
    def prep_output(self):
        
        self.mask_output()
        self.mirror_output()
        
        # Un-Flatten the Array
        # self.changed = np.sign(self.changed) * np.power(np.abs(self.changed), (1 / 5))
    
    def mask_output(self, do_mask=None):
        """Allows you to only show sub-sections of the in_object as reduced images"""
        if not do_mask:
            return False
        
        self.grid_mask = self.get_mask(self.changed, force=True)
        
        if self.grid_mask is not None:
            self.changed[self.grid_mask] = self.original[self.grid_mask]
    
    def mirror_output(self, do_mirror=None):
        # Allows you to mirror horizontally, with only one half rfeduced
        if not do_mirror:
            return False
        
        self.mirror_mask = self.get_mask(self.changed, force=True)
        
        newDat = self.changed[self.mirror_mask]
        xx, yy = self.mirror_mask.shape[0], int(self.mirror_mask.shape[1] / 2)
        grid = newDat.reshape(xx, yy)
        flipped = np.fliplr(grid)
        
        if self.mirror_mask is not None:
            self.changed[~self.mirror_mask] = flipped.flatten()  # np.flip(newDat)
    
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
        self.changed[self.vignette_mask] = np.nan
        self.original[self.vignette_mask] = np.nan
    
    ########################
    ## Plotting Stuff ##
    ########################
    
    
    def plot_radial_norm_keyframes(self, do=False, show=False, save=True):
        """This plot is in radius and has a scatter plot
            overlaid with the norm curves as determined elsewhere"""
        if not do:
            return
        if self.first:
            self.first = False
            return
    
        # self.output_abscissa
        dprint("plot_radial_norm_keyframes")
        # Init the Plots
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex="all")
    
        ## Plot 0 Absolute Coordinates
        self.render_smooth_curves()
        self.plot_inner_outer(fig=fig, ax=ax0, save=False)
    
        ## Plot 1 Normalized
        skip = self.skip_points
        ax1.scatter(self.n2r(self.rad_flat[::skip]), self.changed.flatten()[::skip], c='k', s=2)
        ax1.axhline(2)
        ax1.axhline(1)
        ax1.axhline(0)
    
        ## Plot 0 Formatting
        ax0.set_title("Plot Stats")
        ax0.set_ylim((-10 ** 1, 10 ** 4))
        ax0.set_xlim((0, 1.6))
        # ax0.legend()
        ax0.set_yscale('symlog')
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
    
        ## Plot 1 Formatting
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        
        ax1.set_yscale("symlog")
        ax1.set_ylim((-0.5, 10))
    
        fig.set_size_inches(12, 12)
        plt.tight_layout()
        # plt.show(block=True)
    
        self.force_save_radial_figures(save, fig, ax0, show)
    
    
        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))
        # ax.axvline(self.tRadius, c='r')
    
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
        if not do: return
        # print("Saving {}".format(file_name_1))
        bs = self.params.base_directory()
        folder_name = "analysis\\radial_hist_full"
        file_name_1 = 'full_{}.png'.format(self.file_basename[:-5])
        save_path_1 = join(bs, folder_name, file_name_1)
        
        file_name_2 = 'zoom\\full_zoom_{}.png'.format(self.file_basename[:-5])
        save_path_2 = join(bs, folder_name, file_name_2)
        
        makedirs(dirname(save_path_1), exist_ok=True)
        plt.savefig(save_path_1)
        
        makedirs(dirname(save_path_2), exist_ok=True)
        ax.set_xlim((0.9, 1.1))
        plt.savefig(save_path_2)
        
        if not show:
            plt.close(fig)
        else:
            plt.show()
    
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
        if n is None:
            n = 0
        return n / self.found_limb_radius
    
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
    def norm_formula(image, the_min, the_max):
        """Standard Normalization Formula"""
#         import pdb; pdb.set_trace()
        shape=image.shape
        image=image.flatten()
        diff = np.subtract(the_max, the_min)
        np.subtract(image, the_min, out=image)
        np.divide(image, diff, out=image)
        return image.reshape(shape)

    
    
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


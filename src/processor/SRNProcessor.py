import os
from os import makedirs
from os.path import join, dirname
import numpy as np
from scipy.signal import savgol_filter
from processor.Processor import Processor

import warnings

warnings.filterwarnings("ignore")
import matplotlib as mpl

mpl.use("qt5agg")
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
    found_limb_radius = 1600
    
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

        self.smooth_minimum = None
        self.smooth_maximum = None
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
        self.binMean = None
        self.binMed = None
        self.binAbss = None
        self.norm_curve_min = None
        self.norm_curve_max = None
        
        self.vignette_mask = None
        self.frame_maximum = None
        self.frame_minimum = None
        self.frame_mean = None
        self.frame_med = None
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
    
    def do_fits_function(self, fits_path=None, in_name=None):
        """Calls the do_work function on a single fits path if indicated"""
        if self.load_fits_image(fits_path):
            if (not self.use_keyframes) or (self.fits_path in self.keyframes):
                return self.do_work()  # Do the work on the fits files
        return None
    
    ###################
    ## Top-Level ##
    ###################
    
    def image_learn(self):
        """Analyze the input image to help make normalization curves"""
        self.init_for_learn()
        self.bin_radially()  # Create a cloud of intesity values for each radial bin
        self.radial_statistics()  # Find mean and percentiles vs height
        self.make_smoothed_curves()  # Build smooth curves based on the statistics
        self.add_to_keyframes()  # Update the running curves
        
    
    
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
        if not self.outputs_initialized:
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
        self.frame_mean = np.empty(self.rez)
        self.frame_med = np.empty(self.rez)
        self.frame_abss = np.empty(self.rez)
        
        self.frame_maximum.fill(np.nan)
        self.frame_minimum.fill(np.nan)
        self.frame_mean.fill(np.nan)
        self.frame_med.fill(np.nan)
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
        self.binMean = np.empty(self.bin_rez)
        self.binMed = np.empty(self.bin_rez)
        self.binAbss = np.arange(self.bin_rez)
        
        self.binMax.fill(np.nan)
        self.binMin.fill(np.nan)
        self.binMean.fill(np.nan)
        self.binMed.fill(np.nan)
    
    def init_running_curves(self):
        """Initialize the curves"""
        if not self.outputs_initialized:
            need_to_run = (self.frame_minimum is not None)
            need_to_run = need_to_run and np.sum(np.isfinite(self.frame_minimum)) > 0
            if need_to_run:
                dprint("init_running_curves")
                self.outer_min = self.frame_minimum + 0
                self.avg_min = self.frame_minimum + 0
                self.inner_min = self.frame_minimum + 0
                self.inner_max = self.frame_maximum + 0
                self.avg_max = self.frame_maximum + 0
                self.outer_max = self.frame_maximum + 0
                
                self.absolute_min = max(np.nanmin(self.outer_min), -10)
                self.absolute_max = np.nanmax(self.outer_max)
                # self.check_if_same("init_running_curves")
                self.outputs_initialized = True
            return True
        print("Skipping Init Running Curves")
        return False
    
    def update_running_curves(self):
        """Update the Curves"""
        dprint("update_running_curves")
        
        self.outer_min = np.fmin(self.outer_min, self.frame_minimum)
        self.inner_min = np.fmax(self.inner_min, self.frame_minimum)
        self.inner_max = np.fmin(self.inner_max, self.frame_maximum)
        self.outer_max = np.fmax(self.outer_max, self.frame_maximum)
        
        self.absolute_max = np.fmax(self.absolute_max, np.max(self.outer_max))
        self.absolute_min = np.fmin(self.absolute_min, np.max(self.outer_min))
        
        self.avg_min = self.avg_min + self.frame_minimum
        self.avg_max = self.avg_max + self.frame_maximum
        self.norm_avg_min = self.avg_min / self.n_keyframes
        self.norm_avg_max = self.avg_max / self.n_keyframes
    
    def raster_extrema_curves(self):
        """Raster out the min/max curves from the rendered version"""
        if self.norm_curve_min is None and self.outer_min is not None:
            self.do_raster()
            return True
        return False
    
    def do_raster(self):
        """make the normalization curves that reduce the data to the smaller range"""
        # Roadrunner
        self.rastered_outer_max = np.squeeze(self.outer_max[self.binInds])
        self.rastered_inner_max = np.squeeze(self.inner_max[self.binInds])
        self.rastered_inner_min = np.squeeze(self.inner_min[self.binInds])
        self.rastered_outer_min = np.squeeze(self.outer_min[self.binInds])
        
        self.rastered_smooth_max = np.squeeze(self.smooth_maximum[self.binInds])
        self.rastered_smooth_min = np.squeeze(self.smooth_minimum[self.binInds])
        
        self.norm_curve_max = self.rastered_smooth_max
        self.norm_curve_min = self.rastered_smooth_min
    
    def remove_offset(self):
        """Make sure everything is positive"""
        self.changed   -= self.absolute_min
        self.original  -= self.absolute_min
        self.outer_max -= self.absolute_min
        self.inner_max -= self.absolute_min
        self.inner_min -= self.absolute_min
        self.outer_min -= self.absolute_min
    
    
    def plot_inner_outer(self, show=False, save=True):
        # Save the Image
        fig, ax = plt.subplots()
        plt.ioff()
        ax.set_title("Intensity as a function of radial distance: AIA_{}".format(self.params.current_wave()))
        
        # Plotting
        rrarr = self.n2r(np.arange(len(self.outer_max)))
        ax.plot(rrarr, self.outer_max, zorder=4, lw=2, label="Out Max", c='orange')
        ax.plot(rrarr, self.inner_max, zorder=5,  lw=2, label="In Max", c='gold')
        ax.plot(rrarr, self.inner_min, zorder=6,  lw=2, label="In Min", c='cornflowerblue')
        ax.plot(rrarr, self.outer_min, zorder=3, lw=2, label="Out Min", c='b')
        ax.axvline(1)
        
        ax.plot(self.n2r(self.output_abscissa), self.smooth_maximum, zorder=10, c="r",label="Smooth")
        ax.plot(self.n2r(self.output_abscissa), self.smooth_minimum, zorder=10, c='r')
        
        skip = 500 #TODO Make this sample better, linear isn't appropriate because its a circle
        ax.scatter(self.n2r(self.rad_flat[::skip]), self.original.flatten()[::skip], c='k', s=2)
        
        
        #Plot Formatting
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Distrance from Sun Center")
        ax.set_yscale("symlog")
        ax.set_ylim((-10**2, 10**4))
        plt.legend(loc='lower left')
        
        self.force_save_inner_outer(save, fig, ax, show)
        
        
    def force_save_inner_outer(self, save, fig, ax0, show):
        # Save Path Stuff
        if save:
            bs = self.params.base_directory()
            if save == "single":
                folder_name = "norm_curves"
                file_name_1 = '{}_keyframe.png'.format(self.params.current_wave())
            else:
                folder_name = "radial"
                file_name_1 = 'keyframe_{}.png'.format(self.file_basename[:-5])
            save_path_1 = join(bs, folder_name, file_name_1)
            makedirs(dirname(save_path_1), exist_ok=True)
            while True:
                try:
                    plt.savefig(save_path_1, dpi=300)
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
            
    def do_bin(self, skip=30): # Bin the intensities by radius
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
            self.binMax[ii] = np.percentile(bin_array, 98)  # np.nanmax(subItems)
            self.binMin[ii] = np.percentile(bin_array, 2)  # np.min(subItems)
            self.binMean[ii] = np.mean(bin_array)
            self.binMed[ii] = np.median(bin_array)
    
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
        self.frame_mean[n_index] = self.binMean[idx]
        self.frame_med[n_index] = self.binMed[idx]

    
    def make_curves(self):
        """Build the normalization arrays, treating the domain in 3 seperate regions"""
        
        ## Parameters
        self.highCut = 0.8 * self.rez
        
        # Savgol window size
        lWindow = 7  # 4 * self.extra_rez + 1
        mWindow = 7  # 4 * self.extra_rez + 1
        hWindow = 51  # 30 * self.extra_rez + 1
        fWindow = 7  # int(3 * self.extra_rez) + 51
        rank = 3
        
        ## Algorithm
        # Locate the Limb
        self.theMin = int(0.30 * self.rez)
        self.theMax = int(0.45 * self.rez)
        near_limb = np.arange(self.theMin, self.theMax)
        
        # Split the domain into three regions and treat seperately
        r1 = self.binAbss[np.argmax(self.binMean[near_limb]) + self.theMin]
        r2 = self.binAbss[np.argmax(self.binMax[near_limb]) + self.theMin]
        r3 = self.binAbss[np.argmax(self.binMed[near_limb]) + self.theMin]
        Processor.found_limb_radius = self.found_limb_radius = int(np.mean([r1, r2, r3]))
        self.prep_save_outs()
        
        self.lCut = int(self.found_limb_radius - 0.01 * self.rez)
        self.hCut = int(self.found_limb_radius + 0.01 * self.rez)
        
        # Split into three regions
        self.low_abs = self.binAbss[:self.lCut]
        self.low_max = self.binMax[:self.lCut]
        self.low_min = self.binMin[:self.lCut]
        
        self.mid_abs = self.binAbss[self.lCut:self.hCut]
        self.mid_max = self.binMax[self.lCut:self.hCut]
        self.mid_min = self.binMin[self.lCut:self.hCut]
        
        self.high_abs = self.binAbss[self.hCut:]
        self.high_max = self.binMax[self.hCut:]
        self.high_min = self.binMin[self.hCut:]
        
        # Filter the regions separately
        mode = 'nearest'
        
        self.low_max_filt = savgol_filter(self.low_max, lWindow, rank, mode=mode)
        self.mid_max_filt = savgol_filter(self.mid_max, mWindow, rank, mode=mode)
        self.high_max_filt = savgol_filter(self.high_max, hWindow, rank, mode=mode)
        
        self.low_min_filt = savgol_filter(self.low_min, lWindow, rank, mode=mode)
        self.mid_min_filt = savgol_filter(self.mid_min, mWindow, rank, mode=mode)
        self.high_min_filt = savgol_filter(self.high_min, hWindow, rank, mode=mode)
        
        # Fit the lowest region with a polynomial to make it much smoother
        degree = 5
        p = np.polyfit(self.low_abs, self.low_max_filt, degree)
        self.low_max_fit = np.polyval(p, self.low_abs)  # * 1.1
        p = np.polyfit(self.low_abs, self.low_min_filt, degree)
        self.low_min_fit = np.polyval(p, self.low_abs)
        
        ind = 10
        self.low_max_fit[0:ind] = self.low_max_fit[ind]
        self.low_min_fit[0:ind] = self.low_min_fit[ind]
        
        # Build output curves - max and min as a function of radius
        self.franken_abscissa = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.franken_max = np.hstack((self.low_max_fit, self.mid_max_filt, self.high_max_filt))
        self.franken_min = np.hstack((self.low_min_fit, self.mid_min_filt, self.high_min_filt))
        
        # Filter again to smooth boundaraies
        self.franken_max = self.fill_end(self.fill_start(savgol_filter(self.franken_max, fWindow, rank)))
        self.franken_min = self.fill_end(self.fill_start(savgol_filter(self.franken_min, fWindow, rank)))
        
        # Put the nans back in
        self.output_abscissa = np.arange(self.rez)
        self.frame_maximum[self.franken_abscissa] = self.franken_max
        self.frame_minimum[self.franken_abscissa] = self.franken_min
        pass
        
        # self.plot_radial_norm_keyframes(do=True, show=True, save=False, get_normed=False)
        
        # plt.plot(np.arange(self.rez), self.frame_maximum)
        # plt.plot(np.arange(self.rez), self.frame_minimum)
        # plt.show()
    
    def make_smoothed_curves(self):
        """Build the normalization arrays, treating the domain in 3 seperate regions"""
        
        # Put the nans back in
        # self.output_abscissa = np.arange(self.rez)
        # self.frame_maximum = self.franken_max
        # self.frame_minimum = self.franken_min
        
        ## Parameters
        self.highCut = 0.8 * self.rez
        
        # Savgol window size
        lWindow = 7  # 4 * self.extra_rez + 1
        mWindow = 7  # 4 * self.extra_rez + 1
        hWindow = 51  # 30 * self.extra_rez + 1
        fWindow = 7  # int(3 * self.extra_rez) + 1
        rank = 3
        
        ## Algorithm
        # Locate the Limb
        self.theMin = int(0.30 * self.rez)
        self.theMax = int(0.45 * self.rez)
        near_limb = np.arange(self.theMin, self.theMax)
        
        # Split the domain into three regions and treat seperately
        r1 = self.binAbss[np.argmax(self.binMean[near_limb]) + self.theMin]
        r2 = self.binAbss[np.argmax(self.binMax[near_limb]) + self.theMin]
        r3 = self.binAbss[np.argmax(self.binMed[near_limb]) + self.theMin]
        Processor.found_limb_radius = self.found_limb_radius = int(np.mean([r1, r2, r3]))
        self.prep_save_outs()
        
        self.lCut = int(self.found_limb_radius - 0.01 * self.rez)
        self.hCut = int(self.found_limb_radius + 0.01 * self.rez)
        
        # Split into three regions
        self.low_abs = self.binAbss[:self.lCut]
        self.low_max = self.binMax[:self.lCut]
        self.low_min = self.binMin[:self.lCut]
        
        self.mid_abs = self.binAbss[self.lCut:self.hCut]
        self.mid_max = self.binMax[self.lCut:self.hCut]
        self.mid_min = self.binMin[self.lCut:self.hCut]
        
        self.high_abs = self.binAbss[self.hCut:]
        self.high_max = self.binMax[self.hCut:]
        self.high_min = self.binMin[self.hCut:]
        
        # Filter the regions separately
        mode = 'nearest'
        
        self.low_max_filt = savgol_filter(self.low_max, lWindow, rank, mode=mode)
        self.mid_max_filt = savgol_filter(self.mid_max, mWindow, rank, mode=mode)
        self.high_max_filt = savgol_filter(self.high_max, hWindow, rank, mode=mode)
        
        self.low_min_filt = savgol_filter(self.low_min, lWindow, rank, mode=mode)
        self.mid_min_filt = savgol_filter(self.mid_min, mWindow, rank, mode=mode)
        self.high_min_filt = savgol_filter(self.high_min, hWindow, rank, mode=mode)
        
        # Fit the lowest region with a polynomial to make it much smoother
        # degree = 5
        # p = np.polyfit(self.low_abs, self.low_max_filt, degree)
        # self.low_max_fit = np.polyval(p, self.low_abs)  # * 1.1
        # p = np.polyfit(self.low_abs, self.low_min_filt, degree)
        # self.low_min_fit = np.polyval(p, self.low_abs)
        
        self.low_max_fit = self.low_max_filt
        self.low_min_fit = self.low_min_filt
        
        ind = 25
        self.low_max_fit[0:ind] = self.low_max_fit[ind]
        self.low_min_fit[0:ind] = self.low_min_fit[ind]
        
        # Build output curves - max and min as a function of radius
        self.franken_abscissa = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.franken_max = np.hstack((self.low_max_fit, self.mid_max_filt, self.high_max_filt))
        self.franken_min = np.hstack((self.low_min_fit, self.mid_min_filt, self.high_min_filt))
        
        # Filter again to smooth boundaraies
        # self.franken_max = self.fill_end(self.fill_start(savgol_filter(self.franken_max, fWindow, rank)))
        # self.franken_min = self.fill_end(self.fill_start(savgol_filter(self.franken_min, fWindow, rank)))
        
        # Put data where there is data
        self.output_abscissa = np.arange(self.rez)
        self.smooth_maximum = np.empty_like(self.output_abscissa, dtype=np.float32)
        self.smooth_minimum = np.empty_like(self.output_abscissa, dtype=np.float32)
        self.smooth_maximum.fill(np.nan)
        self.smooth_minimum.fill(np.nan)
        self.smooth_maximum[self.franken_abscissa] = self.franken_max
        self.smooth_minimum[self.franken_abscissa] = self.franken_min
        
        # self.plot_radial_norm_keyframes(do=True, show=True, save=False, get_normed=False)
        
        # plt.plot(np.arange(self.rez), self.smooth_maximum)
        # plt.plot(np.arange(self.rez), self.smooth_minimum)
        # plt.show()
        #
    
    

                
    ####################################
    ## Image Reduction Algorithms ##
    ####################################
    
    def coronaNorm(self, changed=None):
        """Normalize the in_object using the radial percentile curves"""
        # Collect Arrays
        self.init_images(changed)
        
        # Normalize Them
        self.execute_norm()
    
    def execute_norm(self):
        """Apply the Normalization to the Image Array"""
        if self.raster_extrema_curves():
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    # Standard Normalization Formula
                    self.changed_flat = self.norm_formula(self.changed_flat, self.norm_curve_min, self.norm_curve_max)
                except RuntimeWarning as e:
                    print(e)
    
    def coronagraph_touchup(self):
        """Deal with pixel outliers. Lots of adjustable parameters in here"""
        
        ## Deal with too hot things ##
        self.vmax = 1
        self.vmax_plot = 0.95  # np.max(changed_flat) #this is in the header of the imageprocessor now
        hotpowr = 1 / 2
        hot = self.changed_flat > self.vmax
        # self.changed_flat[hot] = self.changed_flat[hot] ** hotpowr
        
        ## Deal with too cold things ##
        self.vmin = 0.3
        self.vmin_plot = -0.05  # np.min(changed_flat)# 0.3# -0.03 #this is in the header of the imageprocessor now
        coldpowr = 1 / 2
        cold = self.changed_flat < self.vmin
        self.changed_flat[cold] = -((np.abs(self.changed_flat[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin
        
        ## Some Final Normalization ##      TODO: I think this might be breaking things!
        # self.changed_flat = self.normalize(self.changed_flat, high=99.99, low=1)
    
    def prep_output(self):
        
        self.mask_output()
        self.mirror_output()
        
        # Un-Flatten the Array
        self.changed = self.changed_flat.reshape(self.changed.shape)
        self.changed = np.sign(self.changed) * np.power(np.abs(self.changed), (1 / 5))
        self.changed = self.changed.astype('float32')
    
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
    
    # def plot_all(self, do=True):
    #     # self.plot_curves_2(do, False)
    #     # self.plot_curves(do, False)
    #     self.plot_radial_norm_keyframes(do, False)
    #     # self.SRNPlot()
    #     pass
    #     # plt.show()
    
    def plot_curves_2(self, do=True, show=False):
        if not do: return
        
        fig, ax = plt.subplots()
        ax.set_title("Plot Curves 2")
        
        # Plot the filtered curves
        ax.plot(self.low_abs, self.low_max_filt, lw=4)
        ax.plot(self.mid_abs, self.mid_max_filt, lw=4)
        ax.plot(self.high_abs, self.high_max_filt, lw=4)
        
        ax.plot(self.binAbss, self.binMax, label="Max")
        
        ax.plot(self.low_abs, self.low_min_filt, lw=4)
        ax.plot(self.mid_abs, self.mid_min_filt, lw=4)
        ax.plot(self.high_abs, self.high_min_filt, lw=4)
        
        ax.plot(self.binAbss, self.binMin, label="Min")
        
        ax.plot(self.low_abs, self.low_min_fit, c='k')
        ax.plot(self.low_abs, self.low_max_fit, c='k')
        
        ax.plot(self.output_abscissa, self.franken_max, label="FinalMax", lw=5)
        ax.plot(self.output_abscissa, self.franken_min, label="FinalMin", lw=5)
        
        # plt.plot(self.binAbss, self.binMean, label="Mid")
        # plt.plot(self.binAbss, self.binMed, label="Med")
        
        # plt.xlim([0.6*theMin,theMax*1.5])
        
        ax.legend()
        if show: plt.show()
    
    def plot_curves(self, do=False, show=False):
        """Plot the radial statistics from the binned array"""
        if not do: return
        
        fig, ax = plt.subplots()
        ax.set_title("Plot Curves")
        
        ax.plot(self.binAbss, self.binMax, label="Max")
        ax.plot(self.binAbss, self.binMin, label="Min")
        ax.plot(self.binAbss, self.binMean, label="Mid")
        ax.plot(self.binAbss, self.binMed, label="Med")
        
        ax.axvline(self.theMin)
        ax.axvline(self.theMax)
        
        ax.axvline(self.found_limb_radius)
        ax.axvline(self.lCut, ls=':')
        ax.axvline(self.hCut, ls=':')
        ax.set_xlim([self.lCut, self.hCut])
        ax.legend()
        if show:
            ax.show()
    
    def plot_radial_norm_keyframes(self, do=False, show=False, save=True, get_normed=False):
        """This plot is in radius and has a scatter plot
            overlaid with the norm curves as determined elsewhere"""
        if not do:
            return
        dprint("plot_radial_norm_keyframes")
        # Init the Plots
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
        
        ## Plot the extrema curves
        self.output_abscissa = np.arange(self.rez)
        ax0.plot(self.n2r(self.output_abscissa), self.outer_max, label="OuterMax", lw=3, c='orange')
        ax0.plot(self.n2r(self.output_abscissa), self.inner_max, label="InnerMax", lw=3, c='gold')
        ax0.plot(self.n2r(self.output_abscissa), self.inner_min, label="InnerMin", lw=3, c='cornflowerblue')
        ax0.plot(self.n2r(self.output_abscissa), self.outer_min, label="OuterMin", lw=3, c='b')
        # ax0.axvspan(self.n2r(self.found_limb_radius), ls='-', label="Limb")
        #
        # try:
        #     ## Vertical Lines
        #     ax0.axvline(self.n2r(self.lCut), ls=':')
        #     ax0.axvline(self.n2r(self.hCut), ls=':')
        #
        #     # ax.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        #     # ax0.axvline(self.n2r(self.highCut))
        # except Exception as e:
        #     print("SRNProc1::", e)
        #     # raise e
        
        ## Scatter Plot the intensities
        skip = 50
        ax0.scatter(self.n2r(self.rad_flat[::skip]), self.original.flatten()[::skip], c='k', s=2)
        
        if False:  # get_normed and self.changed.flatten() is None:
            self.image_modify()
            # if self.changed_flat is not None:
            ax1.scatter(self.n2r(self.rad_flat[::10]), self.changed.flatten()[::10], c='k', s=2)
        
        ## Plot Formatting
        ax0.set_title("Plot Stats")
        ax0.set_ylim((10 ** -2, 10 ** 4))
        ax0.legend()
        ax0.set_yscale('log')
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        # ax1.legend()
        ax1.axhline(1)
        ax1.axhline(0.05)
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax1.set_yscale('log')
        ax1.set_ylim((10 ** -2, 10 ** 2.5))
        
        plt.tight_layout()
        fig.set_size_inches(8, 12)
        
        self.force_save_radial_figures(save, fig, ax0, show)
        
        # ax.axvline(self.tRadius, c='r')
        
        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.binAbss[:-1] + 0.5, self.diff_mean, 'r:')
        
        ## Norm Curves
        
        # ax0.plot(self.n2r(self.low_abs), self.low_max, 'm', label="low_min/max")
        # ax0.plot(self.n2r(self.low_abs), self.low_min, 'm', label="")
        # # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # # plt.plot(self.low_abs, self.low_min_fit, 'r')
        #
        # ax0.plot(self.n2r(self.high_abs), self.high_max, 'c', label="high_min/max")
        # ax0.plot(self.n2r(self.high_abs), self.high_min, 'c', label="")
        #
        # ax0.plot(self.n2r(self.mid_abs), self.mid_max, 'y', label="mid_min/max")
        # ax0.plot(self.n2r(self.mid_abs), self.mid_min, 'y', label="")
        # plt.plot(self.high_abs, self.high_min_fit, 'r')
        # plt.plot(self.high_abs, self.high_max_fit, 'r')
        
        # try:
        #     ax0.plot(self.n2r(self.rendered_abss), self.outer_min, label="FinalMax", lw=4, c='blue')
        #     ax0.plot(self.n2r(self.rendered_abss), self.outer_max, label="FinalMin", lw=4, c='orange')
        # except Exception as e:
        #     print("SRNProc2::", e)
        #     # raise e
        
        # try:
        #     ax1.plot(self.n2r(self.output_abscissa), self.frame_maximum, 'g', label="Smoothed")
        #     ax1.plot(self.n2r(self.output_abscissa), self.frame_minimum, 'g')
        # except:
        #     ax1.plot(self.n2r(self.binAbss), self.frame_maximum, 'g', label="Smoothed")
        #     ax1.plot(self.n2r(self.binAbss), self.frame_minimum, 'g')
        
        # ax1.plot(binAbss, binMax, 'c')
        # ax1.plot(self.n2r(self.binAbss), self.binMin, 'm')
        # ax1.plot(self.n2r(self.binAbss), self.binMean, 'y')
        # ax1.plot(self.n2r(self.binAbss), self.binMax, 'b')
        # ax1.plot(binAbss, binMed, 'r')
        # ax1.plot(binAbss, frame_minimum, 'r')
        # ax1.set_ylim((-0.5, 2))
        # ax1.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        # ax1.set_xlim((0, self.n2r(self.highCut)))
        
        # ax1.axhline(self.vmax, c='r', label='Confinement')
        # ax1.axhline(self.vmin, c='r')
        # ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        # ax1.axhline(self.vmin_plot, c='orange')
        
        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))
        
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
        folder_name = "radial"
        file_name_1 = 'Radial_{}.png'.format(self.file_basename[:-5])
        save_path_1 = join(bs, folder_name, file_name_1)
        file_name_2 = 'zoom\\Radial_zoom_{}.png'.format(self.file_basename[:-5])
        save_path_2 = join(bs, folder_name, file_name_2)
        
        makedirs(dirname(save_path_1), exist_ok=True)
        plt.savefig(save_path_1)
        
        makedirs(dirname(save_path_2), exist_ok=True)
        # ax.set_xlim((0.9, 1.1))
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
        # return np.asarray(np.concatinate(xBox, yBox))
    
    # def SRNPlot(self):
    #     if self.outer_min is None:
    #         self.load_curves()
    #
    #     fig, (ax0) = plt.subplots(1, 1, "all")
    #     ax0.set_title("Plot Stats")
    #
    #     # scat = ax0.scatter(*self.get_points(0), c='k', s=2)
    #
    #     # ax0.scatter(self.n2r(self.rad_flat[::skip]), self.radBins_all[::skip], c='k', s=2)
    #
    #     ## Straight Lines
    #     ax0.axvline(self.n2r(self.found_limb_radius), ls='--', label="Limb")
    #     # ax.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
    #     ax0.axvline(self.n2r(self.lCut), ls=':')
    #     ax0.axvline(self.n2r(self.hCut), ls=':')
    #     # ax.axvline(self.tRadius, c='r')
    #     ax0.axvline(self.n2r(self.highCut))
    #
    #     # plt.plot()
    #     maxline = ax0.plot(self.rendered_abss, self.rendered_max_box[0], label="AvgMax", lw=5, c='g', zorder=10)[0]
    #     minline = ax0.plot(self.rendered_abss, self.rendered_min_box[0], label="AvgMin", lw=5, c='g', zorder=10)[0]
    #
    #     thisMax = ax0.plot(self.rendered_abss, self.rendered_max_box[0], label="ThisMax", lw=1, c='k', zorder=0)[0]
    #     thisMin = ax0.plot(self.rendered_abss, self.rendered_min_box[0], label="ThisMin", lw=1, c='k', zorder=0)[0]
    #
    #     # try:
    #     #     ax.plot(self.n2r(self.output_abscissa), self.frame_maximum, 'g', label="Smoothed")
    #     #     ax.plot(self.n2r(self.output_abscissa), self.frame_minimum, 'g')
    #     # except:
    #     #     ax.plot(self.n2r(self.binAbss), self.frame_maximum, 'g', label="Smoothed")
    #     #     ax.plot(self.n2r(self.binAbss), self.frame_minimum, 'g')
    #
    #     # plt.plot(binAbss, binMax, 'c')
    #     # plt.plot(self.binAbss, self.binMin, 'm')
    #     # plt.plot(self.binAbss, self.binMean, 'y')
    #     # plt.plot(binAbss, binMed, 'r')
    #     # plt.plot(self.binAbss, self.binMax, 'b')
    #     # plt.plot(binAbss, frame_minimum, 'r')
    #     # plt.ylim((-100, 10**3))
    #     # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
    #     # ax.set_xlim((0, self.n2r(self.highCut)))
    #     # ax0.legend()
    #     # fig.set_size_inches((8, 12))
    #     ax0.set_yscale('log')
    #
    #     # ax1.scatter(self.n2r(self.rad_flat[::10]), self.changed_flat[::10], c='k', s=2)
    #     # ax1.set_ylim((-0.25, 2))
    #     #
    #     # # ax1.axhline(self.vmax, c='r', label='Confinement')
    #     # # ax1.axhline(self.vmin, c='r')
    #     # # ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
    #     # # ax1.axhline(self.vmin_plot, c='orange')
    #     #
    #     # # locs = np.arange(self.rez)[::int(self.rez/5)]
    #     # # ax1.set_xticks(locs)
    #     # # ax1.set_xticklabels(self.n2r(locs))
    #     #
    #     # ax1.legend()
    #     # ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
    #     # ax1.set_ylabel(r"Normalized Intensity")
    #     ax0.set_ylabel(r"Absolute Intensity (Counts)")
    #
    #     plt.tight_layout()
    #
    #     # plt.show(block=True)
    #
    #     def init():
    #
    #         maxline.set_ydata(self.rendered_max_box[0])
    #         minline.set_ydata(self.rendered_min_box[0])
    #         thisMax.set_ydata(self.rendered_max_box[0])
    #         thisMin.set_ydata(self.rendered_min_box[0])
    #
    #         # scat.set_offsets(self.get_points(0))
    #
    #         # tit.set_text("Time: {}".format(self.lastIndex))
    #         # tit2.set_text("Time: {}".format(self.lastIndex))
    #         # tickerLine.set_ydata(self.lastIndex)
    #         # quad.set_array(self.slideArray[self.lastIndex][:-1, :-1].flatten())
    #         # line1.set_xdata(self.slideCents[self.lastIndex])
    #         # line9.set_ydata(self.lineArray[self.lastIndex])
    #
    #         return maxline, minline, thisMax, thisMin,
    #
    #     # Animate
    #     from matplotlib import animation
    #
    #     def animate(i):
    #         maxline.set_ydata(self.rendered_max_box[i])
    #         minline.set_ydata(self.rendered_min_box[i])
    #         if i > 0:
    #             thisMax.set_ydata(self.rendered_max_box[i - 1])
    #             thisMin.set_ydata(self.rendered_min_box[i - 1])
    #         # tit.set_text("Time: {}".format(i))
    #         # tit2.set_text("Time: {}".format(i))
    #         # tickerLine.set_ydata(i)
    #         # quad.set_array(self.slideArray[i][:-1, :-1].flatten())
    #         # line1.set_xdata(self.slideCents[i])
    #         # line9.set_ydata(self.lineArray[i]/np.sum(self.lineArray[i]))
    #
    #         return maxline, minline, thisMax, thisMin,
    #
    #     anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                                    frames=np.arange(self.firstIndex, self.lastIndex),
    #                                    repeat=True, interval=100, blit=True)
    #
    #     # grid.maximizePlot()
    #     fig.set_size_inches(14, 6)
    #     plt.subplots_adjust(
    #             top=0.911,
    #             bottom=0.082,
    #             left=0.06,
    #             right=0.985,
    #             hspace=0.2,
    #             wspace=0.233
    #     )
    #     # plt.show()
    #     # fname = r"renders\{}_{}.mp4".format(self.ions[0]['ionString'], name)
    #     # anim.save(filename=fname, writer='ffmpeg', bitrate=1000)
    #     # plt.close()
    #     # print('Save Complete')
    #     plt.tight_layout()
    #     plt.show(block=True)
    #
    #     return
    
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
    def norm_formula(flat_image, the_min, the_max):
        """Standard Normalization Formula"""
        top = np.subtract(flat_image, the_min)
        bottom = np.subtract(the_max, the_min)
        return np.divide(top, bottom)

    
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

#######################
#######################
## Child Classes ##
#######################
#######################



### Depricated Code Segments  ##################################################


# # Locate the Noise Floor
# noiseMin = 550 * self.extra_rez - self.hCut
# near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
# self.diff_max_abs = self.high_abs[near_noise]
# self.diff_max = np.diff(high_max_filt)[near_noise]
# self.diff_max += np.absolute(np.nanmin(self.diff_max))
# self.diff_max /= np.nanmean(self.diff_max) / 100
# self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
# self.noise_radii = 565 * self.extra_rez

# for i in range(len(self.rad_flat)):
#     self.radBins[self.binInds[i]].append(self.dat_flat[i])
# for i in range(len(self.rad_flat)):
#     index = np.floor(self.rad_flat[i]).astype(np.int32)
#     self.radBins[index].append(self.dat_flat[i])

# if not self.original:
#     self.in_name = 0
#     self.load_fits_image()
#     self.init_radius_array()
#

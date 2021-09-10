import os
from os import makedirs
from os.path import join, dirname
from time import sleep

import numpy as np
import matplotlib as mpl
from matplotlib.cbook import mkdirs
from scipy.signal import savgol_filter
from copy import copy
import warnings

warnings.filterwarnings("ignore")

mpl.use("qt5agg")
import matplotlib.pyplot as plt

plt.ioff()

from processor.Processor import Processor


class SRNProcessor(Processor):
    """This is the primary code used in the RadialFiltProcessor"""
    out_name = None
    name = "Default"
    filt_name = '  SRN Radial Base Class'
    description = "Create and Apply the Radial SRN Curves"
    
    do_png = False
    renew_mask = True
    show_plots = True
    image_data = None
    
    multiple_minimum_curves = []
    multiple_maximum_curves = []
    rendered_min_box = []
    rendered_max_box = []
    n_keyframes = 0
    
    running_min = None
    running_max = None
    rendered_min = None
    rendered_max = None
    rendered_abss = None
    
    firstIndex = 0
    lastIndex = 0
    
    radius = None
    
    rad_flat = None
    binInds = None
    more_rez = None
    radBins_all = []
    limb_radii = None
    
    def __init__(self, fits_path=None, in_name=-1, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp)
        # Parse Inputs
        
        self.flat_image = None
        self.dat_corona = None
        self.rez = None
        self.fits_path = fits_path
        self.in_name = in_name
        self.out_name = "SRN"
        self.show = show
        self.verb = verb
        self.do_orig = orig
        self.center = None
        self.original = None
        self.changed = None
        self.can_use_keyframes = False
        self.vignette_mask = None
        self.s_radius = 400  # * self.extra_rez
        self.tRadius = self.s_radius * 1.28
        self.dont_ignore = True
        
        self.binInds = None
        self.more_rez = None
        self.radBins = None
        
        self.fakeMax = None
        self.fakeMin = None
        
        self.rastered_min = None
        self.rastered_max = None
    
    ## Structure
    def do_fits_function(self, fits_path=None, in_name=None):
        """Calls the do_work function on a single fits path"""
        # Run the Reduction Algorithm
        if fits_path is not None:  # Load the input fits path
            self.fits_path = fits_path
        # Define which images to use
        if self.can_use_keyframes:
            if self.fits_path in self.keyframes and self.load_fits_image():
                return self.do_work()  # Do the work on the fits files
        elif self.load_fits_image():
            return self.do_work()
        else:
            return None
    
    def setup(self):
        """Do prep work once before the main algorithm"""
        raise NotImplementedError
    
    def do_work(self):
        """Do whatever you want to each image in the directory"""
        raise NotImplementedError
    
    def cleanup(self):
        """Runs once after all the images have been modified with do_work"""
        raise NotImplementedError
    

    
    ## Top-Level
    def image_learn(self):
        """Analyze the input to help make normalization curves"""
        self.make_radius_array()  # Assign Each Pixel its Radius Value
        self.bin_radially()  # Create a cloud of intesity values for each radial bin
        self.radial_statistics()  # Find mean and percentiles vs height
        self.make_curves()  # Build smooth curves based on the statistics
        self.add_to_keyframes()  # Update the running curves
        self.plot_all(self.show_plots)  # Plot Extra Details
    
    def image_modify(self):
        """Perform the actual normalization on the input array"""
        self.make_radius_array()  # Assign Each Pixel its Radius Value
        self.raster_curves()
        self.coronaNorm()  # Use curves to rescale the in_object
        self.coronagraph_touchup()  # Deal with some outliers
        self.vignette()  # Truncate the in_object above given radius
    
    def add_to_keyframes(self):
        """Records the current analysis as one of the radial samples"""
        SRNProcessor.rendered_abss = self.rendered_abss = self.fakeAbss
        SRNProcessor.n_keyframes += 1
        SRNProcessor.lastIndex += 1
        self.skipped -= 1
    
        # SRNProcessor.running_max
    
        # Use Average Curves
        if SRNProcessor.running_max is None:
            SRNProcessor.running_min = self.fakeMin + 0
            SRNProcessor.running_max = self.fakeMax + 0
        else:
            SRNProcessor.running_min += self.fakeMin
            SRNProcessor.running_max += self.fakeMax
    
    
        SRNProcessor.rendered_min = self.rendered_min = SRNProcessor.running_min / SRNProcessor.n_keyframes
        SRNProcessor.rendered_min = self.rendered_max = SRNProcessor.running_max / SRNProcessor.n_keyframes
    
        self.rendered_min_box.append(self.rendered_min)
        self.rendered_max_box.append(self.rendered_max)
    
        self.multiple_minimum_curves.append(self.fakeMin0)
        self.multiple_maximum_curves.append(self.fakeMax0)
    
    def save_curves(self):  #
        """Save the curves so they don't have to be recalculated"""
        print(" *    Saving Radial Curves...", end='')
        file_name = self.params.curve_path()
        max_curve = self.rendered_max
        min_curve = self.rendered_min
        if self.limb_radii is None:
            limb_out = None
        else:
            limb_out = np.ones_like(min_curve) * self.limb_radii
        
        out_array = np.asarray((min_curve, max_curve, limb_out))  # , abss))
        if None not in out_array:
            np.savetxt(file_name, out_array)
            print("Success!")
        else:
            print("Skipping!")
    
    def load_curves(self):
        """Load the curves so they don't have to be recalculated"""
        file_name = self.params.curve_path()
        # if not self.original:
        #     self.in_name = 0
        #     self.load_fits_image()
        #     self.make_radius_array()
        #
        if os.path.exists(file_name):
            print(" *    Loading Radial Curves...", end='')
            try:
                min_curve, max_curve, limb_radii = np.loadtxt(file_name)
                self.limb_radii = limb_radii[0]
                self.rendered_min = min_curve
                self.rendered_max = max_curve
                # self.raster_curves()
                self.super_flush("Success!")
            except ValueError as e:
                print("Failed: {}".format(e))
                
                self.image_learn()
                self.save_curves()
    
    def raster_curves(self):
        """Raster out the min/max curves from the rendered version"""
        if self.rastered_min is None and self.rendered_min is not None:
            # print("Rastering...", end='')
            # self.rendered_abss = self.fakeAbss
            self.rastered_min = np.squeeze(self.rendered_min[self.binInds])
            self.rastered_max = np.squeeze(self.rendered_max[self.binInds])
    
    # Analysis
    def make_radius_array(self, vignette_radius=1.2, s_radius=400, t_factor=1.28, force=False):
        """Build an r-coordinate array of shape(in_object)"""
        
        if self.radius is None or force or self.changed.shape[0] != self.rez:
            self.rez = self.changed.shape[0]
            
            self.init_curves()
            
            if self.center is None:
                self.center = [self.rez / 2, self.rez / 2]
            
            xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
            xc, yc = xx - self.center[0], yy - self.center[1]
            
            self.s_radius = s_radius
            self.tRadius = self.s_radius * t_factor
            self.radius = np.sqrt(xc * xc + yc * yc)
            
            self.vignette_mask = self.radius > (int(vignette_radius * self.rez // 2))
            
            self.sort_radially()
            0
    
    def init_curves(self):
        """These are the main output curves"""
        # print("INit CURVES")
        self.fakeMax = np.empty(self.rez)
        self.fakeMax.fill(np.nan)
        self.fakeMin = np.empty(self.rez)
        self.fakeMin.fill(np.nan)
    
    def sort_radially(self):
        """ Flatten the in_object and sort by pixel radius """
        self.rad_flat = self.radius.flatten()
        self.binInds = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        
        self.more_rez = np.max(self.binInds) + 10
        self.radBins = [[] for x in np.arange(self.more_rez)]
    
    def bin_radially(self):  # TODO Make this much faster
        """Bin the intensities by radius """
        self.cut_pixels = 100
        for binI, dat in zip(self.binInds[::self.cut_pixels], self.changed.flatten()[::self.cut_pixels]):
            try:
                self.radBins[binI].append(dat)
                # self.radBins_all[binI].append(dat)
            except Exception as e:
                print("bin_radially:: ", e)
        
        # self.radBins_all.append(self.radBins)
        
        # for i in range(len(self.rad_flat)):
        #     self.radBins[self.binInds[i]].append(self.dat_flat[i])
        # for i in range(len(self.rad_flat)):
        #     index = np.floor(self.rad_flat[i]).astype(np.int32)
        #     self.radBins[index].append(self.dat_flat[i])
    
    def radial_statistics(self):  # TODO Make this much faster
        """ Find the statistics in each radial bin"""
        # moreRez = self.radBins
        self.binMax = np.zeros(self.more_rez)
        self.binMin = np.zeros(self.more_rez)
        self.binMid = np.zeros(self.more_rez)
        self.binMed = np.zeros(self.more_rez)
        self.radAbss = np.arange(self.more_rez)
        
        for ii, it in enumerate(self.radBins):
            # For each radial bin
            item = np.asarray(it)
            idx = np.isfinite(item)
            finite = item[idx]
            idx2 = np.nonzero(finite)
            subItems = finite[idx2]
            
            # Do statistics
            if len(subItems) > 0:
                self.binMax[ii] = np.percentile(subItems, 98)  # np.nanmax(subItems)
                self.binMin[ii] = np.percentile(subItems, 5)  # np.min(subItems)
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
        
        pass
    
    def make_curves(self):
        """Build the normalization arrays, treating the domain in 3 seperate regions"""
        
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
        r1 = self.radAbss[np.argmax(self.binMid[near_limb]) + self.theMin]
        r2 = self.radAbss[np.argmax(self.binMax[near_limb]) + self.theMin]
        r3 = self.radAbss[np.argmax(self.binMed[near_limb]) + self.theMin]
        Processor.limb_radii = self.limb_radii = int(np.mean([r1, r2, r3]))
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
        
        # Filter the regions separately
        mode = 'nearest'
        
        self.low_max_filt = savgol_filter(self.low_max, lWindow, rank, mode=mode)
        
        self.mid_max_filt = savgol_filter(self.mid_max, mWindow, rank, mode=mode)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        
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
        self.fakeAbss0 = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax0 = np.hstack((self.low_max_fit, self.mid_max_filt, self.high_max_filt))
        self.fakeMin0 = np.hstack((self.low_min_fit, self.mid_min_filt, self.high_min_filt))
        
        # Filter again to smooth boundaraies
        self.fakeMax0 = self.fill_end(self.fill_start(savgol_filter(self.fakeMax0, fWindow, rank)))
        self.fakeMin0 = self.fill_end(self.fill_start(savgol_filter(self.fakeMin0, fWindow, rank)))
        
        # Put the nans back in
        self.fakeAbss = np.arange(self.rez)
        self.fakeMax[self.fakeAbss0] = self.fakeMax0
        self.fakeMin[self.fakeAbss0] = self.fakeMin0
        
        # self.plot_stats(do=True, show=True, save=False, get_normed=False)
        pass
        
        # plt.plot(np.arange(self.rez), self.fakeMax)
        # plt.plot(np.arange(self.rez), self.fakeMin)
        # plt.show()
    
    ## Modify Images ##
    
    def n2r(self, n):
        if n is None:
            n=0
        
        return n / Processor.limb_radii
    
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
    def normalize(image, high=98, low=15):
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
        
        # # Locate the Noise Floor
        # noiseMin = 550 * self.extra_rez - self.hCut
        # near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        # self.diff_max_abs = self.high_abs[near_noise]
        # self.diff_max = np.diff(high_max_filt)[near_noise]
        # self.diff_max += np.absolute(np.nanmin(self.diff_max))
        # self.diff_max /= np.nanmean(self.diff_max) / 100
        # self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        # self.noise_radii = 565 * self.extra_rez
        
        # Reduction
    
    def coronaNorm(self, changed=None):
        """Normalize the in_object using the radial percentile curves"""
        
        # Collect Arrays
        self.prep_arrays(changed)
        
        # Normalize Them
        self.execute_norm()
    
    def prep_arrays(self, changed=None):
        """Get all the variables ready for the normalization"""
        if changed is not None:
            self.changed = changed
        self.changed = self.changed.astype('float32')
        self.changed[self.changed == 0] = np.nan
        self.flat_image = self.changed.flatten()
        self.raster_curves()
    
    def execute_norm(self):
        """Apply the Normalization to the Image Array"""
        self.dat_corona = None
        # if self.rastered_min is None or len(self.rastered:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # Standard Normalization Formula
                self.dat_corona = self.norm_formula(self.flat_image, self.rastered_min, self.rastered_max)
            except RuntimeWarning as e:
                print(e)
    
    @staticmethod
    def norm_formula(flat_image, the_min, the_max):
        """Standard Normalization Formula"""
        top = np.subtract(flat_image, the_min)
        bottom = np.subtract(the_max, the_min)
        return np.divide(top, bottom)
    
    def coronagraph_touchup(self):
        """Deal with pixel outliers. Lots of adjustable parameters in here"""
        
        # Deal with too hot things
        self.vmax = 1
        self.vmax_plot = 0.95  # np.max(dat_corona) #this is in the header of the imageprocessor now
        hotpowr = 1 / 2
        hot = self.dat_corona > self.vmax
        # self.dat_corona[hot] = self.dat_corona[hot] ** hotpowr
        
        # Deal with too cold things
        self.vmin = 0.3
        self.vmin_plot = -0.05  # np.min(dat_corona)# 0.3# -0.03 #this is in the header of the imageprocessor now
        coldpowr = 1 / 2
        cold = self.dat_corona < self.vmin
        self.dat_corona[cold] = -((np.abs(self.dat_corona[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin
        self.dat_coronagraph = self.dat_corona
        dat_corona_square = self.dat_corona.reshape(self.changed.shape)
        
        # Some More Normalization
        dat_corona_square = np.sign(dat_corona_square) * np.power(np.abs(dat_corona_square), (1 / 5))
        self.changed = self.normalize(self.changed, high=99.99, low=0)
        dat_corona_square = self.normalize(dat_corona_square, high=99.99, low=1)
        
        # Allows you to only show sub-sections of the in_object as reduced images
        if self.renew_mask:
            self.corona_mask = self.get_mask(self.changed)
            self.renew_mask = False
        
        # Allows you to mirror horizontally, with only one half rfeduced
        do_mirror = False
        if do_mirror:
            # Do stuff
            xx, yy = self.corona_mask.shape[0], int(self.corona_mask.shape[1] / 2)
            #
            newDat = self.changed[self.corona_mask]
            grid = newDat.reshape(xx, yy)
            # if self.
            flipped = np.fliplr(grid)
            self.changed[~self.corona_mask] = flipped.flatten()  # np.flip(newDat)
        
        # Clean Outputs
        self.changed[self.corona_mask] = dat_corona_square[self.corona_mask]
        self.changed = self.changed.astype('float32')
    
    def vignette(self):
        """Truncate the in_object above a certain radis"""
        self.changed[self.vignette_mask] = np.nan
        self.original[self.vignette_mask] = np.nan
    
    def get_mask(self, dat_out):
        """ Generates a mask that defines which portion of the in_object will be modified"""
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
    
    ## Plotters
    
    def plot_all(self, do=True):
        # self.plot_curves_2(do, False)
        # self.plot_curves(do, False)
        self.plot_stats(do, False)
        # self.SRNPlot()
        pass
        # plt.show()
    
    def plot_curves_2(self, do=True, show=False):
        if not do: return
        
        fig, ax = plt.subplots()
        ax.set_title("Plot Curves 2")
        
        # Plot the filtered curves
        ax.plot(self.low_abs, self.low_max_filt, lw=4)
        ax.plot(self.mid_abs, self.mid_max_filt, lw=4)
        ax.plot(self.high_abs, self.high_max_filt, lw=4)
        
        ax.plot(self.radAbss, self.binMax, label="Max")
        
        ax.plot(self.low_abs, self.low_min_filt, lw=4)
        ax.plot(self.mid_abs, self.mid_min_filt, lw=4)
        ax.plot(self.high_abs, self.high_min_filt, lw=4)
        
        ax.plot(self.radAbss, self.binMin, label="Min")
        
        ax.plot(self.low_abs, self.low_min_fit, c='k')
        ax.plot(self.low_abs, self.low_max_fit, c='k')
        
        ax.plot(self.fakeAbss, self.fakeMax0, label="FinalMax", lw=5)
        ax.plot(self.fakeAbss, self.fakeMin0, label="FinalMin", lw=5)
        
        # plt.plot(self.radAbss, self.binMid, label="Mid")
        # plt.plot(self.radAbss, self.binMed, label="Med")
        
        # plt.xlim([0.6*theMin,theMax*1.5])
        
        ax.legend()
        if show: plt.show()
    
    def plot_curves(self, do=False, show=False):
        """Plot the radial statistics from the binned array"""
        if not do: return
        
        fig, ax = plt.subplots()
        ax.set_title("Plot Curves")
        
        ax.plot(self.radAbss, self.binMax, label="Max")
        ax.plot(self.radAbss, self.binMin, label="Min")
        ax.plot(self.radAbss, self.binMid, label="Mid")
        ax.plot(self.radAbss, self.binMed, label="Med")
        
        ax.axvline(self.theMin)
        ax.axvline(self.theMax)
        
        ax.axvline(self.limb_radii)
        ax.axvline(self.lCut, ls=':')
        ax.axvline(self.hCut, ls=':')
        ax.set_xlim([self.lCut, self.hCut])
        ax.legend()
        if show:
            ax.show()
    
    def plot_stats(self, do=False, show=False, save=True, get_normed=True):
        """This plot is in radius and has a scatter plot
            overlaid with the norm curves as determined elsewhere"""
        if not do: return
        
        fig, (ax0, ax1) = plt.subplots(2,1, sharex=True)
        ax0.set_title("Plot Stats")
        
        ## Scatter Plot
        skip = 50
        ax0.scatter(self.n2r(self.rad_flat[::skip]), self.changed.flatten()[::skip], c='k', s=2)
        
        ## Straight Lines
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        # ax.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        
        try:
            ax0.axvline(self.n2r(self.lCut), ls=':')
            ax0.axvline(self.n2r(self.hCut), ls=':')
            # ax.axvline(self.tRadius, c='r')
            ax0.axvline(self.n2r(self.highCut))
            
            # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
            # plt.plot(self.radAbss[:-1] + 0.5, self.diff_mean, 'r:')
            
            ## Curves
            
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
            
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, label="ThisMax", lw=2, c='cornflowerblue')
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, label="ThisMin", lw=2, c='gold')
        except Exception as e:
            print("SRNProc1::", e)
            raise e
        
        try:
            if self.rendered_min is None:
                self.raster_curves()
            ax0.plot(self.n2r(self.rendered_abss), self.rendered_min, label="FinalMax", lw=4, c='blue')
            ax0.plot(self.n2r(self.rendered_abss), self.rendered_max, label="FinalMin", lw=4, c='orange')
        except Exception as e:
            print("SRNProc2::", e)
            # raise e
        
        # try:
        #     ax1.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax1.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        # except:
        #     ax1.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax1.plot(self.n2r(self.radAbss), self.fakeMin, 'g')
        
        if get_normed:
            self.image_modify()
        
        if self.dat_coronagraph is not None:
            ax1.scatter(self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c='k', s=2)
        
        # ax1.plot(radAbss, binMax, 'c')
        # ax1.plot(self.n2r(self.radAbss), self.binMin, 'm')
        # ax1.plot(self.n2r(self.radAbss), self.binMid, 'y')
        # ax1.plot(self.n2r(self.radAbss), self.binMax, 'b')
        # ax1.plot(radAbss, binMed, 'r')
        # ax1.plot(radAbss, fakeMin, 'r')
        # ax1.set_ylim((-0.5, 2))
        # ax1.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        # ax1.set_xlim((0, self.n2r(self.highCut)))


        ax1.axhline(1)
        ax1.axhline(0.05)
        # ax1.axhline(self.vmax, c='r', label='Confinement')
        # ax1.axhline(self.vmin, c='r')
        # ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        # ax1.axhline(self.vmin_plot, c='orange')

        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))

        # ax1.legend()
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax1.set_yscale('log')
        ax1.set_ylim((10 ** -2, 10 ** 2.5))

        
        ax0.set_ylim((10 ** -2, 10 ** 4))
        ax0.legend()
        ax0.set_yscale('log')
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        plt.tight_layout()
        fig.set_size_inches(8, 12)
        
        first = True
        while True:
            try:
                self.save_figures(save, fig, ax0, show)
                if not first: print("  Thanks, good job.\n")
                break
            except OSError as e:
                if first:
                    print("\n\n", e)
                    print("  !!!!!!! Close the Dang Plot!", end='')
                    first = False
                print('.', end='')

        
    def save_figures(self, do=False, fig=None, ax=None, show=False):
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
    
    def SRNPlot(self):
        if self.rendered_min is None:
            self.load_curves()
        
        fig, (ax0) = plt.subplots(1, 1, "all")
        ax0.set_title("Plot Stats")
        
        # scat = ax0.scatter(*self.get_points(0), c='k', s=2)
        
        # ax0.scatter(self.n2r(self.rad_flat[::skip]), self.radBins_all[::skip], c='k', s=2)
        
        ## Straight Lines
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        # ax.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=':')
        ax0.axvline(self.n2r(self.hCut), ls=':')
        # ax.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))
        
        # plt.plot()
        maxline = ax0.plot(self.rendered_abss, self.rendered_max_box[0], label="AvgMax", lw=5, c='g', zorder=10)[0]
        minline = ax0.plot(self.rendered_abss, self.rendered_min_box[0], label="AvgMin", lw=5, c='g', zorder=10)[0]
        
        thisMax = ax0.plot(self.rendered_abss, self.rendered_max_box[0], label="ThisMax", lw=1, c='k', zorder=0)[0]
        thisMin = ax0.plot(self.rendered_abss, self.rendered_min_box[0], label="ThisMin", lw=1, c='k', zorder=0)[0]
        
        # try:
        #     ax.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        # except:
        #     ax.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax.plot(self.n2r(self.radAbss), self.fakeMin, 'g')
        
        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMid, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        # ax.set_xlim((0, self.n2r(self.highCut)))
        # ax0.legend()
        # fig.set_size_inches((8, 12))
        ax0.set_yscale('log')
        
        # ax1.scatter(self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c='k', s=2)
        # ax1.set_ylim((-0.25, 2))
        #
        # # ax1.axhline(self.vmax, c='r', label='Confinement')
        # # ax1.axhline(self.vmin, c='r')
        # # ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        # # ax1.axhline(self.vmin_plot, c='orange')
        #
        # # locs = np.arange(self.rez)[::int(self.rez/5)]
        # # ax1.set_xticks(locs)
        # # ax1.set_xticklabels(self.n2r(locs))
        #
        # ax1.legend()
        # ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        # ax1.set_ylabel(r"Normalized Intensity")
        ax0.set_ylabel(r"Absolute Intensity (Counts)")
        
        plt.tight_layout()
        
        # plt.show(block=True)
        
        def init():
            
            maxline.set_ydata(self.rendered_max_box[0])
            minline.set_ydata(self.rendered_min_box[0])
            thisMax.set_ydata(self.rendered_max_box[0])
            thisMin.set_ydata(self.rendered_min_box[0])
            
            # scat.set_offsets(self.get_points(0))
            
            # tit.set_text("Time: {}".format(self.lastIndex))
            # tit2.set_text("Time: {}".format(self.lastIndex))
            # tickerLine.set_ydata(self.lastIndex)
            # quad.set_array(self.slideArray[self.lastIndex][:-1, :-1].flatten())
            # line1.set_xdata(self.slideCents[self.lastIndex])
            # line9.set_ydata(self.lineArray[self.lastIndex])
            
            return maxline, minline, thisMax, thisMin,
        
        # Animate
        from matplotlib import animation
        
        def animate(i):
            maxline.set_ydata(self.rendered_max_box[i])
            minline.set_ydata(self.rendered_min_box[i])
            if i > 0:
                thisMax.set_ydata(self.rendered_max_box[i - 1])
                thisMin.set_ydata(self.rendered_min_box[i - 1])
            # tit.set_text("Time: {}".format(i))
            # tit2.set_text("Time: {}".format(i))
            # tickerLine.set_ydata(i)
            # quad.set_array(self.slideArray[i][:-1, :-1].flatten())
            # line1.set_xdata(self.slideCents[i])
            # line9.set_ydata(self.lineArray[i]/np.sum(self.lineArray[i]))
            
            return maxline, minline, thisMax, thisMin,
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=np.arange(self.firstIndex, self.lastIndex),
                                       repeat=True, interval=100, blit=True)
        
        # grid.maximizePlot()
        fig.set_size_inches(14, 6)
        plt.subplots_adjust(
                top=0.911,
                bottom=0.082,
                left=0.06,
                right=0.985,
                hspace=0.2,
                wspace=0.233
        )
        # plt.show()
        # fname = r"renders\{}_{}.mp4".format(self.ions[0]['ionString'], name)
        # anim.save(filename=fname, writer='ffmpeg', bitrate=1000)
        # plt.close()
        # print('Save Complete')
        plt.tight_layout()
        plt.show(block=True)
        
        return
    
    # Helpers


class SRNSingleProcessor(SRNProcessor):
    name = out_name = 'SRN'
    name = filt_name = 'SRN Single Shot Processor'
    description = "Create and Apply the Radial SRN Curves"
    progress_verb = 'Processing'
    finished_verb = "Applied"
    show_plots = True
    
    def __init__(self, fits_path=None, in_name=-1, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
        self.first = True
        self.go_ahead = True
        
    def setup(self):
        pass
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        self.image_learn()  # Analyze the input to help make normalization curves
        # self.image_modify()  # Actually Normalize This Image
        return self.changed
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        # print("Save/load!")
        # self.save_curves()
        # self.load_curves()
        pass
    



class SRNpreProcessor(SRNProcessor):
    """Analyzes the whole dataset and builds curves"""
    out_name = None
    name = filt_name = 'SRN Radial Pre-Processor'
    description = "Create the Radial SRN Curves"
    progress_verb = 'Analyzing'
    finished_verb = "Analyzed"
    show_plots = True
    
    def __init__(self, fits_path=None, in_name=-1, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
        self.first = True
        self.go_ahead = True
    
    def setup(self):
        self.can_use_keyframes = True
        self.load()
        set_to_make = self.params.remake_norm_curves() or self.reprocess_mode()
        not_made_yet = not os.path.exists(self.params.curve_path()) or self.rendered_min is None
        frame_is_not_loaded = self.original is None
        self.go_ahead = set_to_make or not_made_yet or frame_is_not_loaded
        self.print_keyframes()
        
        # print("GO AHEAD: ", self.go_ahead, set_to_make, not_made_yet, frame_is_not_loaded)
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.go_ahead:
            self.image_learn()
            self.image_modify()
            
        return None
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        self.save_curves()
        
        if self.go_ahead:
            pass
            # self.SRNPlot()
        pass


class SRNradialFiltProcessor(SRNProcessor):
    """Uses radial curves to normalize images"""
    name = out_name = 'SRN'
    filt_name = 'SRN Radial Filter'
    description = "Filter the Images Radially with SRN"
    progress_verb = 'Filtering'
    finished_verb = "Filtered"
    can_use_keyframes = False
    
    def __init__(self, fits_path=None, in_name=-1, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
        self.first = True
        self.go_ahead = True

    def setup(self):
        self.load_curves()
        self.super_flush()
        pass
    
    def do_work(self):
        self.image_modify()
        if self.first:
            self.plot_stats(do=True, show=True, save=False)
            self.first = False
        return self.changed
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        pass

from os.path import dirname

import numpy as np
import os
import matplotlib as mpl

from utils.file_util import load_fits_field, open_fits_hdul

mpl.use("qt5agg")
import matplotlib.pyplot as plt
from time import strftime
from datetime import timedelta
from scipy.signal import savgol_filter
import astropy.units as u
from astropy.io import fits
import datetime
from copy import copy

from .color_tables import aia_color_table
import warnings

warnings.filterwarnings("ignore")
plt.ioff()


class Modify:
    renew_mask = True
    image_data = None
    name = "Default"
    def __init__(self, fits_path, in_name=-1, orig=False, show=False, verb=False):
        """Initialize the main class"""
        self.fits_path = fits_path
        self.in_name = in_name
    
        frame, wave, t_rec, center = load_fits_field(fits_path, in_name)
        self.image_data = str(wave), fits_path, t_rec, frame.shape
        # self.name = self.image_data[0]
        self.center = center
        
        self.original = frame
        self.changed = copy(self.original)
        
        # Parse Inputs
        self.show = show
        self.verb = verb
        self.do_orig = orig
        
        self.confirm_centerpoint()
        
        # Run the Reduction Algorithm
        self.image_modify()  # Primary Algorithm
        self.plot_and_save()
    
        if self.verb: print("Done")
    
        
    def test(self):
        """Run the test case if no input is provided"""
        if self.verb: print("Running Test Case")
        image = load_fits_field("data/0171_MR.fits")
        self.show = True
        return image

    def parse_input_type(self, image):
        """Determine what kind of input image was provided and open it appropriately"""
        # Load the File
        if image is None:
            # Run the Test Case
            self.original = self.test()
        elif type(image) in [str]:
            # Load the file at input path
            path = image
            self.original = load_fits_field(path, -1)[0]
            
        elif type(image) in [np.array, np.ndarray]:
            self.original = image
        else:
            raise TypeError("Invalid Input Data: {}".format(type(image)))
        
        self.clean_input()
        
        
    def confirm_centerpoint(self):
        image_edge = self.original.shape
        center_given = np.abs(self.center)
        
        Top_Tolerance = 0.65
        Bottom_Tolerance = 0.35
        count=0
        while count < 100:
            ratio = center_given/image_edge
            if np.array(ratio > Top_Tolerance).any():
                center_given *= 0.5
            elif np.array(ratio < Bottom_Tolerance).any():
                center_given *= 2
            else:
                break
                
        self.center = center_given
    
    def def_data(self, hdul):
        """Use Defaults Values for Data"""
        try:
            shape = hdul[0].data.shape
        except:
            shape = hdul.shape
        
        wave = 171
        full_name = str(wave)
        save_path = str(wave)
        time_string = '2021-02-02T19:00:12.57Z'
        
        self.image_data = full_name, save_path, time_string, shape
        return self.image_data
    
    def get(self):
        """Returns the reduced image array"""
        return self.changed
    
    
    
    def image_modify(self):
        """Perform the image normalization on the input array"""
        self.make_radius_array()  # Assign Each Pixel its Radius Value
        self.remove_offset()  # Additive Shift of input array
        # self.noise_gate()
        self.sort_radially()  # Build Flattened and Sorted Intensity Arrays
        self.bin_radially()  # Create a cloud of intesity values for each radial bin
        self.radial_statistics()  # Find mean and percentiles vs height
        self.make_curves()  # Build smooth curves based on the statistics
        self.coronaNorm()  # Use curves to rescale the image
        self.coronagraph_touchup()  # Deal with some outliers
        self.vignette()  # Truncate the image above given radius
        self.plot_stats(False)  # Plot Extra Details
    
    # Analysis
    def make_radius_array(self):
        """Build an r-coordinate array of shape(image)"""
        self.rez = self.changed.shape[0]
        if self.center is None:
            self.center = [self.rez / 2, self.rez / 2]
            
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - self.center[0], yy - self.center[1]
        
        # self.extra_rez = 1
        self.sRadius = 400 #* self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.radius = np.sqrt(xc * xc + yc * yc) #* self.extra_rez
        
        pass
    
    def remove_offset(self):
        """Set min of array to zero"""
        self.offset = np.min(self.changed)
        self.changed -= self.offset
    

    def sort_radially(self):
        """ Flatten the image and sort by pixel radius """
        self.rad_flat = self.radius.flatten()
        self.dat_flat = self.changed.flatten()
        self.binInds = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        self.more_rez = np.max(self.binInds)
        self.radBins = [[] for x in np.arange(self.more_rez)]
        pass
    
    def bin_radially(self): # TODO Make this much faster
        """Bin the intensities by radius """
        for binI, dat in zip(self.binInds, self.dat_flat):
            try:
                self.radBins[binI].append(dat)
            except:
                pass
        # for i in range(len(self.rad_flat)):
        #     self.radBins[self.binInds[i]].append(self.dat_flat[i])
        # for i in range(len(self.rad_flat)):
        #     index = np.floor(self.rad_flat[i]).astype(np.int32)
        #     self.radBins[index].append(self.dat_flat[i])

            
    def radial_statistics(self): # TODO Make this much faster
        """ Find the statistics in each radial bin"""
        moreRez = self.radBins
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
            idx2 = np.nonzero(finite - self.offset)
            subItems = finite[idx2]
            
            # Do statistics
            if len(subItems) > 0:
                self.binMax[ii] = np.percentile(subItems, 80)  # np.nanmax(subItems)
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
        self.theMin = int(0.35 * self.rez)
        self.theMax = int(0.45 * self.rez)
        near_limb = np.arange(self.theMin, self.theMax)
        
        # Split the domain into three regions and treat seperately
        r1 = self.radAbss[np.argmax(self.binMid[near_limb]) + self.theMin]
        r2 = self.radAbss[np.argmax(self.binMax[near_limb]) + self.theMin]
        r3 = self.radAbss[np.argmax(self.binMed[near_limb]) + self.theMin]
        self.limb_radii = int(np.mean([r1, r2, r3]))
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
        
        # Plot if desired
        self.plot_curves(False)
        
        # Filter the regions separately
        mode = 'nearest'
        low_max_filt = savgol_filter(self.low_max, lWindow, rank, mode=mode)
        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank, mode=mode)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        
        high_max_filt = savgol_filter(self.high_max, hWindow, rank, mode=mode)
        
        low_min_filt = savgol_filter(self.low_min, lWindow, rank, mode=mode)
        mid_min_filt = savgol_filter(self.mid_min, mWindow, rank, mode=mode)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank, mode=mode)
        
        # Fit the lowest region with a polynomial to make it much smoother
        degree = 5
        p = np.polyfit(self.low_abs, low_max_filt, degree)
        low_max_fit = np.polyval(p, self.low_abs)  # * 1.1
        p = np.polyfit(self.low_abs, low_min_filt, degree)
        low_min_fit = np.polyval(p, self.low_abs)
        
        ind = 10
        low_max_fit[0:ind] = low_max_fit[ind]
        low_min_fit[0:ind] = low_min_fit[ind]

        
        # Build output curves - max and min as a function of radius
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
 
        doPlot = False
        if doPlot:
            # Plot the filtered curves
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
            
            plt.plot(self.fakeAbss, self.fakeMax0, label="FinalMax", lw=5)
            plt.plot(self.fakeAbss, self.fakeMin0, label="FinalMin", lw=5)
            
            # plt.plot(self.radAbss, self.binMid, label="Mid")
            # plt.plot(self.radAbss, self.binMed, label="Med")
            
            # plt.xlim([0.6*theMin,theMax*1.5])
            
            plt.legend()
            plt.show()
 
 
        
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
    def coronaNorm(self):
        """Normalize the image using the radial percentile curves"""
        
        # Collect Arrays
        self.changed = self.changed.astype('float32')
        self.changed[self.changed == 0] = np.nan
        flat_image = self.changed.flatten()
        self.dat_corona = np.ones_like(flat_image)
        
        # Allocate Arrays
        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        the_min = self.fakeMin[radius_bin]
        the_max = self.fakeMax[radius_bin]
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # Standard Normalization Formula
                top = np.subtract(flat_image, the_min)
                bottom = np.subtract(the_max, the_min)
                self.dat_corona = np.divide(top, bottom)
            except RuntimeWarning as e:
                print(e)
                pass
    
    def coronagraph_touchup(self):
        """Deal with pixel outliers. Lots of adjustable parameters in here"""
        
        # Deal with too hot things
        self.vmax = 1
        self.vmax_plot = 0.95  # np.max(dat_corona)
        hotpowr = 1 / 2
        hot = self.dat_corona > self.vmax
        # self.dat_corona[hot] = self.dat_corona[hot] ** hotpowr
        
        # Deal with too cold things
        self.vmin = 0.3
        self.vmin_plot = -0.05  # np.min(dat_corona)# 0.3# -0.03
        coldpowr = 1 / 2
        cold = self.dat_corona < self.vmin
        self.dat_corona[cold] = -((np.abs(self.dat_corona[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin
        self.dat_coronagraph = self.dat_corona
        dat_corona_square = self.dat_corona.reshape(self.changed.shape)
        
        # Some More Normalization
        dat_corona_square = np.sign(dat_corona_square) * np.power(np.abs(dat_corona_square), (1 / 5))
        self.changed = self.normalize(self.changed, high=99.99, low=0)
        dat_corona_square = self.normalize(dat_corona_square, high=99.99, low=1)
        
        # Allows you to only show sub-sections of the image as reduced images
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
    
    def vignette(self, r=1.1):
        """Truncate the image above a certain radis"""
        mask = self.radius > (int(r * self.rez // 2))  # (3.5 * self.noise_radii)
        self.changed[mask] = np.nan
    
    def plot_curves(self, do=True):
        """Plot the radial statistics from the binned array"""
        if not do: return
        
        plt.plot(self.radAbss, self.binMax, label="Max")
        plt.plot(self.radAbss, self.binMin, label="Min")
        plt.plot(self.radAbss, self.binMid, label="Mid")
        plt.plot(self.radAbss, self.binMed, label="Med")
        
        plt.axvline(self.theMin)
        plt.axvline(self.theMax)
        
        plt.axvline(self.limb_radii)
        plt.axvline(self.lCut, ls=':')
        plt.axvline(self.hCut, ls=':')
        plt.xlim([self.lCut, self.hCut])
        plt.legend()
        plt.show()
    
    def get_mask(self, dat_out):
        """ Generates a mask that defines which portion of the image will be modified"""
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
    
    def plot_stats(self, do):
        if not do: return
        fig, (ax0, ax1) = plt.subplots(2, 1, "all")
        ax0.scatter(self.n2r(self.rad_flat[::30]), self.dat_flat[::30], c='k', s=2)
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
        
        ax0.plot(self.n2r(self.fakeAbss), self.fakeMax0, label="FinalMax", lw=5)
        ax0.plot(self.n2r(self.fakeAbss), self.fakeMin0, label="FinalMin", lw=5)
        
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
    
    def plot_and_save(self):
        
        self.render()
        
        self.export_files()
    
    def render(self):
        """Generate the plots"""
        image = self.changed
        original_image = self.original
        
        full_name, save_path, time_string, ii = self.image_data
        time_string2 = self.clean_time_string(time_string)
        name, wave = self.clean_name_string(full_name)
        
        self.figbox = []
        for processed in [False, True]:
            if not self.do_orig:
                if not processed:
                    continue
            # Create the Figure
            fig, ax = plt.subplots()
            self.blankAxis(ax)
            fig.set_facecolor("k")
            
            self.inches = 10
            fig.set_size_inches((self.inches, self.inches))
            
            if 'hmi' in name.casefold():
                inst = ""
                plt.imshow(image, origin='upper', interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05
            
            else:
                
                # from .color_tables import aia_wave_dict
                # aia_wave_dict(wavelength)
                
                inst = '  AIA'
                cmap = 'sdoaia{}'.format(wave)
                cmap = aia_color_table(int(wave) * u.angstrom)
                if processed:
                    plt.imshow(image, cmap=cmap, origin='lower', interpolation=None, vmin=self.vmin_plot, vmax=self.vmax_plot)
                else:
                    toprint = self.normalize(self.absqrt(original_image))
                    # plt.imshow(toprint, cmap='sdoaia{}'.format(wavelength), origin='lower', interpolation=None) #,  vmin=self.vmin_plot, vmax=self.vmax_plot)
                    
                    plt.imshow(self.absqrt(original_image), cmap=cmap, origin='lower', interpolation=None)  # ,  vmin=self.vmin_plot, vmax=self.vmax_plot)
                
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
            self.figbox.append([fig, ax, processed])
            if self.show:
                plt.show()
    
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
                
                save_directory = os.path.dirname(os.path.dirname(save_path))
                # if "fits" in save_directory:
                #     save_directory = os.path.join(save_directory, "fits")
                # else:
                save_directory = os.path.join(save_directory, "png\\")
                
                new_path = os.path.join(save_directory, name + middle + ".png")
                
                if 'aia' in save_path:
                    os.makedirs(dirname(save_path), exist_ok=True)
                    new_path = save_path.replace("fits", "png")
                else:
                    os.makedirs(save_directory, exist_ok=True)
                plt.ioff()
                fig.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                # print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))
                self.pathBox.append(new_path)
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
        cleaned += timedelta(hours=-7)
        
        # tz = timezone(timedelta(range_hours=-1))
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
    def absqrt(image):
        return np.sqrt(np.abs(image))
    
 ##  From Modify Movie ##
class ModifyMovie():
    def execute(self):
        if self.resume:
            self.resume_last_index()
        while self.indexNow < self.waveNum:
            self.indexNow += 1
            self.main_loop(self.indexNow - 1)
            self.save_current_index()
        self.indexNow = 0
        self.save_current_index()

    # # Main Functions  ######################################################

    def main_loop(self, ii):
        """The Main Loop"""

        # Initialize Everything
        if self.init_or_skip(ii):
            return

        # Download the new fits data
        self.space_to_fits()

        # Remove the old fits files
        # self.remove_all_old_files()

        # Analyze the Dataset as a whole
        # if False:
        #     self.fits_analyze_whole_set()

        # Generate the png images
        self.fits_to_pngs()

        # Generate the Movie
        self.pngs_to_movie()

        # Generate the Audio
        # self.fits_to_audio()

        # Add Sound to the Movie
        # self.movie_to_audio_movie()

        # self.soni.thread_lock()
        print("Wavelength Complete: {}, took {:0.2f} minutes\n\n".format(self.this_name, (time() - self.beginTime) / 60))

        if self.params.stop_after_one():
            sys.exit()

    def fits_analyze_whole_set(self):
        """Check several fits files to determine fit curves"""
        print("Analyzing Dataset...", flush=True, end='')
        max_analyze = 5
        minBox = []
        maxBox = []

        self.save_path = join(self.params.local_directory, self.this_name)

        ii = 0
        for filename in listdir(self.save_path):
            if filename.endswith(".fits"):
                ii += 1
                image_path = join(self.save_path, filename)
                fname = filename[3:18]
                fname = fname.replace("_", "")
                time_string = self.parse_time_string_to_local(fname, 2)[0]

                # Load the File
                originalData, single_image_data = self.load_fits_series((self.this_name, image_path, time_string))

                self.radial_analyze(originalData, False)
                minBox.append(self.fakeMin)
                maxBox.append(self.fakeMax)

                print('.', end='')
                if ii >= max_analyze:
                    self.fakeMin = np.mean(np.asarray(minBox), axis=0)
                    self.fakeMax = np.mean(np.asarray(maxBox), axis=0)
                    self.fits_analysis_done = True
                    print("Done!")
                    break

    # def fits_to_pngs(self):
    #     """Re-save all the Fits images into pngs and normed fits files"""
    #     "Converting to Png Images..."
    #     self.apply_func_to_directory(self.do_image_work, doAll=False, desc=">Processing Images", unit="images")

    def fits_to_audio(self):
        """Analyzes the fits files to sonify them"""
        if self.params.sonify_images() and not self.sonify_complete:
            self.apply_func_to_directory(self.do_sonifying_work, doAll=True, desc=">Sonifying Images", unit="images", limit=self.params.sonify_limit())
        self.soni.generate_track(self.soni.wav_path)
            # self.soni.play()



    def movie_to_audio_movie(self):
        """Multiplexes the generated wav and avi files into a single movie"""
        if self.params.allow_muxing() and (self.new_images or self.params.sonify_images()):
            print(">Muxing Main Movie...")
            videoclip_full = VideoFileClip(self.video_name_stem.format("_raw.avi"))
            videoclip_full_muxed = videoclip_full.set_audio(AudioFileClip(self.soni.wav_path))
            from proglog import TqdmProgressBarLogger

            hq_sonFunc = partial(videoclip_full_muxed.write_videofile, self.video_name_stem.format("_HQ.mp4"), codec='libx264', bitrate='400M',
                                                 logger=TqdmProgressBarLogger(print_messages=True))
            t1 = Thread(target=hq_sonFunc)
            t1.start()
            t1.join()


            if self.params.make_compressed():
                clip = videoclip_full_muxed
                lq_sonFunc = partial(clip.write_videofile, self.video_name_stem.format("_LQ.mp4"), codec='libx264', bitrate='50M')
                wq_sonFunc = partial(clip.write_videofile, self.video_name_stem.format(".webm"), codec='libvpx')
                t2 = Thread(target=lq_sonFunc)
                t3 = Thread(target=wq_sonFunc)
                t2.start()
                t2.join()
                t3.start()
                t3.join()



            print("  Successfully Muxed")
            # # Play the Movie
            # startfile(self.video_name_stem.format("_HQ.mp4"))

    # # Support Functions  ###################################################



    def define_single_image(self, filename):
        time_code, time_string = self.time_from_filename(filename)
        image_path = join(self.save_path, filename)
        single_image_data = (self.this_name, image_path, time_string, time_code, filename)
        return single_image_data

    # def remove_all_old_files(self):
    #     files = listdir(self.save_path)
    #     file_idx = 0
    #     for filename in files:
    #         if filename.endswith(".fits") and "norm" not in filename:
    #             if self.remove_old_files(self.define_single_image(filename)):
    #                 file_idx += 1
    #                 continue
    #     if file_idx > 0:
    #         if self.params.remove_old_images():
    #             print("Deleted {} old images".format(file_idx))
    #         # else:
    #         #     print("Excluding {} old images".format(file_idx))
    #
    # def remove_old_files(self, single_image_data):
    #     filename = single_image_data[4]
    #     thisTime = int(self.time_from_filename(filename)[1])
    #     if thisTime < self.earlyLong:
    #         if self.params.remove_old_images():
    #             self.deleteFiles(filename)
    #         return 1
    #     return 0
    #
    # def this_frame_is_bad(self, image_array, single_image_data):
    #     filename = single_image_data[4]
    #     total_counts = np.nansum(image_array)
    #     if total_counts < 0:
    #         # self.deleteFiles(filename)
    #         return 1
    #     return 0
    #
    # def apply_func_to_directory(self, func, doAll=False, desc="Work Done", unit="it", limit=False):
    #     work_list = []
    #     files = listdir(self.save_path)
    #     # print("Saveing to: " + self.save_path)
    #     file_idx = -1
    #     for filename in files:
    #         if filename.endswith(".fits") and "norm" not in filename:
    #             # Define the image
    #             single_image_data = self.define_single_image(filename)
    #             pngPath = abspath(self.save_path + "\\renders\\" + filename[:-4] + 'png')
    #             # # Delete it if it is too old
    #             if self.remove_old_files(single_image_data):
    #                 # print("OLD!!!!!!!!")
    #                 continue
    #
    #             file_idx += 1
    #             if doAll or not exists(pngPath) or self.params.overwrite_pngs():
    #                 if not limit or self.soni.frame_on_any_beat(file_idx):
    #                     work_list.append([single_image_data, file_idx])
    #                 else:
    #                     work_list.append(None)
    #
    #     self.nRem = len(work_list)
    #
    #     if self.nRem > 0:
    #         with tqdm(total=self.nRem, desc=desc, unit=unit) as pbar:
    #             for image in work_list:
    #                 if image:
    #                     func(image)
    #                 pbar.update()
    #         # # pbar.close()
    #         # from pymp import Parallel
    #         # with tqdm(total=self.nRem, desc=desc, unit=unit) as pbar:
    #         #     with Parallel(self.nRem) as p:
    #         #         for i in p.range(self.nRem):
    #         #             image = work_list[i]
    #         #             if image:
    #         #                 func(image)
    #         #             pbar.update()
    #         #     # pbar.close()
    #         # from joblib import Parallel, delayed
    #
    #         # results = Parallel(n_jobs=-1, verbose=verbosity_level, backend="threading")(
    #         #     map(delayed(myfun), arg_instances))
    #
    #         # import threading as mp
    #         # from multiprocessing.pool import ThreadPool
    #         # pool = ThreadPool()
    #         # pool.map(func, work_list)
    #
    #
    # def do_image_work(self, single_image_data_ID):
    #
    #     single_image_data, file_idx = single_image_data_ID
    #     # Load the File, destroying it if it fails
    #     fail, raw_image = self.load_fits_series(single_image_data)
    #
    #     # raw_image = block_reduce(raw_image, 8)
    #
    #
    #     #Change the inputs to work with the external package
    #     full_name, save_path, time_string, time_code, filename = single_image_data
    #     # image_data = full_name, save_path, time_code, np.shape(raw_image)
    #
    #     # if fail:
    #     #     print("Failed 1")
    #     #     return 1
    #
    #     # # # Remove bad frames
    #     # if self.this_frame_is_bad(raw_image, single_image_data):
    #     #     print("Failed 2")
    #     #     return 1
    #
    #     # Modify the data
    #     # processed_image_stats = self.image_modify(raw_image)
    #     img_path = save_path[:-4]+'png'
    #     if not exists(img_path):
    #         # Modify(raw_image, image_data)
    #         Modify(save_path, resolution=self.params.resolution())
    #     # Sonify the data
    #     if False: #self.params.sonify_images():
    #         self.do_sonifying_work(single_image_data_ID, processed_image_stats, raw_image)
    #         self.sonify_complete=True if not self.params.download_images() else False
    #
    #     # Plot and save the Data
    #     # self.plot_and_save(processed_image_stats, single_image_data, raw_image)
    #     self.new_images = True
    #     return 0
    #
    # def do_sonifying_work(self, single_image_data_ID, proc_image_stats=None, raw_image=None):
    #     single_image_data, file_idx = single_image_data_ID
    #
    #     if raw_image is None:
    #         # Load the File, destroying it if it fails
    #
    #         fail1, raw_image = self.load_fits_series(single_image_data)
    #
    #         # # Remove bad frames
    #         if fail1 or self.this_frame_is_bad(raw_image, single_image_data):
    #             return 1
    #
    #     if proc_image_stats is None:
    #         single_image_data_proc = list(single_image_data)
    #         # print(single_image_data_proc)
    #         # import pdb; pdb.set_trace()
    #         single_image_data_proc[1] = single_image_data_proc[1][:-5] + "_norm.fits"
    #         fail2, proc_image_stats = self.load_fits_series(single_image_data_proc)
    #
    #     # Sonify the data
    #     self.soni.sonify_frame(proc_image_stats, raw_image, file_idx)
    #
    #     return 0
    #
    # def load_fits_series(self, image_data):
    #     # Load the Fits File from disk
    #     full_name, save_path, time_string, time_code, filename = image_data
    #     try:
    #         # Parse Inputs
    #         my_map = sunpy.map.Map(save_path)
    #     except (TypeError, OSError) as e:
    #         remove(save_path)
    #         return 1, 1
    #
    #     data = my_map.data
    #     return 0, data
    #
    # def deleteFiles(self, filename):
    #     fitsPath = join(self.save_path, filename[:-5] + '.fits')
    #     pngPath = join(self.save_path, filename[:-5] + '.png')
    #     fitsPath2 = join(self.save_path, filename[:-5] + '_norm.fits')
    #
    #     try:
    #         remove(fitsPath)
    #     except:
    #         pass
    #     try:
    #         remove(fitsPath2)
    #     except:
    #         pass
    #     try:
    #         remove(pngPath)
    #     except:
    #         pass
    #
    # def time_from_filename(self, filename, local=True):
    #     # import pdb; pdb.set_trace()
    #     # fname = filename[3:18]
    #     # time_code = fname.replace("_", "")
    #     return self.parse_time_string_to_local(filename, 1, local)
    #
    # def check_valid_png(self, img):
    #     return True # Hack
    #     image_is_new=(int(self.time_from_filename(img)[0])) < self.earlyLong
    #     return not image_is_new
    #
    # def image_modify(self, data):
    #     data = data + 0
    #     self.radial_analyze(data, False)
    #     data = self.absqrt(data)
    #     data = self.coronagraph(data)
    #     data = self.vignette(data)
    #     data = self.append_stats(data)
    #     return data
    #
    # def append_stats(self, data):
    #     from scipy.signal import savgol_filter
    #     rank = 1
    #     window1 = 31
    #     window2 = 41
    #     mode = 'mirror'
    #     btma = self.binBtm[::self.extra_rez]
    #     mina = self.binMin[::self.extra_rez]
    #     mida = self.binMid[::self.extra_rez]
    #     maxa = self.binMax[::self.extra_rez]
    #     topa = self.binTop[::self.extra_rez]
    #
    #     btma = savgol_filter(btma, window1, rank, mode=mode)
    #     mina = savgol_filter(mina, window1, rank, mode=mode)
    #     mida = savgol_filter(mida, window1, rank, mode=mode)
    #     maxa = savgol_filter(maxa, window1, rank, mode=mode)
    #     topa = savgol_filter(topa, window1, rank, mode=mode)
    #
    #     btma = savgol_filter(btma, window2, rank, mode=mode)
    #     mina = savgol_filter(mina, window2, rank, mode=mode)
    #     mida = savgol_filter(mida, window2, rank, mode=mode)
    #     maxa = savgol_filter(maxa, window2, rank, mode=mode)
    #     topa = savgol_filter(topa, window2, rank, mode=mode)
    #
    #     stacked = np.vstack(
    #         (data, btma, mina, mida, maxa, topa))
    #     return stacked
    #
    # def plot_and_save(self, data, image_data, original_data=None, ii=None):
    #     full_name, save_path, time_string, time_code, filename = image_data
    #     name = self.clean_name_string(full_name)
    #
    #     for processed in [True]:
    #
    #         # if not self.params.is_debug():
    #         #     if not processed:
    #         #         continue
    #         if not processed:
    #             if original_data is None:
    #                 continue
    #
    #         # Save the Fits File
    #         header = read_file_header(save_path)[0]
    #         if "BLANK" in header.keys():
    #             del header["BLANK"]
    #         path = save_path[:-5] + '_norm.fits'
    #         write_file(path, np.asarray(data, dtype=np.float32), header, overwrite=True)
    #
    #         data, _ = self.soni.remove_stats(data)
    #
    #         # Create the Figure
    #         fig, ax = plt.subplots()
    #         self.blankAxis(ax)
    #
    #         # inches = 10
    #         # fig.set_size_inches((inches, inches))
    #         #
    #         # pixels = data.shape[0]
    #         # dpi = pixels / inches
    #         # cocaine
    #         siX = 10
    #         siY = 10
    #         piX = 1080
    #         piY = 1080
    #         dpX = piX / siX
    #         dpY = piY / siY
    #         dpi = np.max((dpX, dpY))
    #         fig.set_size_inches((siX, siY))
    #
    #         if 'hmi' in name.casefold():
    #             inst = ""
    #             plt.imshow(data, origin='upper', interpolation=None)
    #             # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    #             plt.tight_layout(pad=5.5)
    #             height = 1.05
    #
    #         else:
    #             inst = '  AIA'
    #             plt.imshow(data if processed else self.normalize(original_data), cmap='sdoaia{}'.format(name),
    #                        origin='lower', interpolation=None,
    #                        vmin=self.vmin_plot, vmax=self.vmax_plot)
    #             plt.tight_layout(pad=0)
    #             height = 0.95
    #
    #         # Annotate with Text
    #         buffer = '' if len(name) == 3 else '  '
    #         buffer2 = '    ' if len(name) == 2 else ''
    #
    #         title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string, buffer)
    #         title2 = "{} {}, {}".format(inst, name, time_string)
    #         ax.annotate(title, (0.125, height + 0.02), xycoords='axes fraction', fontsize='large',
    #                     color='w', horizontalalignment='center')
    #         # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
    #         # the_time = strftime("%I:%M%p").lower()
    #         # if the_time[0] == '0':
    #         #     the_time = the_time[1:]
    #         # ax.annotate(the_time, (0.125, height), xycoords='axes fraction', fontsize='large',
    #         #             color='w', horizontalalignment='center')
    #
    #         # Format the Plot and Save
    #         self.blankAxis(ax)
    #         middle = '' if processed else "_orig"
    #         new_path = save_path[:-5] + middle + ".png"
    #
    #         if ii is not None and self.nRem > 0:
    #             remString = "{} / {} , {:0.1f}%".format(ii, self.nRem, 100 * ii / self.nRem)
    #         else:
    #             remString = ""
    #
    #         try:
    #             plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi, compression=2, filter=None)
    #             # print("\tSaved {} Image {}, {} of {}   ".format('Processed' if processed else "Unprocessed", time_string, remString, self.this_name), end="\r")
    #         except PermissionError:
    #             new_path = save_path[:-5] + "_b.png"
    #             plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
    #             print("Success")
    #         except Exception as e:
    #             print("Failed...using Cached")
    #             if self.params.is_debug():
    #                 raise e
    #         plt.close(fig)
    #
    #     return new_path

   
    # Level 4

    @staticmethod
    def list_files1(directory, extension):
        from os import listdir
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def get_paths(self, this_result):
        self.name = this_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = '0' + self.name
        file_name = '{}_Now.fits'.format(self.name)
        save_path = join(self.params.local_directory, file_name)
        return self.name, save_path

    @staticmethod
    def parse_time_string_to_local(downloaded_files, which=0, local=True):
        if which == 0:
            time_string = downloaded_files[0][-25:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[9:11])
            minute = time_string[11:13]
        elif which == 3:
            time_string = downloaded_files
            split = time_string.split("_")
            # import pdb; pdb.set_trace()
            year     = split[3]
            month    = split[4]
            day      = split[5].split('t')[0]
            hour_raw = split[5].split('t')[1]
            minute   = split[6]
        elif which == 2:
            time_string = downloaded_files
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = time_string[8:10]
            minute = time_string[10:12]
        elif which == 1:
            time_string = downloaded_files[3:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = time_string[9:11]
            minute = time_string[11:13]


        struct_time = (int(year), int(month), int(day), int(hour_raw), int(minute), 0, 0, 0, -1)
        # print(struct_time)

        if local:
            theTime = localtime(timegm(struct_time))
        else:
            theTime = struct_time

        new_time_string = strftime("%I:%M%p %m/%d/%Y", theTime).lower()
        if new_time_string[0] == '0':
            new_time_string = new_time_string[1:]

        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)
        time_code = strftime("%Y%m%d%I%M%S", theTime)

        return new_time_string, time_code


    @staticmethod
    def parse_time_string_to_local_old(downloaded_files, which=0):
        if which == 0:
            time_string = downloaded_files[0][-25:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[9:11])
            minute = time_string[11:13]
        else:
            time_string = downloaded_files
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = time_string[8:10]
            minute = time_string[10:12]

        struct_time = (int(year), int(month), int(day), int(hour_raw), int(minute), 0, 0, 0, -1)

        new_time_string = strftime("%I:%M%p %m/%d/%Y", localtime(timegm(struct_time))).lower()
        if new_time_string[0] == '0':
            new_time_string = new_time_string[1:]

        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)

        return new_time_string

    @staticmethod
    def clean_name_string(full_name):
        # Make the name strings
        name = full_name + ''
        while name[0] == '0':
            name = name[1:]
        return name

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

    ## Data Manipulations ##

    @staticmethod
    def reject_outliers(data):
        # # Reject Outliers
        # a = data.flatten()
        # remove_num = 20
        # ind = argpartition(a, -remove_num)[-remove_num:]
        # a[ind] = nanmean(a)*4
        # data = a.reshape(data.shape)

        data[data > 10] = np.nan

        return data

    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))

    @staticmethod
    def normalize(data):
        high = 98
        low = 15

        lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)
        return (data - lowP) / (highP - lowP)

    def vignette(self, data):

        mask = self.radius > (self.noise_radii)
        data[mask] = np.nan
        return data

    def vignette2(self, data):

        mask = np.isclose(self.radius, self.tRadius, atol=2)
        data[mask] = 1

        mask = np.isclose(self.radius, self.noise_radii, atol=2)
        data[mask] = 1
        return data

    def coronagraph(self, data):
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        dat_corona = (self.dat_flat - self.fakeMin[radius_bin]) / \
                     (self.fakeMax[radius_bin] - self.fakeMin[radius_bin])

        sys.stderr = original

        # Deal with too hot things
        self.vmax = 1
        self.vmax_plot = 1.9

        hot = dat_corona > self.vmax
        dat_corona[hot] = dat_corona[hot] ** (1 / 2)

        # Deal with too cold things
        self.vmin = 0.06
        self.vmin_plot = -0.03

        cold = dat_corona < self.vmin
        dat_corona[cold] = -((np.abs(dat_corona[cold] - self.vmin) + 1) ** (1 / 2) - 1) + self.vmin

        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)

        if self.renew_mask or self.params.mode() == 'r':
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False

        data = self.normalize(data)

        data[self.corona_mask] = dat_corona_square[self.corona_mask]

        # inds = np.argsort(self.rad_flat)
        # rad_sorted = self.rad_flat[inds]
        # dat_sort = dat_corona[inds]
        #
        # plt.figure()
        # # plt.yscale('log')
        # plt.scatter(rad_sorted[::30], dat_sort[::30], c='k')
        # plt.show()

        return data

    def get_mask(self, dat_out):

        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)

        mode = self.params.mode()

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

    # Basic Analysis

    def radial_analyze(self, data, plotStats=False):

        self.offset = np.abs(np.min(data))
        data += self.offset
        self.make_radius(data)
        self.sort_radially(data)

        stats = self.better_bin_stats(self.rad_sorted, self.dat_sorted, self.rez, self.offset)
        self.binBtm, self.binMin, self.binMax, self.binMid, self.binTop = stats

        if not self.fits_analysis_done:
            self.fit_curves()

        if plotStats:
            self.plot_stats()

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
        self.dat_sorted = self.dat_flat[inds]



    @staticmethod
    # @numba.jit(nopython=True, parallel=True)
    def better_bin_stats(rad_sorted, dat_sorted, rez, offset):
        proper_bin = np.asarray(np.floor(rad_sorted), dtype=np.int32)
        binBtm = np.empty(rez);
        binBtm.fill(np.nan)
        binMin = np.empty(rez);
        binMin.fill(np.nan)
        binMax = np.empty(rez);
        binMax.fill(np.nan)
        binMid = np.empty(rez);
        binMid.fill(np.nan)
        binTop = np.empty(rez);
        binTop.fill(np.nan)

        bin_list = [np.float64(x) for x in range(0)]
        last = 0
        for ii in np.arange(len(proper_bin)):
            binInd = proper_bin[ii]
            if binInd != last:
                bin_array = np.asarray(bin_list)
                finite = bin_array[np.isfinite(bin_array)]
                data_in_bin = finite[np.nonzero(finite - offset)]
                if len(data_in_bin) > 0:
                    out = np.percentile(data_in_bin, [0.001, 2, 50, 95, 99.999])
                    binBtm[last], binMin[last], binMid[last], binMax[last], binTop[last] = out
                bin_list = []
            bin_list.append(dat_sorted[ii])
            last = binInd
        return binBtm, binMin, binMax, binMid, binTop

        # self.radBins = [[] for x in np.arange(self.rez)]
        # for ii, binI in enumerate(self.proper_bin):
        #     self.radBins[binI].append(self.dat_sorted[ii])
        #
        # # Find the statistics by bin
        # for bin_count, bin_list in enumerate(self.radBins):
        #     self.bin_the_slice(bin_count, bin_list)

        # i = 0
        # bin_count = 0
        # not_edge = self.proper_bin[:-1] == self.proper_bin[1:]
        # theRez = len(self.proper_bin)
        # myList = []
        # while i < theRez:
        #     if i < theRez - 1 and not_edge[i]:
        #         i += 1
        #         myList.append(self.dat_sorted[i])
        #         continue
        #     self.bin_the_slice(bin_count, np.asarray(myList))
        #     bin_count += 1
        #     i += 1
        # i = 0
        # i_prev = 0
        # bin_count = 0
        # not_edge = self.proper_bin[:-1] == self.proper_bin[1:]
        # theRez = len(self.proper_bin)
        # myList = []
        # while i < theRez:
        #     if i < theRez - 1 and not_edge[i]:
        #         i += 1
        #         continue
        #     bin_arr = self.dat_sorted[i_prev:i + 1]
        #     self.bin_the_slice(bin_count, bin_arr)
        #     bin_count += 1
        #     i += 1

        # top = int(np.ceil(np.max(self.rad_sorted)))
        # print("Top : {}".format(top))
        # bin_edges = np.arange(top)  # or whatever
        # self.binMin = np.empty(bin_edges.size - 1)
        # self.binMax = np.empty(bin_edges.size - 1)
        # self.binMid = np.empty(bin_edges.size - 1)
        #
        # for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        #     data_in_bin = self.dat_sorted[(self.rad_sorted >= bin_start) * (self.rad_sorted < bin_end)]
        #     if np.any(data_in_bin):
        #         self.binMin[i], self.binMax[i], self.binMid[i] = np.percentile(data_in_bin, [2, 95, 50])
        #     else:
        #         self.binMin[i], self.binMax[i], self.binMid[i] = np.nan, np.nan, np.nan

        # Bin the intensities by radius
        # self.radial_bins = np.empty((self.rez, self.rez))
        # self.radial_bins.fill(np.nan)
        # radial_counter = np.zeros(self.rez, dtype=np.int)
        # proper_bin = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)

        # for item, binInd in enumerate(proper_bin):
        #     self.radial_bins[binInd, radial_counter[binInd]] = self.dat_sorted[item]
        #     radial_counter[binInd] += 1

        # for ii in np.arange(self.rez):
        #     useItems = np.isin(proper_bin, )
        # self.binMin , self.binMax, self.binMid = self.nan_percentile(self.radial_bins, [2, 95, 50], axis=1)

    # def bin_prep(self):
    #     self.proper_bin = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)
    #
    #     self.binBtm = np.empty(self.rez)
    #     self.binMin = np.empty(self.rez)
    #     self.binMax = np.empty(self.rez)
    #     self.binMid = np.empty(self.rez)
    #     self.binTop = np.empty(self.rez)
    #
    #     self.binBtm.fill(np.nan)
    #     self.binMin.fill(np.nan)
    #     self.binMax.fill(np.nan)
    #     self.binMid.fill(np.nan)
    #     self.binTop.fill(np.nan)

    # def bin_stats(self):
    #     self.bin_prep()
    #
    #     bin_list = []
    #     last = 0
    #     for ii, (binInd, dat) in enumerate(zip(self.proper_bin, self.dat_sorted)):
    #         if binInd != last:
    #             self.bin_the_slice(last, bin_list)
    #             bin_list = []
    #         bin_list.append(dat)
    #         last = binInd

    def bin_the_slice(self, bin_count, bin_list):
        bin_array = np.asarray(bin_list)
        finite = bin_array[np.isfinite(bin_array)]
        data_in_bin = finite[np.nonzero(finite - self.offset)]
        if len(data_in_bin) > 0:
            self.binMin[bin_count], self.binMax[bin_count], self.binMid[bin_count] = np.percentile(data_in_bin, [2, 95, 50])

    def nan_percentile2(self, arr, q, interpolation='linear'):
        # valid (non NaN) observations along the first axis
        valid_obs = np.sum(np.isfinite(arr))
        if valid_obs <= 0:
            return np.nan, np.nan, np.nan
        # replace NaN with maximum
        max_val = np.nanmax(arr)
        arr[np.isnan(arr)] = max_val
        # sort - former NaNs will move to the end
        arr = np.sort(arr)

        # loop over requested quantiles
        if type(q) is list:
            qs = []
            qs.extend(q)
        else:
            qs = [q]
        if len(qs) < 2:
            quant_arr = np.zeros(shape=(arr.shape[0]))
        else:
            quant_arr = np.zeros(shape=(len(qs), arr.shape[0]))

        result = []
        for i in range(len(qs)):
            quant = qs[i]
            # desired position as well as floor and ceiling of it
            k_arr = (valid_obs - 1) * (quant / 100.0)
            f_arr = np.floor(k_arr).astype(np.int32)
            c_arr = np.ceil(k_arr).astype(np.int32)
            fc_equal_k_mask = f_arr == c_arr

            if interpolation == 'linear':
                # linear interpolation (like numpy percentile) takes the fractional part of desired position
                floor_val = np.take(arr, f_arr) * (c_arr - k_arr)
                ceil_val = np.take(arr, c_arr) * (k_arr - f_arr)

                quant_arr = floor_val + ceil_val
                if fc_equal_k_mask:
                    quant_arr = np.take(arr, k_arr.astype(np.int32))  # if floor == ceiling take floor value


            elif interpolation == 'nearest':
                f_arr = np.around(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'lowest':
                f_arr = np.floor(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'highest':
                f_arr = np.ceiling(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            result.append(quant_arr)

        return result

    def nan_percentile(self, arr, q, interpolation='linear', axis=0):
        # valid (non NaN) observations along the first axis
        valid_obs = np.sum(np.isfinite(arr), axis=axis)
        # replace NaN with maximum
        max_val = np.nanmax(arr)
        arr[np.isnan(arr)] = max_val
        # sort - former NaNs will move to the end
        arr = np.sort(arr, axis=axis)

        # loop over requested quantiles
        if type(q) is list:
            qs = []
            qs.extend(q)
        else:
            qs = [q]
        if len(qs) < 2:
            quant_arr = np.zeros(shape=(arr.shape[0], arr.shape[1]))
        else:
            quant_arr = np.zeros(shape=(len(qs), arr.shape[0], arr.shape[1]))

        result = []
        for i in range(len(qs)):
            quant = qs[i]
            # desired position as well as floor and ceiling of it
            k_arr = (valid_obs - 1) * (quant / 100.0)
            f_arr = np.floor(k_arr).astype(np.int32)
            c_arr = np.ceil(k_arr).astype(np.int32)
            fc_equal_k_mask = f_arr == c_arr

            if interpolation == 'linear':
                # linear interpolation (like numpy percentile) takes the fractional part of desired position
                floor_val = np.take(arr, f_arr) * (c_arr - k_arr)
                ceil_val = np.take(arr, c_arr) * (k_arr - f_arr)

                quant_arr = floor_val + ceil_val
                quant_arr[fc_equal_k_mask] = np.take(arr, k_arr.astype(np.int32))[fc_equal_k_mask]

            elif interpolation == 'nearest':
                f_arr = np.around(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'lowest':
                f_arr = np.floor(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'highest':
                f_arr = np.ceiling(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)

            result.append(quant_arr)

        return result

    def fit_curves(self):
        # Input Stuff
        self.radAbss = np.arange(self.rez)
        self.highCut = 730 * self.extra_rez
        theMin = 380 * self.extra_rez
        near_limb = np.arange(theMin, theMin + 50 * self.extra_rez)

        # Find the derivative of the binned Mid
        self.diff_Mid = np.diff(self.binMid)
        self.diff_Mid += np.abs(np.nanmin(self.diff_Mid))
        self.diff_Mid /= np.nanmean(self.diff_Mid) / 100

        # Locate the Limb
        self.limb_radii = np.argmin(self.diff_Mid[near_limb]) + theMin
        self.lCut = self.limb_radii - 10 * self.extra_rez
        self.hCut = self.limb_radii + 10 * self.extra_rez

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
        from scipy.signal import savgol_filter

        lWindow = 20 * self.extra_rez + 1
        mWindow = 4 * self.extra_rez + 1
        hWindow = 30 * self.extra_rez + 1
        fWindow = int(3 * self.extra_rez) + 1

        rank = 3

        low_max_filt = savgol_filter(self.low_max, lWindow, rank)

        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)

        high_max_filt = savgol_filter(self.high_max, hWindow, rank)

        low_min_filt = savgol_filter(self.low_min, lWindow, rank)
        mid_min_filt = savgol_filter(self.mid_min, mWindow, rank)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank)

        # Fit the low curves
        lmaxf = self.fill_start(low_max_filt)
        lminf = self.fill_start(low_min_filt)
        idx = np.isfinite(lmaxf)
        p = np.polyfit(self.low_abs[idx], lmaxf[idx], 9)
        low_max_fit = np.polyval(p, self.low_abs)
        p = np.polyfit(self.low_abs[idx], lminf[idx], 9)
        low_min_fit = np.polyval(p, self.low_abs)

        # Build output curves
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin = np.hstack((low_min_fit, mid_min_filt, high_min_filt))

        # Filter again to smooth boundaraies
        self.fakeMax = self.fill_end(self.fill_start(savgol_filter(self.fakeMax, fWindow, rank)))
        self.fakeMin = self.fill_end(self.fill_start(savgol_filter(self.fakeMin, fWindow, rank)))

        # Locate the Noise Floor
        noiseMin = 550 * self.extra_rez - self.hCut
        near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        self.diff_max_abs = self.high_abs[near_noise]
        self.diff_max = np.diff(high_max_filt)[near_noise]
        self.diff_max += np.abs(np.nanmin(self.diff_max))
        self.diff_max /= np.nanmean(self.diff_max) / 100
        self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        self.noise_radii = 565 * self.extra_rez

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
            try:
                val = use[iii]
            except:
                return use
        use[:iii] = val
        return use

    def plot_stats(self):

        fig, (ax0, ax1) = plt.subplots(2, 1, True)
        ax0.scatter(self.n2r(self.rad_sorted[::30]), self.dat_sorted[::30], c='k', s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=':')
        ax0.axvline(self.n2r(self.hCut), ls=':')
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))

        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_Mid, 'r:')

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

        try:
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        except:
            ax0.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.radAbss), self.fakeMin, 'g')

        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMid, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        ax0.set_xlim((0, self.n2r(self.highCut)))
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
        if self.params.is_debug():
            file_name = '{}_Radial.png'.format(self.name)
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)

            file_name = '{}_Radial_zoom.png'.format(self.name)
            ax0.set_xlim((0.9, 1.1))
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)
            # plt.show()
            plt.close(fig)
        else:
            plt.show()

    def n2r(self, n):
        if True:
            return n / self.limb_radii
        else:
            return n

# Test Functions



def print_banner():
    """Prints a message at code start"""
    print("\nSunback Web: SDO Website and Background Updater \nWritten by Chris R. Gilly")
    print("Check out my website: http://gilly.space\n")


def test_all(test_path="data/0171_MR.fits", show=True):
    print_banner()
    print("\nTesting Module...")
    print("    No input method...", end='')
    test_mod = Modify(show=show)
    print("Success", flush=True)
    print("    Input String Method...", end='')
    test_mod2 = Modify(test_path, show=show)
    print("Success", flush=True)
    print("    Input Array Method...", end='')
    image, image_data = load_fits_field(test_path)
    test_mod3 = Modify(image, image_data, show=show)
    print("Success", flush=True)
    print("\nAll Tests Run Successfully\n")


if __name__ == "__main__":
    test_all(show=False)
    
    """        # dat2 = self.renormalize(dat)
            # half = int(dat.shape[0]/2)
            # dat[:, :half] = dat2[:, :half]
            # dat[:, half:] = dat2[:, half:]
            # return dat"""
    
    # print(image.dtype)
    #
    # inds = np.argsort(self.rad_flat)
    # rad_sorted = self.rad_flat[inds]
    # dat_sort = dat_corona[inds]
    #
    # plt.figure()
    # # plt.yscale('log')
    # plt.scatter(rad_sorted[::30], dat_sort[::30], c='k')
    # plt.show()
    
    # image = image / np.mean(image)
    
    # image = image**(1/2)
    # image = np.log(image)
    
    # image = self.normalize(image, high=85, low=5)

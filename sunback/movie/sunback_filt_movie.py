import numpy as np
import os
import matplotlib.pyplot as plt
from time import strftime
from datetime import timedelta
from scipy.signal import savgol_filter
import astropy.units as u
from astropy.io import fits
import datetime
from copy import copy

from sunback.science.color_tables import aia_color_table
import warnings

warnings.filterwarnings("ignore")
from astropy.nddata import block_reduce


class Modify:
    renew_mask = True
    image_data = None

    def __init__(
        self,
        image="../../data/0171_MR.fits",
        image_data=None,
        orig=False,
        show=False,
        verb=False,
        resolution=False,
    ):
        """Initialize the main class"""

        # Parse Inputs
        self.show = show
        self.image_data = image_data
        self.verb = verb
        self.do_orig = orig
        self.resolution = resolution
        self.reduceby = 1

        try:
            self.parse_input_type(image)

            if self.original is not None:
                # Run the Reduction Algorithm
                self.image_modify()  # Primary Algorithm
                self.plot_and_save()
        except KeyError as e:
            print(e)
            pass

        if self.verb:
            print("Done")

    def parse_input_type(self, image):
        """Determine what kind of input image was provided and open it appropriately"""
        # Load the File
        if image is None:
            # Run the Test Case
            self.original = self.test()
        elif type(image) in [str]:
            # Load the file at input path
            path = image
            self.original = self.load_file(path)
        elif type(image) in [np.array, np.ndarray]:
            self.original = image

        else:
            raise TypeError("Invalid Input Data: {}".format(type(image)))

        if self.original is None:
            return None

        if self.resolution:
            self.reduceby = int(self.original.shape[0] / self.resolution)
            if self.reduceby > 1:
                self.original = block_reduce(image, self.reduceby)

        # Copy the input array
        self.changed = copy(self.original)

        if self.image_data is None:
            # Use default Metadata
            self.image_data = self.def_data(self.changed)
        self.name = self.image_data[0]
        self.file_path = self.image_data[1]

    def test(self):
        """Run the test case if no input is provided"""
        if self.verb:
            print("Running Test Case")
        image = self.load_file("../../data/0171_MR.fits")
        self.show = True
        return image

    def load_file(self, path):
        """Load a fits file from disk"""

        try:
            with fits.open(path, cache=False) as hdul:
                hdul.verify("silentfix+ignore")
                # Get data out of the header, and grab the data array
                try:
                    wave, t_rec = hdul[0].header["WAVELNTH"], hdul[0].header["T_OBS"]
                    image = hdul[0].data
                except Exception as e:
                    wave, t_rec = hdul[1].header["WAVELNTH"], hdul[1].header["T_OBS"]
                    image = hdul[1].data
                if self.reduceby:
                    image = block_reduce(image, self.reduceby)
                    reduce = self.reduceby
                else:
                    reduce = 1

                # Make sure the frame is good
                total_counts = np.nansum(image)
                if total_counts < 0:
                    return None

                try:
                    self.center = (
                        hdul[0].header["X0_MP"] / reduce,
                        hdul[0].header["Y0_MP"] / reduce,
                    )
                except:
                    self.center = (
                        hdul[1].header["X0_MP"] / reduce,
                        hdul[1].header["Y0_MP"] / reduce,
                    )

                while self.center[0] > 0.9 * self.resolution:
                    self.center = np.asarray(self.center) / 2

                self.image_data = str(wave), path, t_rec, image.shape

        except (TypeError, OSError) as e:
            # from os import remove
            # remove(path)
            return None

        return image

    def def_data(self, hdul):
        """Use Defaults Values for Data"""
        try:
            shape = hdul[0].data.shape
        except:
            shape = hdul.shape

        wave = 171
        full_name = str(wave)
        save_path = str(wave)
        time_string = "2021-02-02T19:00:12.57Z"

        self.image_data = full_name, save_path, time_string, shape
        return self.image_data

    def get(self):
        """Returns the reduced image array"""
        return self.changed

    def image_modify(self):
        """Perform the image normalization on the input array"""
        self.make_radius_array()  # Assign Each Pixel its Radius Value
        self.remove_offset()  # Additive Shift of input array
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
        # centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - self.center[0], yy - self.center[1]

        self.extra_rez = 1  # An attempt to give extra resolution
        self.sRadius = 400 * self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
        self.rez *= self.extra_rez

    def remove_offset(self):
        """Set min of array to zero"""
        self.offset = np.min(self.changed)
        self.changed -= self.offset

    def sort_radially(self):
        """Flatten the image and sort by pixel radius"""
        self.rad_flat = self.radius.flatten()
        self.dat_flat = self.changed.flatten()
        inds = np.argsort(self.rad_flat)
        self.rad_sorted = self.rad_flat[inds]
        self.dat_sort = self.dat_flat[inds]

    def bin_radially(self):
        """Bin the intensities by radius"""
        self.radBins = [[] for x in np.arange(self.rez)]
        binInds = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)
        try:
            for ii, binI in enumerate(binInds):
                self.radBins[binI].append(self.dat_sort[ii])
        except IndexError as e:
            a = 1

    def radial_statistics(self):
        """Find the statistics in each radial bin"""
        self.binMax = np.zeros(self.rez)
        self.binMin = np.zeros(self.rez)
        self.binMid = np.zeros(self.rez)
        self.binMed = np.zeros(self.rez)
        self.radAbss = np.arange(self.rez)

        for ii, it in enumerate(self.radBins):
            # For each radial bin
            item = np.asarray(it)
            idx = np.isfinite(item)
            finite = item[idx]
            idx2 = np.nonzero(finite - self.offset)
            subItems = finite[idx2]

            # Do statistics
            if len(subItems) > 0:
                self.binMax[ii] = np.percentile(subItems, 85)  # np.nanmax(subItems)
                self.binMin[ii] = np.percentile(subItems, 1)  # np.min(subItems)
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
        hWindow = 7  # 30 * self.extra_rez + 1
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
        self.low_abs = self.radAbss[: self.lCut]
        self.low_max = self.binMax[: self.lCut]
        self.low_min = self.binMin[: self.lCut]

        self.mid_abs = self.radAbss[self.lCut : self.hCut]
        self.mid_max = self.binMax[self.lCut : self.hCut]
        self.mid_min = self.binMin[self.lCut : self.hCut]

        self.high_abs = self.radAbss[self.hCut :]
        self.high_max = self.binMax[self.hCut :]
        self.high_min = self.binMin[self.hCut :]

        # Plot if desired
        self.plot_curves(False)

        # Filter the regions separately
        mode = "nearest"
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

            plt.plot(self.low_abs, low_min_fit, c="k")
            plt.plot(self.low_abs, low_max_fit, c="k")

            # plt.plot(self.radAbss, self.binMid, label="Mid")
            # plt.plot(self.radAbss, self.binMed, label="Med")

            # plt.xlim([0.6*theMin,theMax*1.5])

            plt.legend()
            plt.show()

        # Build output curves - max and min as a function of radius
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax0 = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin0 = np.hstack((low_min_fit, mid_min_filt, high_min_filt))

        # Filter again to smooth boundaraies
        self.fakeMax0 = self.fill_end(
            self.fill_start(savgol_filter(self.fakeMax0, fWindow, rank))
        )
        self.fakeMin0 = self.fill_end(
            self.fill_start(savgol_filter(self.fakeMin0, fWindow, rank))
        )

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

    # Reduction
    def coronaNorm(self):
        """Normalize the image using the radial percentile curves"""

        # Collect Arrays
        self.changed = self.changed.astype("float32")
        self.changed[self.changed == 0] = np.nan
        flat_image = self.changed.flatten()
        self.dat_corona = np.ones_like(flat_image)

        # Allocate Arrays
        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        the_min = self.fakeMin[radius_bin - 1]
        the_max = self.fakeMax[radius_bin - 1]

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
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
        self.vmax = 1.0
        hotpowr = 1 / 2
        hot = self.dat_corona > self.vmax
        self.dat_corona[hot] = self.dat_corona[hot] ** hotpowr

        # Deal with too cold things
        self.vmin = 0.5
        coldpowr = 1 / 2
        cold = self.dat_corona < self.vmin
        self.dat_corona[cold] = (
            -((np.abs(self.dat_corona[cold] - self.vmin) + 1) ** coldpowr - 1)
            + self.vmin
        )
        self.dat_coronagraph = self.dat_corona
        dat_corona_square = self.dat_corona.reshape(self.changed.shape)

        # Some More Normalization
        dat_corona_square = np.sign(dat_corona_square) * np.power(
            np.abs(dat_corona_square), (1 / 5)
        )
        self.changed = self.normalize(self.changed, high=100, low=0)

        self.vmax_plot = 1.3  # np.max(dat_corona)
        self.vmin_plot = 0.75  # np.min(dat_corona)# 0.3# -0.03
        # dat_corona_square = self.normalize(dat_corona_square, high=99, low=1, mid=None)

        # Allows you to only show sub-sections of the image as reduced images
        if self.renew_mask:
            self.corona_mask = self.get_mask(self.changed)
            self.renew_mask = False

        # Allows you to mirror horizontally, with only one half reduced
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
        self.changed = self.changed.astype("float32")

    def vignette(self, r=1.1):
        """Truncate the image above a certain radis"""
        mask = self.radius > (int(r * self.rez // 2))  # (3.5 * self.noise_radii)
        self.changed[mask] = np.nan

    def plot_curves(self, do=True):
        """Plot the radial statistics from the binned array"""
        if not do:
            return

        plt.plot(self.radAbss, self.binMax, label="Max")
        plt.plot(self.radAbss, self.binMin, label="Min")
        plt.plot(self.radAbss, self.binMid, label="Mid")
        plt.plot(self.radAbss, self.binMed, label="Med")

        plt.axvline(self.theMin)
        plt.axvline(self.theMax)

        plt.axvline(self.limb_radii)
        plt.axvline(self.lCut, ls=":")
        plt.axvline(self.hCut, ls=":")
        plt.xlim([self.lCut, self.hCut])
        plt.legend()
        plt.show()

    def get_mask(self, dat_out):
        """Generates a mask that defines which portion of the image will be modified"""
        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)

        mode = "y"

        if type(mode) in [float, int]:
            mask_num = mode
        elif "y" in mode:
            mask_num = 1
        elif "n" in mode:
            mask_num = 2
        else:
            if "r" in mode:
                if len(mode) < 2:
                    mode += "a"

            if "a" in mode:
                top = 8
                btm = 1
            elif "h" in mode:
                top = 6
                btm = 3
            elif "d" in mode:
                top = 8
                btm = 7
            elif "w" in mode:
                top = 2
                btm = 1
            else:
                print("Unrecognized Mode")
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
        if not do:
            return
        fig, (ax0, ax1) = plt.subplots(2, 1, True)
        ax0.scatter(self.n2r(self.rad_sorted[::30]), self.dat_sort[::30], c="k", s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls="--", label="Limb")
        # ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=":")
        ax0.axvline(self.n2r(self.hCut), ls=":")
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))

        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_mean, 'r:')

        ax0.plot(self.n2r(self.low_abs), self.low_max, "m", label="Percentile")
        ax0.plot(self.n2r(self.low_abs), self.low_min, "m")
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')

        ax0.plot(self.n2r(self.high_abs), self.high_max, "c", label="Percentile")
        ax0.plot(self.n2r(self.high_abs), self.high_min, "c")

        ax0.plot(self.n2r(self.mid_abs), self.mid_max, "y", label="Percentile")
        ax0.plot(self.n2r(self.mid_abs), self.mid_min, "y")
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
        ax0.set_yscale("log")

        ax1.scatter(
            self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c="k", s=2
        )
        ax1.set_ylim((-0.25, 2))

        ax1.axhline(self.vmax, c="r", label="Confinement")
        ax1.axhline(self.vmin, c="r")
        ax1.axhline(self.vmax_plot, c="orange", label="Plot Range")
        ax1.axhline(self.vmin_plot, c="orange")

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
            file_name = "{}_Radial.png".format(self.name)
            # print("Saving {}".format(file_name))
            # save_path = join(r"data\images\radial", file_name)
            # plt.savefig(save_path)

            file_name = "{}_Radial_zoom.png".format(self.name)
            ax0.set_xlim((0.9, 1.1))
            # save_path = join(r"data\images\radial", file_name)
            # plt.savefig(save_path)
            # plt.show()
            plt.close(fig)
        else:
            plt.show(block=True)
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
    def normalize(image, high=98, low=15, mid=None):
        if mid is not None:
            midP = np.nanpercentile(image, mid)
            image /= midP

        if low is None:
            lowP = 0
        else:
            lowP = np.nanpercentile(image, low)
        highP = np.nanpercentile(image, high)

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
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

            if "hmi" in name.casefold():
                inst = ""
                plt.imshow(image, origin="upper", interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05

            else:
                # from color_tables import aia_wave_dict
                # aia_wave_dict(wave)

                inst = "  AIA"
                cmap = "sdoaia{}".format(wave)
                cmap = aia_color_table(int(wave) * u.angstrom)
                if processed:
                    plt.imshow(
                        image,
                        cmap=cmap,
                        origin="lower",
                        interpolation=None,
                        vmin=self.vmin_plot,
                        vmax=self.vmax_plot,
                    )
                else:
                    toprint = self.normalize(self.absqrt(original_image))
                    # plt.imshow(toprint, cmap='sdoaia{}'.format(wave), origin='lower', interpolation=None) #,  vmin=self.vmin_plot, vmax=self.vmax_plot)

                    plt.imshow(
                        self.absqrt(original_image),
                        cmap=cmap,
                        origin="lower",
                        interpolation=None,
                    )  # ,  vmin=self.vmin_plot, vmax=self.vmax_plot)

                plt.tight_layout(pad=0)
                height = 0.95

            # Annotate with Text
            buffer = "" if len(name) == 3 else "  "
            buffer2 = "    " if len(name) == 2 else ""

            title = "{}    {} {}, {}{}".format(
                buffer2, inst, wave, time_string2, buffer
            )
            ax.annotate(
                title,
                (0.15, height + 0.02),
                xycoords="axes fraction",
                fontsize="large",
                color="w",
                horizontalalignment="center",
            )
            # title2 = "{} {}, {}".format(inst, name, time_string2)
            # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
            the_time = strftime("%Z %I:%M%p")
            if the_time[0] == "0":
                the_time = the_time[1:]
            ax.annotate(
                the_time,
                (0.15, height),
                xycoords="axes fraction",
                fontsize="large",
                color="w",
                horizontalalignment="center",
            )

            # Format the Plot and Save
            self.blankAxis(ax)
            self.figbox.append([fig, ax, processed])
            if self.show:
                print("hooooo")
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
                ax.axis("off")
                fig.tight_layout(pad=0)
                # To remove the huge white borders
                ax.margins(0)
                ax.set_facecolor("k")

                fig.canvas.draw()

                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8
                )
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,)
                )

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
                middle = "" if processed else "_orig"

                # name, wave = self.clean_name_string(full_name)
                abPath, fileName = save_path.rsplit("\\", 1)
                new_path = fileName[:-5] + middle + ".png"
                directory = abPath + "\\renders\\"
                path = directory + new_path

                os.makedirs(directory, exist_ok=True)
                fig.savefig(path, facecolor="black", edgecolor="black", dpi=dpi)
                # print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))
                self.pathBox.append(path)  # changed this from path
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
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    @staticmethod
    def clean_name_string(full_name):
        digits = "".join(i for i in full_name if i.isdigit())
        # Make the name strings
        name = digits + ""
        digits = "{:04d}".format(int(name))
        # while name[0] == '0':
        #     name = name[1:]
        return digits, name

    @staticmethod
    def clean_time_string(time_string):
        # Make the name strings

        try:
            cleaned = datetime.datetime.strptime(time_string[:-4], "%Y-%m-%dT%H:%M:%S")
        except ValueError as e:
            cleaned = datetime.datetime.strptime(time_string, "%H:%M%p %m/%d/%Y")
        cleaned += timedelta(hours=-7)

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
    def absqrt(image):
        return np.sqrt(np.abs(image))


# Test Functions
def load_file(path):
    """Load a fits file from disk"""
    with fits.open(path, cache=False) as hdul:
        hdul.verify("silentfix+warn")
        wave, t_rec = hdul[0].header["WAVELNTH"], hdul[0].header["T_OBS"]
        image = hdul[0].data
        image_data = str(wave), str(wave), t_rec, image.shape
    return image, image_data


def print_banner():
    """Prints a message at code start"""
    print(
        "\nSunback Web: SDO Website and Background Updater \nWritten by Chris R. Gilly"
    )
    print("Check out my website: http://gilly.space\n")


def test_all(
    test_path="renders/background_server_lingon/rainbow/imgs/fits/AIAsynoptic0171.fits",
    show=True,
):
    print_banner()
    print("\nTesting Module...")
    print("    No input method...", end="")
    test_mod = Modify(show=show)
    print("Success", flush=True)
    print("    Input String Method...", end="")
    test_mod2 = Modify(test_path, show=show)
    print("Success", flush=True)
    print("    Input Array Method...", end="")
    image, image_data = load_file(test_path)
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

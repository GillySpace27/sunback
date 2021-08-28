from time import time
from processor.Processor import Processor
from science.SRNFilter import SRNFilter

set_local_background = False

# Initialization
last_time = time()
start_time = last_time

default_sleep = 30


class RadialFiltProcessor(Processor):
    out_name = 'SRN'
    filt_name = '  Radial Filter'
    do_function =  SRNFilter
    do_png = False
    description = "Filter the Images Radially with SRN"


class SRNFilter:
    """This is the primary code used in the RadialFiltProcessor"""
    renew_mask = True
    image_data = None
    name = "Default"
    def __init__(self, fits_path, in_name=-1, orig=False, show=False, verb=False):
        """Initialize the main class"""
        
        # Parse Inputs
        self.fits_path = fits_path
        self.in_name = in_name
        self.show = show
        self.verb = verb
        self.do_orig = orig
        self.center = None
        self.original = None
        self.changed = None
        
        # Load Image
        self.load_fits_image(fits_path, -1)
        
        # Run the Reduction Algorithm
        self.image_modify()  # Primary Algorithm

    def load_fits_image(self, fits_path=None, in_field=None):
        """Select how to load the image"""
        if fits_path is None:
            # Run the Test Case
            self.original = self.test()
        elif type(fits_path) in [str]:
            # Load the file at input path
            self.ingest_fits_file(fits_path, in_field)
        else:
            raise TypeError("Invalid Input Data: {}".format(type(fits_path)))

    def ingest_fits_file(self, fits_path, get_field=None):
        """open the fits file and grab the necessary data"""
        if get_field is None:
            get_field = self.in_name
            
        frame, wave, t_rec, center = load_fits_field(fits_path, field=get_field)
        self.original = copy(frame)
        self.changed = copy(frame)
        self.image_data = str(wave), fits_path, t_rec, frame.shape
        self.set_centerpoint(center)
        
    def set_centerpoint(self, center):
        """Parse the centerpoint and ensure correct scaling"""
        self.center = center
        image_edge = self.original.shape
        center_given = np.abs(self.center)
        
        Top_Tolerance = 0.65
        Bottom_Tolerance = 0.35
        count=0
        while count < 10:
            ratio = center_given/image_edge
            if np.array(ratio > Top_Tolerance).any():
                center_given *= 0.5
            elif np.array(ratio < Bottom_Tolerance).any():
                center_given *= 2
            else:
                break
        self.center = center_given
    
    def get(self):
        """Returns the reduced in_object array"""
        return self.changed

    # Analysis
    def image_modify(self):
        """Perform the in_object normalization on the input array"""
        self.make_radius_array()  # Assign Each Pixel its Radius Value
        self.remove_offset()  # Additive Shift of input array
        self.sort_radially()  # Build Flattened and Sorted Intensity Arrays
        self.bin_radially()  # Create a cloud of intesity values for each radial bin
        self.radial_statistics()  # Find mean and percentiles vs height
        self.make_curves()  # Build smooth curves based on the statistics
        self.coronaNorm()  # Use curves to rescale the in_object
        self.coronagraph_touchup()  # Deal with some outliers
        self.vignette()  # Truncate the in_object above given radius
        self.plot_stats(False)  # Plot Extra Details
    
    def make_radius_array(self):
        """Build an r-coordinate array of shape(in_object)"""
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
        """ Flatten the in_object and sort by pixel radius """
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
        """Normalize the in_object using the radial percentile curves"""
        
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
    
    def vignette(self, r=1.1):
        """Truncate the in_object above a certain radis"""
        mask = self.radius > (int(r * self.rez // 2))  # (3.5 * self.noise_radii)
        self.changed[mask] = np.nan
    
    # Helpers
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

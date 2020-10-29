
import numpy as np
import sunpy.map
import os
import matplotlib.pyplot as plt
from time import strftime
from datetime import timezone
from astropy.time import Time
from scipy.signal import savgol_filter


class Modify:

    def __init__(self, data, image_data):
        """Initialize a new parameter object or use the provided one"""
        self.image_data = image_data
        self.name = "AWS Trial"
        self.renew_mask = True
        self.original = data
        self.changed = self.image_modify(data+0)
        self.plot_and_save()
        pass

    def get(self):
        return self.changed

    def get_path(self):
        return self.newPath

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

        #Remove NANs
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
        theMin = int(0.35*self.rez)
        theMax = int(0.45*self.rez)
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


        lWindow = 7 # 4 * self.extra_rez + 1
        mWindow = 7 # 4 * self.extra_rez + 1
        hWindow = 7 # 30 * self.extra_rez + 1
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
        # print(self.count_nan(low_max_filt))
        p = np.polyfit(self.low_abs, low_max_filt, degree)
        low_max_fit = np.polyval(p, self.low_abs) #* 1.1
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

        mask = self.radius > (int(1.1* self.rez // 2)) #(3.5 * self.noise_radii)
        data[mask] = np.nan
        return data

    def coronagraph(self, data):
        # original = sys.stderr
        # sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')
        # print(data.dtype)
        data[data==0] = np.nan
        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        # 
        dat_corona = (data.flatten() - self.fakeMin[radius_bin]) / \
                     (self.fakeMax[radius_bin] - self.fakeMin[radius_bin])


        # sys.stderr = original

        # Deal with too hot things
        self.vmax = 0.95
        self.vmax_plot = 0.85 #np.max(dat_corona)
        hotpowr = 1/1.5


        hot = dat_corona > self.vmax
        # dat_corona[hot] = dat_corona[hot] ** hotpowr

        # Deal with too cold things
        self.vmin = 0.3
        self.vmin_plot = -0.05 #np.min(dat_corona)# 0.3# -0.03
        coldpowr = 1/2

        cold = dat_corona < self.vmin
        dat_corona[cold] = -((np.abs(dat_corona[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin

        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)


        if self.renew_mask:
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False
        dat_corona_square = np.sign(dat_corona_square) * np.power(np.abs(dat_corona_square), (1/5))
        data = self.normalize(data, high = 100, low=0)
        dat_corona_square = self.normalize(dat_corona_square, high = 100, low=1)


        # 
        do_mirror = False
        if do_mirror:
            #Do stuff
            xx, yy = self.corona_mask.shape[0], int(self.corona_mask.shape[1]/2)
            # 
            newDat = data[self.corona_mask]
            grid = newDat.reshape(xx,yy)
            # if self.
            flipped = np.fliplr(grid)
            data[~self.corona_mask] = flipped.flatten() # np.flip(newDat)

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

        fig, (ax0, ax1) = plt.subplots(2, 1, "True")
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
        if doPlot: #self.params.is_debug():
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

        return (data - lowP) / (highP - lowP)

    def plot_and_save(self):

        self.render()

        self.export()

    def render(self):
        """Generate the plots"""
        data = self.changed
        original_data = self.original

        full_name, save_path, time_string, ii = self.image_data
        time_string2 = self.clean_time_string(time_string)
        name = self.clean_name_string(full_name)

        self.figbox = []
        for processed in [False, True]:

            if not True:
                if not processed:
                    continue
            if not processed:
                if original_data is None:
                    continue

            # Create the Figure
            fig, ax = plt.subplots()
            self.blankAxis(ax)

            self.inches = 10
            fig.set_size_inches((self.inches, self.inches))



            if 'hmi' in name.casefold():
                inst = ""
                plt.imshow(data, origin='upper', interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05

            else:
                inst = '  AIA'
                if processed:
                    plt.imshow(data , cmap='sdoaia{}'.format(name), origin='lower', interpolation=None,  vmin=self.vmin_plot, vmax=self.vmax_plot)
                else:
                    toprint = self.normalize(self.absqrt(original_data))
                    plt.imshow(toprint , cmap='sdoaia{}'.format(name), origin='lower', interpolation=None) #,  vmin=self.vmin_plot, vmax=self.vmax_plot)


                plt.tight_layout(pad=0)
                height = 0.95

            # Annotate with Text
            buffer = '' if len(name) == 3 else '  '
            buffer2 = '    ' if len(name) == 2 else ''

            title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string2, buffer)
            title2 = "{} {}, {}".format(inst, name, time_string2)
            ax.annotate(title, (0.15, height + 0.02), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')
            # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
            the_time = strftime("%I:%M%p").lower()
            if the_time[0] == '0':
                the_time = the_time[1:]
            ax.annotate(the_time, (0.15, height), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')

            # Format the Plot and Save
            self.blankAxis(ax)
            self.figbox.append([fig, processed])

    def export(self):
        full_name, save_path, time_string, ii = self.image_data
        pixels = self.changed.shape[0]
        dpi = pixels / self.inches
        try:
            for fig, processed in self.figbox:
                middle = '' if processed else "_orig"

                new_path = save_path[:-5] + middle + ".png"
                name = self.clean_name_string(full_name)
                directory = "renders/"
                path = directory + new_path.rsplit('/')[1]
                os.makedirs(directory, exist_ok=True)
                plt.close(fig)
                self.newPath = path
                fig.savefig(path, facecolor='black', edgecolor='black', dpi=dpi)
                print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))

        except Exception as e:
            raise e
        finally:
            for fig, processed in self.figbox:
                plt.close(fig)




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
        while name[0] == '0':
            name = name[1:]
        return name

    @staticmethod
    def clean_time_string(time_string):
        # Make the name strings

        # 
        # cleaned = time_string.replace(tzinfo=timezone.utc).astimezone(tz=None)
        cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime("%I:%M%p, %b-%d, %Y")

        return cleaned
        # name = full_name + ''
        # while name[0] == '0':
        #     name = name[1:]
        # return name

    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))

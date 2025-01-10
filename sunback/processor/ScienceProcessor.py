# import os
# from os import makedirs
# from os.path import join, dirname
import numpy as np

# from scipy.signal import savgol_filter
# from src.processor.Processor import Processor
#
# import warnings
#
# warnings.filterwarnings("ignore")
import matplotlib as mpl
#
# mpl.use("qt5agg")

import matplotlib.pyplot as plt

#
# plt.ioff()
#
# do_dprint = False
#
#
# def dprint(txt, **kwargs):
#     if do_dprint:
#         print(txt, **kwargs)
#
#
import os
from os.path import join

from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from sunback.processor.Processor import Processor


class ScienceProcessor(Processor):
    filt_name = "Scientist"
    description = "Examine the files"
    progress_verb = "Examining"
    progress_unit = "Fits Files"

    def __init__(
        self,
        fits_path=None,
        in_name=-1,
        orig=False,
        show=False,
        verb=False,
        quick=False,
        rp=None,
        params=None,
    ):
        """Initialize the main class"""
        self.save_to_fits = False
        self.current_frame_name = None
        self.flat_im = None
        self.do_png = False
        super().__init__(params, quick, rp)
        self.params.do_single = False

    #         # Parse Inputs
    #
    #     ###################
    #     ## Structure ##
    #     ###################
    #
    def setup(self):
        """Do prep work once before the main algorithm"""
        print("Setup ran!")

        self.ii = 0
        self.n_hist = 50
        self.locs = []
        self.vals = []
        self.annulus_width = 5
        self.rr = 1.25

        # n_heights = 20
        # viridis = mpl.colormaps['viridis']#.resampled(n_lines)
        # LinearSegmentedColormap
        # annulus = 0

    def cleanup(self):
        print("Cleanup time!")

        fig, axes = plt.subplots(1, sharex="all")
        # fig.suptitle("Annulus Width: {}".format(self.annulus_width))

        array = np.asarray(self.vals).T
        xx, yy = np.meshgrid(self.locs, self.bins)
        hhist = axes.pcolormesh(xx, yy, array, cmap="YlOrRd", label="Sim Hist")
        axes.set_ylabel("QRN Normalized Intensity")
        axes.set_xlabel("Days of January 2019")
        axes.set_title(r"Height = {} $R_\odot$".format(self.rr))
        plt.show(block=True)
        a = 1
        super().cleanup()

    def do_work(self):
        # print("I did work!")
        # rr = 1.2
        #
        # for idx, rr in enumerate(np.linspace(1.05, 1.65, n_heights)):
        # Pull out a line at a given height, with a given band width
        # rr = np.round(self.rr, 3)
        # clr =viridis(idx/n_heights)
        if self.ii == 0:
            self.init_radius_array()

        good_coord, bin_array, radii, the_mean, the_std, want_bin = self.get_annulus(
            self.rr, "qrn", width=self.annulus_width, load=False
        )
        n, self.bins = np.histogram(bin_array, range=(0, 1), bins=self.n_hist)
        nn = n.tolist()
        nn.append(0)
        nnn = np.asarray(nn)
        normed_nnn = nnn / np.nansum(nnn)
        # axes[0].plot(bins, normed_nnn, color=clr, label=str(rr))
        self.locs.append(self.ii)
        self.vals.append(normed_nnn)
        self.ii += 1
        # axes[1].scatter(1.0, rr, color=clr, zorder=1050+idx)
        #
        #
        # array = np.asarray(vals)
        # xx, yy = np.meshgrid(bins, locs)
        # hhist = axes[1].pcolormesh(xx, yy, array, cmap='YlOrRd', label="Sim Hist")
        #
        # # xx, yy = np.meshgrid([0.98, 1.0], locs)
        #
        # # hhist = axes[1].pcolormesh(xx, yy, locs[:, None], cmap='viridis', label="Sim Hist")
        #
        # # cbar = fig.colorbar(hhist, ax=axes[1])
        #
        # # fig.subplots_adjust(right=0.8)
        # # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # # cbar=fig.colorbar(hhist, cax=cbar_ax, aspect=40)
        # # cbar.set_label('Cccurance')
        #
        #
        #
        # # plt.legend()
        # # axes[0].set_xlabel("Intensity Value")
        # axes[0].set_ylabel("Occurance")
        #
        # axes[1].set_xlabel("Intensity Value")
        # axes[1].set_ylabel("Distance from Sun Center")
        # # plt.savefig
        # # plt.show(block=True)
        #
        # # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms_radial_hq.png", dpi=600)
        # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms_radial_lq.png", dpi=400)
        # # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms_radial_vlq.png", dpi=300)
        # plt.close(fig)
        #
        # a=1
        #
        pass

    #
    # def do_work_2(self):
    #     fig, axes = plt.subplots(2, sharex='all')
    #
    #     n_heights = 20
    #     n_hist = 50
    #     viridis = mpl.colormaps['viridis']#.resampled(n_lines)
    #     # LinearSegmentedColormap
    #     locs = []
    #     vals = []
    #     # annulus = 0
    #     annulus_width = 20
    #     fig.suptitle("Annulus Width: {}".format(annulus_width))
    #
    #     for idx, rr in enumerate(np.linspace(1.05, 1.65, n_heights)):
    #         # Pull out a line at a given height, with a given band width
    #         rr = np.round(rr, 3)
    #         clr =viridis(idx/n_heights)
    #         good_coord, bin_array, radii, the_mean, the_std, want_bin = self.get_annulus(rr, 'qrn', width=annulus_width)
    #         n, self.bins = np.histogram(bin_array, range=(0,1), bins=n_hist)
    #         nn = n.tolist()
    #         nn.append(0)
    #         nnn = np.asarray(nn)
    #         normed_nnn = nnn/np.nansum(nnn)
    #         axes[0].plot(self.bins, normed_nnn, color=clr, label=str(rr))
    #         locs.append(rr)
    #         vals.append(normed_nnn)
    #         axes[1].scatter(1.0, rr, color=clr, zorder=1050+idx)
    #
    #
    #     array = np.asarray(vals)
    #     xx, yy = np.meshgrid(self.bins, locs)
    #     hhist = axes[1].pcolormesh(xx, yy, array, cmap='YlOrRd', label="Sim Hist")
    #
    #     # xx, yy = np.meshgrid([0.98, 1.0], locs)
    #
    #     # hhist = axes[1].pcolormesh(xx, yy, locs[:, None], cmap='viridis', label="Sim Hist")
    #
    #     # cbar = fig.colorbar(hhist, ax=axes[1])
    #
    #     # fig.subplots_adjust(right=0.8)
    #     # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #     # cbar=fig.colorbar(hhist, cax=cbar_ax, aspect=40)
    #     # cbar.set_label('Cccurance')
    #
    #
    #
    #     # plt.legend()
    #     # axes[0].set_xlabel("Intensity Value")
    #     axes[0].set_ylabel("Occurance")
    #
    #     axes[1].set_xlabel("Intensity Value")
    #     axes[1].set_ylabel("Distance from Sun Center")
    #     # plt.savefig
    #     # plt.show(block=True)
    #
    #     # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms_radial_hq.png", dpi=600)
    #     plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms_radial_lq.png", dpi=400)
    #     # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms_radial_vlq.png", dpi=300)
    #     plt.close(fig)
    #
    #     a=1
    #
    #
    #     # self.params.raw_image
    #     # self.params.modified_image
    #
    def get_annulus(self, r=1.2, name="qrn", want_bin=None, width=0, load=True):
        # if not self.current_frame_name == name or self.flat_im is None:
        if load:
            frame, _, _, _, _, _ = self.load_this_fits_frame(self.fits_path, name)
        elif self.params.modified_image is not None:
            frame = self.params.modified_image
        else:
            raise FileNotFoundError

        self.flat_im = np.flipud(frame).flatten()

        want_bin = (
            int(r * self.limb_radius_from_fit_shrunken)
            if want_bin is None
            else want_bin
        )

        if width:
            the_bin_list = []
            want_range = (want_bin - width, want_bin + width)
            indices = np.arange(*want_range, 1)
            for ii in indices:
                entries, the_mean, the_std = self.get_bin_entries(ii, self.flat_im)
                (good_coord, bin_array, radii) = entries.T
                the_bin_list.append(bin_array)

            arraysize = np.nanmax([len(x) for x in the_bin_list])
            binsize = len(the_bin_list)
            newbox = np.empty((binsize, arraysize))
            newbox.fill(np.nan)
            for ii, thing in enumerate(the_bin_list):
                newbox[ii, np.arange(0, len(thing))] = thing
            bin_array = newbox
        else:
            entries, the_mean, the_std = self.get_bin_entries(want_bin, self.flat_im)
            (good_coord, bin_array, radii) = entries.T
        return good_coord, bin_array, radii, the_mean, the_std, want_bin

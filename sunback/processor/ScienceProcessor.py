# import os
# from os import makedirs
# from os.path import join, dirname
from pathlib import Path
import numpy as np
import logging
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.rcParams['image.cmap'] = 'viridis'
import os
# import itertools

# from scipy.signal import savgol_filter
# from src.processor.Processor import Processor
#
# import warnings
#
# warnings.filterwarnings("ignore")
# import matplotlib as mpl
#
# mpl.use("qt5agg")

import matplotlib.pyplot as plt

#
import os
# from os.path import join

# from astropy.io import fits
# from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
# import numpy as np
from sunback.processor.Processor import Processor
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def normalized_squared_difference_similarity(Pn, Rn, epsilon=1e-8):
    """
    Compute the Normalized Squared Difference Similarity (NSDS) between observed and modeled ratios.

    This metric is defined as:
        S = 1 - ((P - R)^2) / (P^2 + R^2 + ε)

    It is:
    - Bounded between 0 and 1
    - Smooth and differentiable
    - Robust to small magnitudes (due to ε)
    - Conceptually similar to normalized squared error

    Parameters
    ----------
    Pn : np.ndarray
        Observed ratios with shape (n_ratios, 1, height, width)
    Rn : np.ndarray
        Modeled ratios with shape (n_ratios, n_temps, 1, 1)
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-8

    Returns
    -------
    np.ndarray
        Similarity scores with shape (n_ratios, n_temps, height, width)
    """
    numerator = (Pn - Rn) ** 2
    denominator = Pn**2 + Rn**2 + epsilon
    term = numerator / denominator
    term = np.clip(term, 0, 1)
    similarity = 1 - term
    S_cube = np.mean(similarity, axis=0)
    return S_cube


AIA_TEMPERATURE_RESPONSE_TABLE = np.array([
        [7.40163203e-29, 1.46072515e-26, 5.34888608e-26, 8.50682605e-26, 2.03073186e-26, 3.35720499e-27],
        [1.40367160e-28, 2.45282476e-26, 5.69953432e-26, 7.63243171e-26, 1.69626905e-26, 2.63732660e-27],
        [2.34619332e-28, 3.55303726e-26, 8.99381587e-26, 8.00857294e-26, 1.61667405e-26, 2.65194161e-27],
        [3.50314745e-28, 4.59075911e-26, 1.69634100e-25, 8.95460085e-26, 1.65561573e-26, 2.97594545e-27],
        [4.78271991e-28, 5.36503515e-26, 3.12477095e-25, 9.96536550e-26, 1.75607919e-26, 3.35107900e-27],
        [6.12070853e-28, 5.68444094e-26, 5.24806168e-25, 1.08520987e-25, 1.91638109e-26, 3.66488663e-27],
        [7.55587098e-28, 5.47222296e-26, 7.88759175e-25, 1.16018001e-25, 2.13919425e-26, 3.86907993e-27],
        [9.17523861e-28, 4.81119761e-26, 1.05392610e-24, 1.23253858e-25, 2.41030560e-26, 3.98249755e-27],
        [1.08989852e-27, 3.85959059e-26, 1.24465730e-24, 1.33290059e-25, 2.69021271e-26, 4.00495098e-27],
        [1.24045630e-27, 2.80383406e-26, 1.28593896e-24, 1.53051294e-25, 2.93365234e-26, 3.88113016e-27],
        [1.32502623e-27, 1.83977650e-26, 1.14316216e-24, 1.96447388e-25, 3.18854130e-26, 3.62558008e-27],
        [1.31341529e-27, 1.11182849e-26, 8.56234347e-25, 2.81167898e-25, 3.70341037e-26, 3.33826036e-27],
        [1.20609026e-27, 6.50748885e-27, 5.28341859e-25, 4.06492781e-25, 4.99880175e-26, 3.04882976e-27],
        [1.04215049e-27, 3.96843481e-27, 2.62971839e-25, 5.18462720e-25, 7.85106324e-26, 2.71070234e-27],
        [9.10286530e-28, 2.61876319e-27, 1.06118152e-25, 5.21700296e-25, 1.24919334e-25, 2.50715036e-27],
        [8.66408740e-28, 1.85525324e-27, 4.07695483e-26, 3.80438138e-25, 1.66057469e-25, 2.81207972e-27],
        [7.91289650e-28, 1.39717024e-27, 2.52503209e-26, 1.89390887e-25, 1.57937960e-25, 3.70484370e-27],
        [5.79545741e-28, 1.11504283e-27, 2.46082719e-26, 6.76946014e-26, 1.03164467e-25, 4.56853237e-27],
        [3.76980809e-28, 9.38169611e-28, 1.99231167e-26, 2.29005173e-26, 5.22396021e-26, 4.79107749e-27],
        [3.07180587e-28, 8.24801234e-28, 1.03293005e-26, 1.04739857e-26, 2.40915758e-26, 4.45474563e-27],
        [3.80396260e-28, 7.43331919e-28, 3.95012736e-27, 7.10153621e-27, 1.12964568e-26, 3.86954529e-27],
        [5.90602903e-28, 6.74537063e-28, 1.80531856e-27, 6.11725282e-27, 5.94188041e-27, 3.22466361e-27],
        [9.44297921e-28, 6.14495760e-28, 1.34240926e-27, 5.72716914e-27, 3.88586017e-27, 2.60315761e-27],
        [1.44620154e-27, 5.70805277e-28, 1.22599436e-27, 5.81269453e-27, 3.15819472e-27, 2.03714253e-27],
        [2.07453706e-27, 5.61219786e-28, 1.05241421e-27, 6.32309411e-27, 2.71019768e-27, 1.53754765e-27],
        [2.75538782e-27, 6.31981777e-28, 8.14794974e-28, 6.39019697e-27, 2.11642133e-27, 1.11256462e-27],
        [3.34721420e-27, 9.19747307e-28, 6.22671197e-28, 5.43856741e-27, 1.48862847e-27, 7.71472387e-28],
        [3.65940611e-27, 1.76795732e-27, 5.15934126e-28, 4.02432562e-27, 1.02304623e-27, 5.25924670e-28],
        [3.52350628e-27, 3.77985446e-27, 4.75354344e-28, 2.82301679e-27, 7.31784516e-28, 3.94053508e-28],
        [2.91029046e-27, 7.43166191e-27, 4.74903304e-28, 2.11704426e-27, 5.53478904e-28, 3.84310717e-28],
        [2.00104132e-27, 1.19785603e-26, 4.78687900e-28, 2.23337165e-27, 4.46856485e-28, 4.58924041e-28],
        [1.11792532e-27, 1.48234676e-26, 4.55829373e-28, 3.97858677e-27, 3.92863480e-28, 5.15941951e-28],
        [5.06298440e-28, 1.36673114e-26, 4.07150741e-28, 7.95264076e-27, 3.72931301e-28, 4.64395852e-28],
        [1.98518377e-28, 9.61047146e-27, 3.57245583e-28, 1.28401071e-26, 3.65339284e-28, 3.28256715e-28],
        [8.04488438e-29, 5.61209353e-27, 3.20550268e-28, 1.62347002e-26, 3.55630868e-28, 1.96954455e-28],
        [4.22436839e-29, 3.04779780e-27, 2.95165454e-28, 1.72337033e-26, 3.40535153e-28, 1.12675591e-28],
        [3.04428119e-29, 1.69378976e-27, 2.76040754e-28, 1.64931273e-26, 3.22463539e-28, 6.77096153e-29],
        [2.64974316e-29, 1.02113491e-27, 2.60329069e-28, 1.49059780e-26, 3.03925227e-28, 4.49240089e-29],
        [2.48278975e-29, 6.82223774e-28, 2.46810279e-28, 1.30615389e-26, 2.86214839e-28, 3.30827404e-29],
        [2.38426184e-29, 5.02099099e-28, 2.34850846e-28, 1.12339098e-26, 2.69680531e-28, 2.65114499e-29],
        [2.30756823e-29, 3.99377760e-28, 2.24072357e-28, 9.54159826e-27, 2.54348518e-28, 2.25550937e-29]
    ])

AIA_TEMPERATURES = np.array([5.5 , 5.55, 5.6 , 5.65, 5.7 , 5.75, 5.8 , 5.85, 5.9 , 5.95, 6.,
                    6.05, 6.1 , 6.15, 6.2 , 6.25, 6.3 , 6.35, 6.4 , 6.45, 6.5 ,
                    6.55, 6.6 , 6.65, 6.7 , 6.75, 6.8 , 6.85, 6.9 , 6.95, 7.  ,
                    7.05, 7.1 , 7.15, 7.2 , 7.25, 7.3 , 7.35, 7.4 , 7.45, 7.5 ])


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
        # print("Setup ran!")

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

    def add_label_overlay_to_axes(self, ax, image_shape, frame_name=False, color=None):
        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(
            time_string_raw, targetZone="US/Mountain", out_fmt="%m-%d-%Y %I:%M%p %Z"
        )
        time_list = time_string.split()
        clock, day, year = time_list[1].lower(), time_list[0][:-5], time_list[0][-4:]

        inst = "AIA"
        wave = str(self.wave)

        if frame_name is None:
            frame_name = fname = str(self.current_frame_name)
        elif frame_name is False:
            fname = "None"
        else:
            fname = frame_name

        name, prev = (
            fname.split("(")[0],
            fname.split("(")[1][:-1] if "(" in fname else "-",
        )

        rez = image_shape[0]
        scale, h, wid_of_char = (
            (6, 128, 60) if rez >= 4000 else (3, 64, 30) if rez >= 2000 else (1.5, 32, 5)
        )
        h0, thickness = (100, 4) if rez >= 4000 else (50, 2) if rez >= 2000 else (30, 2)

        positions_left = [(0, rez - height) for height in [h0, h0 + h, h0 + 2 * h, h0 + 3 * h]]
        labels_left = [clock, day, year, "MT"]

        positions_right = [
            (rez - wid_of_char * len(text) - 10, rez - height)
            for text, height in zip([name, prev, inst, wave], [h0, h0 + h, h0 + 2 * h, h0 + 3 * h])
        ]

        for text, (x, y) in zip(labels_left, positions_left):
            ax.text(x, y, text, fontsize=scale * 8, color='white', ha='left', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        if isinstance(self, DEMReconstructionProcessor):
            label = f"{name}"
            # ax.text(rez, rez-60, label, fontsize=scale * 10, color="white", ha='right', va='bottom', fontweight='bold', alpha=0.8,
            # bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

            # ax.text(rez, rez-60, label, fontsize=scale * 10, color=color or "white", ha='right', va='bottom', fontweight='bold',)

            import matplotlib.patheffects as path_effects

            text = ax.text(
                rez, rez - 60, label,
                fontsize=scale * 10,
                color=color or "white",
                ha='right', va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
            )

            text.set_path_effects([
                path_effects.Stroke(linewidth=1.0, foreground='grey'),
                path_effects.Normal()
            ])

class DEMReconstructionProcessor(ScienceProcessor):
    filt_name = "DEM Reconstructor"
    description = "Perform DEM and isothermal temperature reconstruction from AIA images"
    progress_verb = "Reconstructing"
    progress_unit = "FITS Files"
    out_name = "isotherm"
    save_to_fits = False
    can_do_parallel = True
    shrink_factor = 1
    n_temp_interp = 164
    vminn = 1.25
    vmaxx = 2.75

    def setup(self):
        super().setup()
        self.channel_waves = ["0094", "0131", "0171", "0193", "0211", "0335"]
        self.response_curves, self.response_ratios = self.load_temperature_response_curves()
        self.output_folder = (
            Path(self.params.base_directory()).parent / "rainbow" / "imgs" / "mod"
        )
        logger.debug("DEMProcessor Initialized")

    def load_temperature_response_curves(self):
        import astropy.units as u
        import numpy as np
        import itertools

        # Use a finer temperature grid
        self.temperatures = np.linspace(5.5, 7.5, self.n_temp_interp) * u.K
        self.temperatures_lin = 10**self.temperatures.to_value() * u.K
        self.response_curves = {}

        for ii, wave in enumerate(self.channel_waves):
            try:
                tr = AIA_TEMPERATURE_RESPONSE_TABLE[:, ii]
                ki = AIA_TEMPERATURES * u.K
                interp_response = np.interp(
                    self.temperatures.to_value(u.K),
                    ki.to_value(u.K),
                    tr,
                    left=0,
                    right=0
                )
                self.response_curves[wave] = interp_response
            except Exception as e:
                logger.error(f"Failed to load response for {wave} Å: {e}")
                self.response_curves[wave] = np.ones_like(self.temperatures.value)

        self.response_ratios = {}
        for (w1, w2) in itertools.pairwise(self.channel_waves):
            r1 = self.response_curves[w1]
            r2 = self.response_curves[w2]
            ratio = np.divide(r1, r2, out=np.zeros_like(r1), where=r2 != 0)
            self.response_ratios[f"{w1}/{w2}"] = ratio
        logger.debug("Response Curves and Ratios Calculated")

        return self.response_curves, self.response_ratios

    def do_work(self):
        logger.debug("Starting do_work")

        self.init_radius_array()

        # Load AIA intensities across channels
        intensity_stack = self.load_channel_intensities()
        if intensity_stack is None:
            return None

        # Compute intensity ratios for adjacent channels
        self.ratios = self.compute_adjacent_ratios(intensity_stack)

        # Evaluate S(T) for all temperature models
        self.S_cube = self.evaluate_temperature_match(self.ratios)

        # Isothermal: max S(T) temperature index for each pixel
        self.plot_isothermal()

        # DEM mode: save S_cube for later plotting
        # self.dem_stack = self.S_cube  # could also write to disk
        raise StopIteration
        # return self.isothermal_map

    def load_channel_intensities(self):
        base_dir = Path(self.params.base_directory()).parent / "rainbow" / "imgs" / "fits"
        if not base_dir.exists():
            logger.error(f"FITS directory does not exist: {base_dir}")
            return None

        intensity_images = []
        for wave in self.channel_waves:
            file_path = sorted(base_dir.glob(f"*{wave}*.fits"))
            if not file_path:
                logger.warning(f"No FITS file found for wavelength {wave}")
                return None
            file_path = file_path[0]
            img = self.load_fits_data(str(file_path), 1)
            if img is None:
                logger.warning(f"Could not load image for {wave}")
                return None
            intensity_images.append(np.nan_to_num(img))

        return np.array(intensity_images)

    def compute_adjacent_ratios(self, intensity_stack):
        ratios = []
        for i in range(len(self.channel_waves) - 1):
            top = intensity_stack[i]
            bottom = intensity_stack[i + 1]
            ratio = np.divide(top, bottom, out=np.zeros_like(top), where=bottom != 0)
            ratios.append(ratio)
        return np.array(ratios)

    def evaluate_temperature_match(self, ratios, plot=True, use_chunking="auto", chunk_size=1024):
        import psutil

        logger.debug("Evaluating Similarity")

        ratios = np.array(ratios)  # (n_ratios, height, width)
        model_ratios = np.array(list(self.response_ratios.values()))  # (n_ratios, n_temps)

        n_ratios, height, width = ratios.shape
        n_temps = model_ratios.shape[1]
        n_pixels = height * width

        # Estimate memory needed for full S_cube
        bytes_needed = np.dtype(np.float32).itemsize * n_temps * n_pixels

        if use_chunking == "auto":
            in_CI = os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"
            if in_CI:
                logger.debug("Detected GitHub Actions CI environment — forcing chunked processing")
                use_chunking = True
            else:
                available_memory = psutil.virtual_memory().available
                logger.debug(f"Memory check: Need {bytes_needed/1e6:.2f} MB, Available {available_memory/1e6:.2f} MB")
                use_chunking = bytes_needed > available_memory * 0.5

        if not use_chunking:
            logger.debug("Using full-memory mode")
            Pn = ratios[:, None, :, :]  # (n_ratios, 1, height, width)
            Rn = model_ratios[:, :, None, None]  # (n_ratios, n_temps, 1, 1)
            S_cube = normalized_squared_difference_similarity(Pn, Rn)
        else:
            logger.debug("Using chunked mode")
            S_cube = np.empty((n_temps, n_pixels), dtype=np.float32)
            ratios_2d = ratios.reshape(n_ratios, -1)  # (n_ratios, n_pixels)

            for start in range(0, n_pixels, chunk_size):
                end = min(start + chunk_size, n_pixels)
                Pn_chunk = ratios_2d[:, start:end][:, None, :]  # (n_ratios, 1, chunk)
                Rn = model_ratios[:, :, None]  # (n_ratios, n_temps, 1)

                similarity = normalized_squared_difference_similarity(Pn_chunk, Rn)  # (n_temps, chunk)
                S_cube[:, start:end] = similarity

            S_cube = S_cube.reshape(n_temps, height, width)

        if plot:
            logger.debug("Plotting Similarity")
            self.plot_similarities(S_cube, True)

        return S_cube

    def plot_similarities(self, S_cube, plot_all_temps=True) -> None:
        import matplotlib.pyplot as plt

        # Flatten the spatial dimensions
        n_temps = len(self.temperatures)
        S_lines = S_cube.reshape(n_temps, -1)  # shape: (200, 1048576)

        # Optional: downsample to avoid plotting over a million lines
        n_lines_to_plot = 5000
        indices = np.linspace(0, S_lines.shape[1] - 1, n_lines_to_plot, dtype=int)
        subset = S_lines[:, indices]

        # Plot
        fig = plt.figure(figsize=(7, 6))
        plt.plot(self.temperatures_lin/10**6, subset, alpha=0.1, linewidth=1.0)
        plt.xlabel("Temperature [MK]")
        plt.ylabel("Similarity Score")
        plt.xscale("log")
        plt.title(f"NSDS Similarity Curves for {n_lines_to_plot} Pixels")
        plt.grid(True)
        plt.tight_layout()
        import os
        temp_path = os.path.join(self.output_folder, "temps")
        if os.path.exists(temp_path):
            import shutil
            shutil.rmtree(temp_path)
        os.makedirs(temp_path)

        pth = os.path.join(temp_path, "model_similarity.png")
        plt.savefig(pth)
        plt.close(fig)
        del fig

        if plot_all_temps:
            import cv2
            use_custom_colormap = False  # Set True to enable color mapping by temp

            video_path = os.path.join(self.output_folder, "a_temp_video.mp4")

            # TEMP: Render the first frame just to get correct dimensions
            dummy_img = np.zeros_like(S_cube[0])
            img_shape = dummy_img.shape
            fig = plt.figure(figsize=(img_shape[0]/100, img_shape[1]/100), dpi=100)
            fig.set_dpi(100)
            ax = fig.add_subplot(111)
            ax.imshow(dummy_img, origin='lower', cmap="viridis")
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            canvas = FigureCanvas(fig)
            canvas.draw()

            width, height = canvas.get_width_height()
            plt.close(fig)

            fps = 10
            video_path = video_path.replace(".mp4", ".avi")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mjpg'),
                fps,
                (width, height)
            )

            for ii, imgg in enumerate(tqdm(S_cube, desc="Writing Video Frames")):
                the_temp = self.temperatures_lin[ii] / 10**6
                current_temp_MK = the_temp.to_value()
                norm_val = (current_temp_MK - self.vminn) / (self.vmaxx - self.vminn)
                norm_val = np.clip(norm_val, 0.0, 1.0)
                selected_color = plt.get_cmap("plasma")(norm_val)
                if use_custom_colormap:
                    from matplotlib.colors import LinearSegmentedColormap
                    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["black", selected_color])
                    rgba = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1)).to_rgba(imgg)
                else:
                    rgba = plt.cm.viridis(imgg)

                rgb = (rgba[..., :3] * 255).astype(np.uint8)
                # height, width = rgb.shape[:2]

                fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
                fig.set_dpi(100)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove borders
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)

                ax.imshow(rgb, origin='lower', interpolation='none')  # Already RGB
                ax.set_position([0, 0, 1, 1])
                ax.set_axis_off()

                label = f"{current_temp_MK:.2f} MK"
                self.add_label_overlay_to_axes(ax, rgb.shape[:2], frame_name=label, color=selected_color)

                canvas.draw()
                canvas_width, canvas_height = canvas.get_width_height()
                # Correct shape and dtype from canvas
                rgba_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas_height, canvas_width, 4)

                # Convert to BGR for OpenCV
                bgr_img = cv2.cvtColor(rgba_img[..., :3], cv2.COLOR_RGB2BGR)
                writer.write(bgr_img)

                plt.close(fig)

            writer.release()
            logger.info(f"Video saved to {video_path}")

            # Transcode the video to H.264 for browser compatibility
            final_video_path = video_path.replace(".avi", ".mp4")
            try:
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-c:v", "libx264", "-preset", "slow", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    final_video_path
                ], check=True)
                logger.info(f"Transcoded H.264 video saved to {final_video_path}")
                import os
                os.remove(video_path)
                # os.rename(final_video_path, video_path)
                # logger.info(f"Replaced original video with transcoded version: {video_path}")
            except Exception as e:
                logger.warning(f"FFmpeg transcoding failed: {e}")


    def plot_isothermal(self):
        logger.debug("Plotting Isothermal Image")

        # Axis 0 is temperature
        isothermal_inds = np.argmax(self.S_cube, axis=0)  # (1000, 1000)

        # Get the max similarity value at each pixel
        self.similarity_map = np.take_along_axis(self.S_cube, isothermal_inds[None, :, :], axis=0)[0]

        # Map index -> temperature value
        self.isothermal_map = self.temperatures[isothermal_inds]

        self.params.modified_image = self.vignette(self.isothermal_map)
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import matplotlib.ticker as ticker
        import time

        # Get a copy of the colormap and define over/under colors
        cmap = cm.get_cmap("plasma").copy()
        cmap.set_under('navy')   # for values below vmin
        cmap.set_over('yellow') # for values above vmax


        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)  # make it square
        fig.patch.set_facecolor('grey')
        ax.set_facecolor('grey')
        ax.set_aspect('equal')

        image = 10 ** self.params.modified_image.to_value() / 1e6  # convert from log(K) to MK
        alpha = self.similarity_map
        vmin, vmax = self.vminn, self.vmaxx

        style = getattr(self.params, "visualization_style", "none")  # "alpha", "threshold", or "hsv"

        if style == "threshold":
            confidence_threshold = 0.6
            masked_image = np.where(alpha >= confidence_threshold, image, np.nan)

            cmap.set_bad('grey')
            im = ax.imshow(masked_image, origin='lower', cmap=cmap,
                        interpolation='none', vmin=vmin, vmax=vmax)

        elif style == "hsv":
            import matplotlib.colors as mcolors

            normed_temp = (image - vmin) / (vmax - vmin)
            normed_temp = np.clip(normed_temp, 0, 1)

            hsv_img = np.zeros((*image.shape, 3))
            hsv_img[..., 0] = normed_temp       # Hue = temperature
            hsv_img[..., 1] = alpha             # Saturation = similarity
            hsv_img[..., 2] = 1.0               # Brightness fixed

            rgb_img = mcolors.hsv_to_rgb(hsv_img)
            im = ax.imshow(rgb_img, origin='lower', interpolation='none')

        elif style == "overlay":  # default = alpha overlay
            im = ax.imshow(image, origin='lower', cmap=cmap, interpolation='none',
                        vmin=vmin, vmax=vmax)

            confidence_mask = 1 - alpha
            gray_overlay = np.full((*image.shape, 4), fill_value=0.0)
            gray_overlay[..., :3] = 0.5
            gray_overlay[..., 3] = confidence_mask * 0.6
            ax.imshow(gray_overlay, origin='lower', interpolation='none')
        else:
            cmap.set_bad('grey')
            im = ax.imshow(image, origin='lower', cmap=cmap, interpolation='none',
            vmin=vmin, vmax=vmax)



        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Create a small vertical colorbar inset in the upper right
        cax = inset_axes(ax, width="2.5%", height="15%", loc='upper right',
                        bbox_to_anchor=(0, 0, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0.0)

        cbar = plt.colorbar(im, cax=cax, orientation='vertical', extend='both')

        # Add a thin white border around the colorbar
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(0.5)  # You can increase to 1.0 for a stronger frame

        # # Format tick labels to display in MK
        # cbar.formatter = ticker.FuncFormatter(lambda x, _: f"{10**x/1e6:.1f}")
        # cbar.update_ticks()

        # Linearly spaced ticks in MK
        tick_labels_MK = np.arange(self.vminn, self.vmaxx + 0.5, 0.5)  #  [1.0, 1.5, 2.0, 2.5]
        cbar.set_ticks(tick_labels_MK)
        cbar.set_ticklabels([f"{mk:.1f}" for mk in tick_labels_MK])

        # cbar.set_ticks(tick_locs_log)
        # cbar.set_ticklabels([f"{mk:.1f}" for mk in tick_labels_MK])

        # Set ticks on the left
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        # Style the ticks and label
        cbar.ax.tick_params(colors='white', labelsize=8)
        label = cbar.ax.set_ylabel("Fit IsoTemp [MK]", color='white', fontsize=8, labelpad=4)
        label.set_bbox(dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'))

        # Add semi-transparent background to each tick label
        for label in cbar.ax.get_yticklabels():
            label.set_color('white')
            label.set_fontsize(8)
            label.set_bbox(dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.1'))

        # Remove axes and whitespace
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Put time and wavelength labels here:
        # img_shape = self.params.modified_image.to_value().shape
        self.add_label_overlay_to_axes(ax, image.shape, None)

        pth = os.path.join(self.output_folder, "C_isothermal.png")
        print(f"Saving to {pth}")
        plt.savefig(pth, dpi=170.66666667, facecolor='black')
        plt.close(fig)

    def cleanup(self):
        return
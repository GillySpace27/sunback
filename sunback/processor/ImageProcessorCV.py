import os
import shutil
import time
import cv2
from astropy.io import fits
from astropy.nddata import block_reduce
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from sunback.processor.ImageProcessor import ImageProcessor
from sunback.science.color_tables import aia_color_table
from sunback.utils.stretch_intensity_module import upsilon_stretch
from sunpy.map import Map
import OpenImageIO as oiio

import logging
import sys
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ImageProcessorCV(ImageProcessor):
    filt_name = "CV Image Writer"
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"
    finished_verb = "Written to Disk"
    out_name = None
    save_to_fits = False

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.frame_name = self.params.png_frame_name
        self.rhe_count = 0
        self.shrink_factor = 1
        self.save_to_fits = False

    def do_fits_function(self, fits_path, in_name=None):
        self.fits_path = fits_path
        self.params.double_rhe_flag = False
        target = -1
        out = None

        try:
            self.wave = self.params.current_wave(
                int(self.fits_path.split(".")[0][-4:])
            )
        except Exception as e:
            # print(38, int(self.params.current_wave()))
            pass
            # raise e

        try:
            # print(f"{self.params.current_wave()}")
            self.params.cmap = self.cmap = aia_color_table(
                int(self.params.current_wave()) * u.angstrom
            )
        except ValueError:
            for wv in self.params.all_wavelengths:
                if wv in fits_path:
                    self.wave = wv
                    self.params.cmap = self.cmap = aia_color_table(
                        int(self.wave) * u.angstrom
                    )
                    self.params.current_wave(int(self.wave))
                    break

        if isinstance(in_name, (int, str)) and str(in_name).isdigit():
            in_name = int(in_name)
            self.params.png_frame_name = self.find_frames_at_path(self.fits_path)[
                in_name
            ]

        if "all" in str(self.params.png_frame_name):
            self.params.png_frame_name = self.find_frames_at_path(self.fits_path)

        if target in self.params.png_frame_name:
            self.params.png_frame_name.append("mgn_" + target)
            self.params.double_rhe_flag = True

        if isinstance(self.params.png_frame_name, list) and len(
            self.params.png_frame_name
        ):
            for name in self.params.png_frame_name:
                self.current_frame = name
                self.wave = self.params.current_wave()
                self.init_frame(self.fits_path, self.current_frame)
                out = self.render_all(reference=False)
        else:
            self.current_frame = self.params.png_frame_name
            self.init_frame(self.fits_path, self.current_frame)
            out = self.render_all(reference=False)

        return out or self

    def display_all(self):
        self.display_raw()
        self.display_changed()

    def display_raw(self):
        print("lev1p0")
        self.frame = np.flipud(self.params.raw_image)
        self.prep_save()
        plt.imshow(self.frame)
        plt.title("lev1p0")
        plt.show(block=True)

    def display_changed(self):
        print("Changed")
        self.frame = np.flipud(self.params.modified_image)
        self.prep_save()
        plt.imshow(self.frame)
        plt.title("Changed")
        plt.show(block=True)

    def render_all(self, reference=False):
        if reference:
            self.plot_aia_orig()

        try:
            out = self.plot_aia_changed(self.frame_name)
            return out
        except ValueError as e:
            print(e)
            self.skipped += 1

    def do_shortcut(self):
        cat_png_path = self.cat_path
        root_folder = os.path.dirname(self.params.base_directory())
        fits_folder = os.path.dirname(self.params.use_image_path())
        cat_png_filename = os.path.basename(cat_png_path)
        shorts_folder = os.path.join(root_folder, "shorts")

        timestamp = self.image_data[2]
        short_path = os.path.join(
            shorts_folder,
            "{}_{}.png".format(self.params.current_wave(), timestamp.split(".")[0]),
        )
        os.makedirs(shorts_folder, exist_ok=True)
        shutil.copyfile(
            cat_png_path, os.path.normpath(short_path), follow_symlinks=True
        )

    def plot_aia_orig(self):
        self.frame_name = "compressed_image"
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(
            self.fits_path, self.frame_name
        )
        self.params.raw_image = self.frame = np.flipud(frame)
        self.out_path = self.get_orig_path(mod="orig")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.do_save()

    def plot_aia_log(self):
        self.frame_name = self.params.master_frame_list_oldest
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(
            self.fits_path, self.frame_name
        )

        frame = np.log10(frame)
        frame = frame / np.nanpercentile(frame, 50) / 2

        self.frame = np.flipud(frame)
        self.out_path = self.get_orig_path(mod="log")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.do_save()

    def do_save(self, do_small=False):
        self.prep_save(do_small=do_small)
        self.img_save(self.out_path)

    def plot_aia_changed(self, frame_name=None):
        if frame_name is None:
            frame_name = self.params.png_frame_name

        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(
            self.fits_path, frame_name
        )
        self.frame = np.flipud(frame)
        if self.frame.max() > 2 or self.frame.min() < 0:
            self.frame = self.normalize(self.frame)

        upsilon = self.params.upsilon
        if upsilon is not None:
            self.params.upsilon_high = upsilon[1] if len(upsilon) > 1 else upsilon
            self.params.upsilon_low = upsilon[0] if len(upsilon) > 1 else upsilon

        if upsilon and False:
            self.frame = upsilon_stretch(
                self.frame,
                upsilon=self.params.upsilon_low,
                upsilon_high=self.params.upsilon_high,
            )

        self.out_path = self.get_changed_path()
        self.out_path = self.out_path.replace("AIAsynoptic", "DrGilly_").replace(
            ".png", f"_{self.frame_name}.png"
        )

        self.do_save(
            do_small=True
            if "MultiImage".casefold() in self.filt_name.casefold()
            else False
        )
        self.params.current_wave(None)
        return self.frame if self.params.double_rhe_flag else None

    @staticmethod
    def geo_mean(iterable, axis=0):
        a = np.array(iterable)
        return np.prod(a, axis=axis) ** (1.0 / len(a))

    def prep_save(self, do_small=False):
        self.make_image(do_small)

    def make_image(self, do_small=False):
        out = self.frame_touchup(self.frame_name, self.frame + 0)
        # out = self.frame + 0.0
        self.params.rbg_image = []
        self.params.rbg_labels = ["hq"]
        self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
        frames = [out]

        if do_small:
            frames.extend(
                [block_reduce(out, 2, np.nanmean), block_reduce(out, 4, np.nanmean)]
            )
            self.params.rbg_labels.extend(["lq", "vlq"])
        for frame in frames:
            self.img_frame = (self.params.cmap(frame)[:, :, :3] * 255).astype(np.uint8)
            try:
                img = self.label_plot(self.img_frame)
            except (ValueError, AttributeError) as e:
                print(186, e)
                img = self.img_frame
            b, g, r = cv2.split(img)
            rgb_img = cv2.merge([r, g, b])
            self.params.rbg_image.append(rgb_img)
        self.path_box.append(self.out_path)

    def img_save(self, path, save=True, stamp=False):
        aH, aL = self.params.upsilon_high, self.params.upsilon_low
        master_path = path
        if "rhe" in self.frame_name and stamp:
            path = path.replace(".png", f"_ah-{aH:0.2f}_aL-{aL:0.2f}.png")

        if save:
            for img, rez in zip(self.params.rbg_image, self.params.rbg_labels):
                if len(self.params.rbg_labels) > 1:
                    path = master_path.replace(".png", f"_{rez}.png")
                cv2.imwrite(path, img)
        else:
            plt.imshow(self.params.rbg_image)
            plt.show()

    def label_plot(self, img_in=None, max=None, do=None):
        if do is False or self.params.label_imgs is False:
            return img_in
        if img_in is None:
            img = self.params.rbg_image[0]
        else:
            img = img_in + 0

        from sunback.processor.CompositeRainbowImageProcessor import RainbowRGBImageProcessor

        if isinstance(self, RainbowRGBImageProcessor):
            self.init_rainbow_frame()
            wave = "-"

        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(
            time_string_raw, targetZone="US/Mountain", out_fmt="%m-%d-%Y %I:%M%p %Z"
        )
        time_list = time_string.split()
        clock, day, year = time_list[1].lower(), time_list[0][:-5], time_list[0][-4:]

        inst = "AIA"

        if not isinstance(self, RainbowRGBImageProcessor):
            wave = self.clean_name_string(full_name)[1]

        fname, frame_name = self.frame_name.casefold(), self.frame_name
        name, prev = (
            fname.split("(")[0],
            fname.split("(")[1][:-1] if "(" in fname else "-",
        )

        rez = img.shape[0]
        scale, h, wid_of_char = (
            (6, 120, 60)
            if rez >= 4000
            else (3, 60, 30)
            if rez >= 2000
            else (1.5, 30, 15)
        )
        h0, thickness = (100, 4) if rez >= 4000 else (50, 2) if rez >= 2000 else (25, 2)

        positions = [
            (rez - wid_of_char * len(text) - 10, height)
            for text, height in zip(
                [name, prev, inst, wave], [h0, h0 + h, h0 + 2 * h, h0 + 3 * h]
            )
        ]
        val = max or (255 if np.max(img) > 1 else 1.0)
        max3 = (val, val, val)

        for text, (x, y) in zip([name, prev, inst, wave], positions):
            cv2.putText(img, text, (x, y), 1, scale, max3, thickness)

        positions = [(0, height) for height in [h0, h0 + h, h0 + 2 * h, h0 + 3 * h]]
        for text, (x, y) in zip([clock, day, year, "MT"], positions):
            cv2.putText(img, text, (x, y), 1, scale, max3, thickness)

        try:
            if wave == "-":
                return img
            self.get_alphas(wave=wave)
            aH = self.params.upsilon_high if self.params.do_upsilon else None
            aL = self.params.upsilon_low if self.params.do_upsilon else None
            cv2.putText(
                img,
                f"aH: {aH}",
                (0, int(0.97 * rez)),
                1,
                scale,
                max3,
                thickness,
            )
            cv2.putText(
                img,
                f"aL: {aL}",
                (0, int(0.99 * rez)),
                1,
                scale,
                max3,
                thickness,
            )
        except (SystemError, ValueError) as e:
            print(238, e)

        return img

    def draw_reticle(self, img):
        cv2.circle(
            img,
            (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
            int(self.params.header["R_SUN"]),
            (255, 255, 255),
            3,
        )
        cv2.circle(
            img,
            (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
            10,
            (255, 0, 0),
            10,
        )


    @staticmethod
    def peek_frame(img):
        cv2.imshow("win2", img[::5, ::5, ::5])
        cv2.waitKey(0)


class MultiImageProcessorCv(ImageProcessorCV):
    filt_name = "MultiImage Plotter"
    description = "Look at the different methods compared"
    progress_verb = "Writing"
    progress_unit = "Images"

    # list_of_inputs = ["lev1p5", "t_int", "lev1p0"]

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.dont_vminmax = False
        self.max_width = 20
        self.main_save_path = None
        self.last_frame_name = None
        self.base_image = None
        self.n_plots = None
        self.n_cols = None
        self.n_rows = None
        self.init = False
        self.count_frames = 0
        self.frame_names = []
        self.frames = []
        self.good_frames = []
        self.good_frame_stems = []
        self.fig = None

    def do_fits_function(self, fits_path, in_name=None, doBar=False):
        """Main Call on the Fits Path"""

        self.init_frame_from_fits(fits_path)
        self.init_quad_figure()
        self.init_radius_array()

        self.collect_frames(fits_path, doBar)

        self.finalize_and_save_plots()
        self.reinit_constants()

        # self.open_folder(self.main_save_path)
        return False

    def collect_frames(self, fits_path, doBar, hist=False):
        self.max_width = np.max([len(x) for x in self.good_frames])
        iterable = tqdm(self.good_frames, desc="") if doBar else self.good_frames
        for frame_name in iterable:
            if self.image_is_plottable(frame_name):
                if frame_name == "jpeg":
                    self.plot_jpeg(fits_path, frame_name, doBar, iterable)
                else:
                    self.handle_one_frame(fits_path, frame_name, doBar, iterable)
                self.count_frames += 1
        # print("\rCollected {} frames for comparison".format(self.count_frames))
        if doBar:
            iterable.set_description(" *    Plots Complete", refresh=True)

    def plot_jpeg(self, fits_path, frame_name, doBar, iterable):
        import PIL
        from PIL import Image

        j_directory = os.path.join(self.params.imgs_top_directory(), "jpeg")
        try:
            paths = os.listdir(j_directory)
        except FileNotFoundError as e:
            print("\nNo JPEG Image Found")
            self.params.doing_jpeg = False
            return
            # paths = []
        full_paths = [os.path.join(j_directory, pat) for pat in paths]
        wavenum = int("".join(i for i in fits_path if i.isdigit()))
        wave_path = [x for x in full_paths if str(wavenum) in x]
        if len(wave_path):
            correct = wave_path[0]
        else:
            rr = self.params.rez
            correct = 0.75 * np.ones((rr, rr))
        image = Image.open(correct)
        # from astropy.nddata import block_reduce
        # frame = block_reduce(frame, self.shrink_factor/2)
        # frame=  frame.rotate(180)
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        ax = self.axArray[self.count_frames]
        ax.imshow(image, origin="lower", interpolation="None")
        ax.set_title('"The Sun Today"')

        return

    def handle_one_frame(self, fits_path, frame_name, doBar, iterable):
        if doBar:
            iterable.set_description(
                " *     Plotting {}".format(frame_name.ljust(self.max_width)),
                refresh=True,
            )
        frame1, wave1, t_rec1, center1, int_time, name = self.load_this_fits_frame(
            fits_path, frame_name
        )
        # frame1[self.vignette_mask] = np.nan
        self.add_to_plot(name, frame1)

    def init_frame_from_fits(self, fits_path=None, in_name=-1):
        """Load the fits file from disk and get a in_name or two"""

        self.fits_path = fits_path or self.fits_path
        self.params.fits_path = self.fits_path

        self.params.raw_image, _, _, _, _, self.raw_name = self.load_this_fits_frame(
            fits_path, self.params.master_frame_list_oldest
        )

        self.params.modified_image, wave1, t_rec1, _, _, self.mod_name = (
            self.load_this_fits_frame(fits_path, in_name)
        )

        # self.peek_frames()
        self.image_data = (
            str(wave1),
            fits_path,
            t_rec1,
            self.params.modified_image.shape,
        )
        self.params.make_file_paths(self.image_data)
        self.name, self.wave = self.clean_name_string(str(wave1))

    def open_folder(self, path):
        import webbrowser

        webbrowser.open("file:///" + path)

    def init_quad_figure(self):
        self.good_frames = [x for x in self.hdu_name_list if self.image_is_plottable(x)]
        use_cmap = True
        if use_cmap is not None:
            self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
        else:
            from matplotlib import cm

            self.params.cmap = cm.greens

        # try:
        #     lev1p5_mask = ['lev1p5' in x for x in self.good_frames]
        #     lev1p5_loc = np.where(lev1p5_mask)[0][0]
        #     repeat_frame = lev1p5_loc
        # except IndexError as e:
        #     print('\r' + str(e))
        #     repeat_frame = 0
        #
        # self.good_frames.insert(repeat_frame, self.good_frames[repeat_frame])
        # self.good_frames.pop(0)
        self.params.doing_jpeg = self.params.doing_jpeg
        if self.params.doing_jpeg:
            self.good_frames.insert(0, "jpeg")

        self.good_frame_stems = [x.split("(")[0] for x in self.good_frames]

        self.n_plots = len(self.good_frames)
        self.n_rows = 2 if self.n_plots > 2 else 1
        self.n_cols = max((self.n_plots // self.n_rows, 1)) if self.n_plots > 2 else 2
        self.n_slots = self.n_rows * self.n_cols
        while self.n_slots < self.n_plots:
            self.n_cols += 1
            self.n_slots = self.n_rows * self.n_cols

        self.fig, self.axArray = plt.subplots(
            self.n_rows, self.n_cols, sharex="all", sharey="all"
        )

        try:
            t_rec = self.header["T_REC"]
        except KeyError as e:
            t_rec = self.header["T_OBS"]

        self.fig.suptitle("{}  at  {}".format(self.wave, t_rec))
        self.axArray = self.axArray.flatten()

        blank = np.zeros_like(self.params.modified_image)

        for ax in self.axArray:
            ax.imshow(blank, interpolation="None")
            ax.set_title(" ")

    def add_to_plot(self, frame_name_in, frame):
        # print("\r * Adding Plot  {}".format(frame_name_in))
        # if 'primary' in frame_name_in:
        #     suffix = "_orig"
        # else:
        suffix = ""
        frame_name = frame_name_in + suffix
        self.last_frame_name = frame_name_in

        # frame = self.frame_touchup(frame_name, frame)

        if "rht" in frame_name:
            self.axArray[self.count_frames].imshow(
                frame, cmap="hsv", origin="lower", interpolation="None"
            )
        else:
            vmin = None if self.dont_vminmax else 0.0
            vmax = None if self.dont_vminmax else 1.0
            self.axArray[self.count_frames].imshow(
                frame,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap=self.params.cmap,
                interpolation="None",
            )

        self.axArray[self.count_frames].set_title(frame_name)
        self.axArray[self.count_frames].patch.set_alpha(0)

        frame = self.vignette(frame)
        self.frame_names.append(frame_name)
        self.frames.append(frame)

    def finalize_and_save_plots(self, dpi=250):
        inches = 4
        colWid = self.n_cols * inches
        rowWid = self.n_rows * inches

        self.fig.set_size_inches(w=colWid, h=rowWid)
        plt.tight_layout()

        save_path = os.path.join(
            self.params.imgs_top_directory(),
            "compare",
            "{:04}_compare.png".format(int(self.wave)),
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.main_save_path = save_path
        logger.info(self.save_path)
        self.fig.savefig(save_path, dpi=dpi)
        # plt.show(block=True)
        if False:
            self.plot_zooms()

    def plot_zooms(self, dpi=500):
        zooms = os.path.join(self.params.imgs_top_directory(), "zooms")
        os.makedirs(zooms, exist_ok=True)

        save_path = os.path.join(zooms, "{:04}_compare.png".format(int(self.wave)))
        self.fig.savefig(save_path, dpi=dpi)

        save_path = os.path.join(
            zooms, "1_zoom_{:04}_compare.png".format(int(self.wave))
        )
        plt.xlim((3250 / self.shrink_factor, 4000 / self.shrink_factor))
        plt.ylim((2250 / self.shrink_factor, 3000 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)

        save_path = os.path.join(
            zooms, "2_zoom_{:04}_compare.png".format(int(self.wave))
        )
        plt.xlim((2404 / self.shrink_factor, 3500 / self.shrink_factor))
        plt.ylim((3000 / self.shrink_factor, 4096 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)

        # plt.close(self.fig)
        # print("Done - Files Saved in {}".format(self.params.imgs_top_directory()))

    def reinit_constants(self):
        self.count_frames = 0
        self.last_frame_name = None
        plt.close(self.fig)


class MultiHistogramProcessorCv(MultiImageProcessorCv):
    filt_name = "MultiHistogram Plotter"
    description = "Look at the different methods compared"
    progress_verb = "Writing"
    progress_unit = "Images"

    # list_of_inputs = ["lev1p5", "t_int", "lev1p0"]

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.dont_vminmax = False
        self.max_width = 20
        self.main_save_path = None
        self.last_frame_name = None
        self.base_image = None
        self.n_plots = None
        self.n_cols = None
        self.n_rows = None
        self.init = False
        self.count_frames = 0
        self.frame_names = []
        self.frames = []
        self.good_frames = []
        self.good_frame_stems = []
        self.fig = None

    def modify_one_fits(self, fits_path):
        """Apply the given funtion to the given fits path"""
        self.confirm_fits_file(fits_path)
        self.do_fits_function(fits_path)
        self.ii = 1 if self.ii is None else self.ii + 1
        return False

    def do_fits_function(self, fits_path, in_name=None, doBar=False):
        """Main Call on the Fits Path"""
        self.init_frame_from_fits(fits_path)

        self.init_radius_array()
        self.init_bin_array()
        self.init_statistics()

        # self.init_quad_figure()

        self.collect_frames(fits_path, doBar)

        # self.finalize_and_save_plots()
        # self.reinit_constants()

        # self.open_folder(self.main_save_path)
        return None

    @staticmethod
    def gamma_correct_map(frame, gamma=0.65):
        """
        Applies gamma correction to an ndarray.

        Parameters:
            frame (np.ndarray): frame to be operated on
            gamma (float): Gamma correction factor (>1 brightens, <1 darkens).

        Returns:
            sunpy.map.Map: The gamma-corrected SunPy map.
        """
        # Normalize the data between 0 and 1
        data_min, data_max = np.nanmin(frame), np.nanmax(frame)
        normalized_data = (frame - data_min) / (data_max - data_min)

        # Apply gamma correction
        gamma_corrected_data = normalized_data ** gamma

        # Rescale back to original range
        corrected_data = gamma_corrected_data * (data_max - data_min) + data_min

        # Create a new SunPy frame with corrected data
        return corrected_data



    def collect_frames(self, fits_path, doBar=False, short=False):
        self.find_frames_at_path(fits_path)
        logger.info(self.hdu_name_list)
        self.good_frames = [x for x in self.hdu_name_list if self.image_is_plottable(x)]
        logger.info(self.good_frames)

        self.max_width = np.max([len(x) for x in self.good_frames])
        iterable = tqdm(self.good_frames, desc="") if doBar else self.good_frames
        images, names = [], []
        for frame_name in iterable:
            frame_name = frame_name.casefold()
            frame, wave1, t_rec1, _, _, mod_name = self.load_this_fits_frame(
                fits_path, frame_name
            )

            frame_name = frame_name.replace("primary data array", "prim")

            if len(frame.shape) > 2:
                frame0 = frame[0]
                frame1 = frame[1]

                images.append(frame0), names.append(frame_name + "_total")
                images.append(frame1), names.append(frame_name + "_pb")

            if False:
                frame = self.resize_image(frame, prnt=False, func=np.nanmean)

            if frame_name == "compressed_image" or frame_name == "prim":
                images.append(frame), names.append("original")
                images.append(self.gamma_correct_map(frame)), names.append("gamma")
                # images.append(np.log10(frame)), names.append("log10")

                if short:
                    continue
            elif "msgn(" in frame_name:
                images.insert(3, frame), names.insert(3, frame_name.replace("compressed_image", "lev1p5"))
            elif frame_name.startswith("nrgf"):
                images.insert(2, frame), names.insert(2, frame_name.replace("compressed_image", "lev1p5"))

            elif "compressed_image" in frame_name:
                frame_name = frame_name.replace("compressed_image", "lev1p5")
                images.append(frame), names.append(frame_name)
            else:
                if frame_name.startswith("uncertainty"):
                    continue
                images.append(frame), names.append(frame_name)

        logger.info(names)

        # self.do_compare_histogramplot_images(images, names)

        # self.do_compare_histogramplot_rheonly(
        #     images, names, target_names=["lev1p5", "rhef", "upsilon(rhef)"]
        # )
        self.do_compare_histogramplot(images, names, even_points=25 if self.params.use_image_path() else 50)

        if doBar:
            iterable.set_description(" *    Plots Complete", refresh=True)



    def finalize_and_save_plots(self, dpi=200):
        pass

    def plot_zooms(self, dpi=500):
        zooms = os.path.join(self.params.imgs_top_directory(), "zooms")
        os.makedirs(zooms, exist_ok=True)

        save_path = os.path.join(zooms, "{:04}_compare.png".format(int(self.wave)))
        self.fig.savefig(save_path, dpi=dpi)

        save_path = os.path.join(
            zooms, "1_zoom_{:04}_compare.png".format(int(self.wave))
        )
        plt.xlim((3250 / self.shrink_factor, 4000 / self.shrink_factor))
        plt.ylim((2250 / self.shrink_factor, 3000 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)

        save_path = os.path.join(
            zooms, "2_zoom_{:04}_compare.png".format(int(self.wave))
        )
        plt.xlim((2404 / self.shrink_factor, 3500 / self.shrink_factor))
        plt.ylim((3000 / self.shrink_factor, 4096 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)

        # plt.close(self.fig)
        # print("Done - Files Saved in {}".format(self.params.imgs_top_directory()))

    def reinit_constants(self):
        self.count_frames = 0
        self.last_frame_name = None
        plt.close(self.fig)


    def plot_jpeg(self, fits_path, frame_name, doBar, iterable):
        import PIL
        from PIL import Image

        j_directory = os.path.join(self.params.imgs_top_directory(), "jpeg")
        try:
            paths = os.listdir(j_directory)
        except FileNotFoundError as e:
            print("\nNo JPEG Image Found")
            self.params.doing_jpeg = False
            return
            # paths = []
        full_paths = [os.path.join(j_directory, pat) for pat in paths]
        wavenum = int("".join(i for i in fits_path if i.isdigit()))
        wave_path = [x for x in full_paths if str(wavenum) in x]
        if len(wave_path):
            correct = wave_path[0]
        else:
            rr = self.params.rez
            correct = 0.75 * np.ones((rr, rr))
        image = Image.open(correct)
        # from astropy.nddata import block_reduce
        # frame = block_reduce(frame, self.shrink_factor/2)
        # frame=  frame.rotate(180)
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        ax = self.axArray[self.count_frames]
        ax.imshow(image, origin="lower", interpolation="None")
        ax.set_title('"The Sun Today"')

        return


import os
import numpy as np
import OpenEXR, Imath
import subprocess
import astropy.io.fits as fits

class ImageProcessorHDR(ImageProcessorCV):
    """Processor for converting FITS files into HDR formats (OpenEXR or HEIC)"""

    name = filt_name = "HDR Image Writer"
    description = "Convert FITS files into HDR OpenEXR or HEIC"
    progress_verb = "Processing"
    finished_verb = "Saved"
    out_name = "HDR({})"
    save_to_fits = False

    def __init__(self, params=None, quick=False, rp=None, output_format="heic"):
        """Initialize the HDR processor.

        Args:
            output_format (str): 'exr' for OpenEXR, 'heic' for HEIC HDR.
        """
        super().__init__(params, quick, rp)
        self.output_format = output_format.lower()
        self.out_name = self.out_name.format(self.output_format)

    def do_fits_function(self, fits_path, in_name=None):
        """Load, process, and save FITS data in HDR format."""
        self.fits_path = fits_path
        self.init_frame()
        hdr_image = self.load_fits_data(fits_path)

        hdr_image=self.label_plot(hdr_image, max=0.1)

        if hdr_image is None:
            print(f"[ERROR] Skipping {fits_path}: Failed to load data.")
            return None

        output_path_heic = self.get_output_path(fits_path, "heic")
        self.save_heic(hdr_image, output_path_heic)

        output_path_exr = self.get_output_path(fits_path, "exr")
        self.save_exr(hdr_image, output_path_exr)

        self.convert_exr_to_hdr_video(output_path_exr)

        return self

    def load_fits_data(self, fits_path):
        """Load a FITS file and normalize the image."""
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[-1].data.astype(np.float32)
                data = np.nan_to_num(data, nan=0.5, posinf=1.0, neginf=0.0)
                data = np.flipud(data)
                return data
        except Exception as e:
            print(f"[ERROR] Failed to load FITS {fits_path}: {e}")
            return None



    def save_exr_RGB(self, image, filename):
        """Save image as 16-bit OpenEXR."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header = OpenEXR.Header(image.shape[1], image.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}

        # Convert to OpenEXR format
        print(image.shape, image.dtype)
        image_data = (image.astype(np.float16)).tobytes()

        exr_file = OpenEXR.OutputFile(filename, header)
        exr_file.writePixels({'R': image_data, 'G': image_data, 'B': image_data})
        exr_file.close()
        print(f"Saved HDR image: {filename}")

    def apply_pq_curve(self, image):
        """Apply Perceptual Quantizer (PQ) transfer function for HDR mapping."""
        return np.power(image, 2.2)  # PQ Approximation

    def save_exr(self, image, filename):
        """Save grayscale image as a 16-bit OpenEXR with a single luminance (Y) channel."""
        import os
        import OpenEXR
        import Imath
        import numpy as np

        # image = self.apply_pq_curve(image*100)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Create header with a single Y (luminance) channel
        header = OpenEXR.Header(image.shape[1], image.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = {'Y': half_chan}  # Single-channel grayscale

        # def log_tonemap(image, scale=1000):
        #     """Apply logarithmic tone mapping to preserve detail"""
        #     return np.log1p(image * scale) / np.log1p(scale)

        scale = 2.2
        # image_half = log_tonemap(image.astype(np.float16))
        image_half = np.power(image.astype(np.float16), scale)
        image_half *= 3.5  # Scale into HDR10 nits
        # image_half = np.log10(image_half)*10
        image_bytes = image_half.tobytes()

        # Write the single Y channel
        exr_file = OpenEXR.OutputFile(filename, header)
        exr_file.writePixels({'Y': image_bytes})
        exr_file.close()
        print(f"Saved grayscale HDR image: {filename}")


    def convert_exr_to_hdr_video(self, input_exr, frame_rate=1):
        """
        Converts a grayscale EXR file to an HDR10 MP4 video.
        Steps:
        1. Converts EXR to a 16-bit PNG (RGB)
        2. Uses FFmpeg to encode the PNG into an HDR video
        """

        # Ensure the output directory exists
        # temp_png = "temp_hdr.png"
        temp_png = input_exr.replace(".exr", "hdr.png")
        output_video = input_exr.replace(".exr", ".mp4")

        # Load EXR file
        img = oiio.ImageInput.open(input_exr)
        if not img:
            raise ValueError(f"❌ Failed to open EXR file: {input_exr}")

        spec = img.spec()
        pixels = img.read_image(oiio.TypeDesc("half"))  # Corrected format
        img.close()

        if pixels is None:
            raise ValueError(f"❌ Error reading image data from {input_exr}")

        # Ensure it's grayscale, then expand to RGB
        if spec.nchannels == 1:
            pixels = np.repeat(pixels, 3, axis=-1)  # Convert 1-channel grayscale to 3-channel RGB

        # Apply tonemapping (Gamma correction + PQ scaling)
        pixels = np.clip(pixels ** 2.2 * 100, 0, 1)  # Scale to HDR10 nits

        # Save as 16-bit PNG
        out_spec = oiio.ImageSpec(spec.width, spec.height, 3, oiio.TypeDesc("uint16"))
        out_img = oiio.ImageOutput.create(temp_png)
        if not out_img:
            raise ValueError(f"❌ Failed to create output PNG file: {temp_png}")

        out_img.open(temp_png, out_spec)
        out_img.write_image((pixels * 65535).astype(np.uint16))
        out_img.close()

        print(f"✅ Saved PNG: {temp_png}")

        # Verify PNG file size
        png_size = os.path.getsize(temp_png)
        if png_size < 1024 * 10:  # At least 10 KB
            raise ValueError("❌ Error: PNG file seems too small, conversion may have failed.")

        # Run FFmpeg to convert PNG to HDR10 MP4
        ffmpeg_cmd = f"""
        ffmpeg -r {frame_rate} -loop 1 -i {temp_png} -vf "scale=in_color_matrix=bt2020:out_color_matrix=bt2020,format=yuv420p10le" \
        -c:v libx265 -t 5 -pix_fmt yuv420p10le -color_primaries bt2020 -color_trc smpte2084 -color_range tv \
        -b:v 50M -crf 18 -preset slow -x265-params "hdr10=1:hdr10-opt=1" {output_video}
        """

        print("Running FFmpeg command:\n", ffmpeg_cmd)
        subprocess.run(ffmpeg_cmd, shell=True, check=True)

        # Verify MP4 file size
        mp4_size = os.path.getsize(output_video)
        if mp4_size < 1024 * 3000:  # At least 3 MB
            print("❌ Error: MP4 file seems too small, encoding may have failed.")

        # Clean up
        # os.remove(temp_png)

        print(f"✅ HDR10 video created: {output_video}")

    # def convert_exr_to_hdr_video(self, input_exr, frame_rate=1):
    #     """
    #     Converts a grayscale EXR file to an HDR10 MP4 video.
    #     Steps:
    #     1. Converts EXR to a 16-bit PNG (RGB)
    #     2. Uses FFmpeg to encode the PNG into an HDR video
    #     """

    #     # Ensure the output directory exists
    #     temp_png = "temp_hdr.png"
    #     # output_video = os.path.join(os.path.dirname(input_exr), output_video)
    #     output_video = input_exr.replace(".exr", ".mp4")
    #     # Load EXR file
    #     img = oiio.ImageInput.open(input_exr)
    #     spec = img.spec()
    #     pixels = img.read_image(oiio.TypeDesc("half"))  # Corrected format
    #     img.close()

    #     # Ensure it's grayscale, then expand to RGB
    #     if spec.nchannels == 1:
    #         pixels = np.repeat(pixels, 3, axis=-1)  # Convert 1-channel grayscale to 3-channel RGB

    #     # Apply basic tonemapping (Gamma correction)
    #     pixels = np.clip(pixels ** 2.2 * 3.5, 0, 1)  # Scale to HDR10 nits

    #     # Save as 16-bit PNG
    #     out_spec = oiio.ImageSpec(spec.width, spec.height, 3, oiio.TypeDesc("uint16"))
    #     out_img = oiio.ImageOutput.create(temp_png)
    #     out_img.open(temp_png, out_spec)
    #     out_img.write_image((pixels * 65535).astype(np.uint16))
    #     out_img.close()

    #     # Run FFmpeg to convert PNG to HDR10 MP4
    #     ffmpeg_cmd = f"""
    #     ffmpeg -r {frame_rate} -loop 1 -i {temp_png} -c:v libx265 -t 5 -pix_fmt yuv420p10le \
    #     -color_primaries bt2020 -color_trc smpte2084 -color_range tv -b:v 50M -crf 18 -preset slow \
    #     -x265-params "hdr10=1:hdr10-opt=1" {output_video}
    #     """

    #     print("Running FFmpeg command:\n", ffmpeg_cmd)
    #     subprocess.run(ffmpeg_cmd, shell=True, check=True)

    #     # Clean up
    #     os.remove(temp_png)

    #     print(f"✅ HDR10 video created: {output_video}")


    def save_heic(self, image, filename):
        """Save a grayscale HDR image as a 10-bit HEIC file using FFmpeg, with HDR PNG for debugging."""

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        height, width = image.shape
        raw_yuv = filename.replace(".heic", ".yuv")
        png_filename = filename.replace(".heic", ".png")

        # Convert grayscale image to 16-bit and scale to full range (HEVC expects 10-bit YUV)
        yuv_data = (image * 65535).astype(np.uint16)
        yuv_data.tofile(raw_yuv)

        # Convert raw YUV to PNG for debugging (preserving HDR dynamic range)
        print(f"🖼️  Converting YUV to HDR PNG: {png_filename}")

        png_conversion_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-video_size", f"{width}x{height}",
            "-pixel_format", "gray16le",
            "-i", raw_yuv,
            "-vf", "format=gray16le",
            png_filename
        ]
        result = subprocess.run(png_conversion_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"❌ Failed to create HDR PNG: {png_filename}")
            print(result.stderr.decode())
            return

        print(f"✅ HDR PNG saved: {png_filename} (Check for correct dynamic range)")

        # FFmpeg command for HEIC conversion
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-video_size", f"{width}x{height}",
            "-pixel_format", "gray16le",
            "-i", raw_yuv,
            "-c:v", "libx265",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020",
            "-color_trc", "pq",
            "-colorspace", "bt2020nc",
            "-f", "heif",
            "-frames:v", "1",
            filename
        ]

        print(f"[DEBUG] Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0 and os.path.exists(filename):
            print(f"✅ HEIC saved via FFmpeg: {filename}")
            # os.remove(raw_yuv)  # Clean up intermediate YUV file
            return

        print(f"⚠️ FFmpeg failed to create HEIC. Trying ImageMagick...")

        # Try ImageMagick (`convert`) as a fallback
        if shutil.which("convert"):
            convert_cmd = ["convert", png_filename, "-define", "heic:preserve-ar", "-quality", "90", filename]
            result = subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode == 0 and os.path.exists(filename):
                print(f"✅ HEIC saved via ImageMagick: {filename}")
                # os.remove(raw_yuv)  # Clean up intermediate YUV file
                return

            print(f"⚠️ ImageMagick failed. Trying `heif-enc`...")

        # Try `heif-enc` as the final fallback
        if shutil.which("heif-enc"):
            heif_cmd = ["heif-enc", png_filename, "-o", filename]
            result = subprocess.run(heif_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode == 0 and os.path.exists(filename):
                print(f"✅ HEIC saved via heif-enc: {filename}")
                # os.remove(raw_yuv)  # Clean up intermediate YUV file
                return

            print(f"❌ Failed to save HEIC using all methods.")

        print(f"⚠️ The PNG file {png_filename} has been preserved for manual inspection.")

    def get_output_path(self, fits_path, extension):
        """Generate output file path based on input FITS file."""
        base_dir = os.path.dirname(os.path.dirname(fits_path))
        base_name = os.path.splitext(os.path.basename(fits_path))[0]
        output_dir = os.path.join(base_dir, "HDR_Output")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{base_name}.{extension}")

    # def save_heic(self, image, filename):
    #     """Save image as 10-bit HEIC using ffmpeg."""
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     height, width = image.shape
    #     raw_yuv = filename.replace(".heic", ".yuv")

    #     # Convert NumPy array to raw 16-bit YUV (HEVC compatible format)
    #     yuv_data = (image * 65535).astype(np.uint16)
    #     yuv_data.tofile(raw_yuv)

    #     # Use ffmpeg to encode HEIC
    #     ffmpeg_cmd = [
    #         "ffmpeg", "-y", "-f", "rawvideo",
    #         "-video_size", f"{width}x{height}",
    #         "-pixel_format", "gray16le",
    #         "-i", raw_yuv,
    #         "-c:v", "hevc",
    #         "-pix_fmt", "yuv420p10le",
    #         "-color_primaries", "bt2020",
    #         "-color_trc", "pq",
    #         "-colorspace", "bt2020nc",
    #         filename
    #     ]

    #     print(f"[DEBUG] Running ffmpeg command: {' '.join(ffmpeg_cmd)}")

    #     result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    #     if os.path.exists(filename):
    #         print(f"Saved HDR image: {filename}")
    #         os.remove(raw_yuv)  # Clean up intermediate file
    #     else:
    #         print(f"[ERROR] Failed to create HEIC file: {filename}")



    # def process_directory(self, directory):
    #     """Process all FITS files in the specified directory."""
    #     fits_files = list(Path(directory).rglob("*.fits"))
    #     if not fits_files:
    #         print("No FITS files found in the directory.")
    #         return

    #     for fits_path in fits_files:
    #         self.do_fits_function(str(fits_path))

# # Example usage
# if __name__ == "__main__":
#     processor = ImageProcessorHDR(output_format="heic")  # or "exr"
#     processor.process_directory("/path/to/your/fits/directory")

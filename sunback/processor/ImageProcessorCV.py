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
from sunback.utils.stretch_intensity_module import norm_stretch

class ImageProcessorCV(ImageProcessor):
    filt_name = "CV Image Writer"
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"
    finished_verb = "Written to Disk"
    out_name = "final"

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.frame_name = self.params.png_frame_name
        self.rhe_count = 0
        self.shrink_factor = 1

    def do_fits_function(self, fits_path, in_name=None):
        self.params.double_rhe_flag = False
        self.fits_path = fits_path
        target = "rhe(lev1p5)"
        out = None

        if self.params.current_wave() in [None, "rainbow"]:
            try:
                self.wave = self.params.current_wave(int(self.fits_path.split('.')[0][-4:]))
            except Exception as e:
                print(38, int(self.params.current_wave()))
                raise e

        try:
            self.params.cmap = self.cmap = aia_color_table(int(self.params.current_wave()) * u.angstrom)
        except ValueError:
            for wv in self.params.all_wavelengths:
                if wv in fits_path:
                    self.wave = wv
                    self.params.cmap = self.cmap = aia_color_table(int(self.wave) * u.angstrom)
                    self.params.current_wave(int(self.wave))
                    break

        if isinstance(in_name, (int, str)) and str(in_name).isdigit():
            in_name = int(in_name)
            self.params.png_frame_name = self.find_frames_at_path(self.fits_path)[in_name]

        if "all" in str(self.params.png_frame_name):
            self.params.png_frame_name = self.find_frames_at_path(self.fits_path)

        if target in self.params.png_frame_name:
            self.params.png_frame_name.append("mgn_" + target)
            self.params.double_rhe_flag = True

        if isinstance(self.params.png_frame_name, list) and len(self.params.png_frame_name):
            for name in self.params.png_frame_name:
                self.current_frame = name
                self.wave = self.params.current_wave()
                self.init_frame(self.fits_path, self.current_frame)
                out = self.render_all(reference=False)
        else:
            self.init_frame(self.fits_path)
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
        short_path = os.path.join(shorts_folder, "{}_{}.png".format(self.params.current_wave(), timestamp.split(".")[0]))
        os.makedirs(shorts_folder, exist_ok=True)
        shutil.copyfile(cat_png_path, os.path.normpath(short_path), follow_symlinks=True)

    def plot_aia_orig(self):
        self.frame_name = "compressed_image"
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)
        self.params.raw_image = self.frame = np.flipud(frame)
        self.out_path = self.get_orig_path(mod="orig")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.do_save()

    def plot_aia_log(self):
        self.frame_name = self.params.master_frame_list_oldest
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)

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

        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, frame_name)
        self.frame = np.flipud(frame)
        if self.frame.max() > 2 or self.frame.min() < 0:
            self.frame = self.normalize(self.frame)

        upsilon = self.params.upsilon
        if upsilon:
            self.params.upsilon_high = upsilon[1] if len(upsilon) > 1 else upsilon
            self.params.upsilon_low = upsilon[0] if len(upsilon) > 1 else upsilon
            self.frame = norm_stretch(self.frame, upsilon=self.params.upsilon_low, upsilon_high=self.params.upsilon_high)

        self.out_path = self.get_changed_path()
        self.out_path = self.out_path.replace("AIAsynoptic", "DrGilly_").replace(".png", f"_{self.frame_name}.png")

        self.do_save(do_small=True if "MultiImage".casefold() in self.filt_name.casefold() else False)
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

        self.params.rbg_image = []
        self.params.rbg_labels = ["hq"]
        self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
        frames = [out]

        if do_small:
            frames.extend([block_reduce(out, 2, np.nanmean), block_reduce(out, 4, np.nanmean)])
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

    def label_plot(self, img_in=None):
        if img_in is None:
            img = self.params.rbg_image[0]
        else:
            img = img_in + 0
        if self.image_data is None:
            self.init_rainbow_frame()
        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(time_string_raw, targetZone="US/Mountain", out_fmt="%m-%d-%Y %I:%M%p %Z")
        time_list = time_string.split()
        clock, day, year = time_list[1].lower(), time_list[0][:-5], time_list[0][-4:]

        inst, wave = "AIA", self.clean_name_string(full_name)[1]
        fname, frame_name = self.frame_name.casefold(), self.frame_name
        name, prev = fname.split("(")[0], fname.split("(")[1][:-1] if "(" in fname else "-"

        rez = img.shape[0]
        scale, h, wid_of_char = (6, 120, 60) if rez == 4096 else (3, 60, 30) if rez == 2048 else (1.5, 30, 15)
        h0, thickness = (100, 4) if rez == 4096 else (50, 2) if rez == 2048 else (25, 2)

        positions = [(rez - wid_of_char * len(text) - 10, height) for text, height in zip([name, prev, inst, wave], [h0, h0 + h, h0 + 2 * h, h0 + 3 * h])]
        for text, (x, y) in zip([name, prev, inst, wave], positions):
            cv2.putText(img, text, (x, y), 1, scale, (255, 255, 255), thickness)

        positions = [(0, height) for height in [h0, h0 + h, h0 + 2 * h, h0 + 3 * h]]
        for text, (x, y) in zip([clock, day, year, "MT"], positions):
            cv2.putText(img, text, (x, y), 1, scale, (255, 255, 255), thickness)

        try:
            aH = self.params.upsilon_high
            aL = self.params.upsilon_low
            cv2.putText(img, f"aH: {aH}", (0, int(0.97 * rez)), 1, scale, (255, 255, 255), thickness)
            cv2.putText(img, f"aL: {aL}", (0, int(0.99 * rez)), 1, scale, (255, 255, 255), thickness)
        except (SystemError, ValueError) as e:
            print(238, e)

        return img

    def draw_reticle(self, img):
        cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])), int(self.params.header["R_SUN"]), (255, 255, 255), 3)
        cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])), 10, (255, 0, 0), 10)

    # def cleanup(self):
    #     try:
    #         print("Writing Video...", end="")
    #         radial_hist_path = "analysis\\radial_hist_post"
    #         hist_path_0 = os.path.join(self.params.base_directory(), radial_hist_path)
    #         hist_path_1 = hist_path_0[:-5]

    #         if len(os.listdir(hist_path_0)):
    #             self.write_video_in_directory(directory=hist_path_0, fps=15, destroy=False)
    #         if len(os.listdir(hist_path_1)):
    #             self.write_video_in_directory(directory=hist_path_1, fps=15, destroy=False)
    #         if self.params.do_cat:
    #             self.write_video_in_directory(directory=self.params.cat_directory, file_name="concatinated.avi", fps=15, destroy=False)
    #         print("Success!")
    #     except (FileNotFoundError, AttributeError) as e:
    #         print("ImageProcessorCV")
    #         raise e
    #     super().cleanup()

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

        frame = self.frame_touchup(frame_name, frame)

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

    def finalize_and_save_plots(self, dpi=200):
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
        # self.ii += 1
        self.confirm_fits_file(fits_path)
        self.do_fits_function(fits_path)
        return False

    def do_fits_function(self, fits_path, in_name=None, doBar=False):
        """Main Call on the Fits Path"""
        self.init_frame_from_fits(fits_path)

        self.init_radius_array()
        # self.init_bin_array()
        self.init_statistics()

        self.init_quad_figure()

        self.collect_frames(fits_path, doBar)

        # self.finalize_and_save_plots()
        # self.reinit_constants()

        # self.open_folder(self.main_save_path)
        return None

    def collect_frames(self, fits_path, doBar=False):
        # with open(fits_path) as fp:
        # with fits.open(fits_path, cache=False, reprocess_mode="update") as hdul:
        #     self.hdu_name_list = self.list_hdus(hdul)
        self.good_frames = self.find_frames_at_path(fits_path)

        print(self.good_frames)

        self.max_width = np.max([len(x) for x in self.good_frames])
        iterable = tqdm(self.good_frames, desc="") if doBar else self.good_frames

        images, names = [], []
        for frame_name in iterable:
            if "rht" in frame_name or "final" in frame_name:
                continue
            frame, wave1, t_rec1, _, _, mod_name = self.load_this_fits_frame(
                fits_path, frame_name
            )
            if False:
                frame = self.resize_image(frame, prnt=False)

            if "comp" in frame_name:
                images.append(frame), names.append("lev1p5")
                images.append(np.power(frame, 0.65)), names.append("gamma corrected")
            elif "nrgf" in frame_name:
                images.insert(2, frame), names.insert(2, frame_name)
            elif "msgn" in frame_name:
                images.insert(3, frame), names.insert(3, frame_name)
            else:
                images.append(frame), names.append(frame_name)

            if "rhe" in frame_name:
                (
                    images.append(norm_stretch(frame)),
                    names.append(f"upsilon({frame_name})"),
                )

        print(names)

        # self.do_compare_histogramplot_images(images, names)

        # self.do_compare_histogramplot_rheonly(
        #     images, names, target_names=["lev1p5", "rhef", "upsilon(rhef)"]
        # )
        self.do_compare_histogramplot(images, names)

        # for frame_name in iterable:

        # self.handle_one_frame(fits_path, frame_name, doBar, iterable)
        # self.count_frames += 1

        if doBar:
            iterable.set_description(" *    Plots Complete", refresh=True)

    # def handle_one_frame(self, fits_path, frame_name, doBar, iterable):
    #     if doBar: iterable.set_description(" *     Plotting {}".format(frame_name.ljust(self.max_width)), refresh=True)
    #     frame1, wave1, t_rec1, center1, int_time, name = self.load_this_fits_frame(fits_path, frame_name)
    #     # frame1[self.vignette_mask] = np.nan
    #     self.add_to_histplot(name, frame1)

    # def get_one_fits_frame(self, fits_path=None, in_name=-1):
    #
    #     frame, wave1, t_rec1, _, _, mod_name = self.load_this_fits_frame(fits_path, in_name)
    #
    #     if True:
    #         self.params.modified_image = self.resize_image(self.params.modified_image, prnt=False)
    #         self.params.raw_image = self.resize_image(self.params.raw_image, prnt=False)
    #
    #
    #     # self.peek_frames()
    #     self.image_data = str(wave1), fits_path, t_rec1, self.params.modified_image.shape
    #     self.params.make_file_paths(self.image_data)
    #     self.name, self.wave = self.clean_name_string(str(wave1))
    #     return self.params.modified_image, self.name

    def init_frame_from_fits(self, fits_path=None, in_name=-1):
        """Load the fits file from disk and get a in_name or two"""

        self.fits_path = fits_path or self.fits_path
        self.params.fits_path = self.fits_path
        # self.load()
        if self.params.raw_image is None:
            self.params.raw_image, _, _, _, _, self.raw_name = (
                self.load_this_fits_frame(
                    fits_path, self.params.master_frame_list_oldest
                )
            )

        self.params.modified_image, wave1, t_rec1, _, _, self.mod_name = (
            self.load_this_fits_frame(fits_path, in_name)
        )

        if False:
            self.params.modified_image = self.resize_image(
                self.params.modified_image, prnt=False
            )
            self.params.raw_image = self.resize_image(self.params.raw_image, prnt=False)

        # self.peek_frames()
        self.image_data = (
            str(wave1),
            fits_path,
            t_rec1,
            self.params.modified_image.shape,
        )
        self.params.make_file_paths(self.image_data)
        self.name, self.wave = self.clean_name_string(str(wave1))
        return self.params.modified_image, self.name

    def open_folder(self, path):
        import webbrowser

        webbrowser.open("file:///" + path)

    def init_quad_figure(self, use_cmap=True):
        pass

        # if use_cmap:
        #     self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
        # else:
        #     from matplotlib import cm
        #     self.params.cmap = cm.gray
        #
        # # try:
        # #     lev1p5_mask = ['lev1p5' in x for x in self.good_frames]
        # #     lev1p5_loc = np.where(lev1p5_mask)[0][0]
        # #     repeat_frame = lev1p5_loc
        # # except IndexError as e:
        # #     print('\r' + str(e))
        # #     repeat_frame = 0
        # #
        # # self.good_frames.insert(repeat_frame, self.good_frames[repeat_frame])
        # # self.good_frames.pop(0)
        # self.params.doing_jpeg = False
        # if self.params.doing_jpeg:
        #     self.good_frames.insert(0, "jpeg")
        #
        # self.good_frame_stems = [x.split('(')[0] for x in self.good_frames]
        #
        # self.n_plots = len(self.good_frames)
        # self.n_rows = 2 if self.n_plots > 2 else 1
        # self.n_cols = max((self.n_plots // self.n_rows, 1)) if self.n_plots > 2 else 2
        # self.n_slots = self.n_rows * self.n_cols
        # while self.n_slots < self.n_plots:
        #     self.n_cols += 1
        #     self.n_slots = self.n_rows * self.n_cols
        #
        # self.fig, self.axArray = plt.subplots(self.n_rows, self.n_cols, sharex="all", sharey="all")
        #
        # try:
        #     t_rec = self.header["T_REC"]
        # except KeyError as e:
        #     t_rec = self.header["T_OBS"]
        #
        # self.fig.suptitle("{}  at  {}".format(self.wave, t_rec))
        # self.axArray = self.axArray.flatten()
        #
        # blank = np.zeros_like(self.params.modified_image)
        #
        # for ax in self.axArray:
        #     ax.imshow(blank, interpolation="None")
        #     ax.set_title(" ")

    # def add_to_histplot(self, frame_name_in, frame):
    #     # print("\r * Adding Plot  {}".format(frame_name_in))
    #     # if 'primary' in frame_name_in:
    #     #     suffix = "_orig"
    #     # else:
    #     suffix = ""
    #     frame_name = frame_name_in + suffix
    #     self.last_frame_name = frame_name_in
    #
    #     frame = self.frame_touchup(frame_name, frame)
    #
    #
    #     vmin = None if self.dont_vminmax else 0.
    #     vmax = None if self.dont_vminmax else 1.
    #     self.axArray[self.count_frames].imshow(frame, origin="lower", vmin=vmin, vmax=vmax,
    #                                            cmap=self.params.cmap, interpolation="None")
    #
    #     self.axArray[self.count_frames].set_title(frame_name)
    #     self.axArray[self.count_frames].patch.set_alpha(0)
    #
    #     frame = self.vignette(frame)
    #     self.frame_names.append(frame_name)
    #     self.frames.append(frame)

    def finalize_and_save_plots(self, dpi=200):
        pass
        # inches = 4
        # colWid = self.n_cols * inches
        # rowWid = self.n_rows * inches
        #
        # self.fig.set_size_inches(w=colWid, h=rowWid)
        # plt.tight_layout()
        #
        # save_path = os.path.join(self.params.imgs_top_directory(), "compare", "{:04}_compare.png".format(int(self.wave)))
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.main_save_path = save_path
        # self.fig.savefig(save_path, dpi=dpi)
        # # plt.show(block=True)
        # if False:
        #     self.plot_zooms()

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

    #
    # def image_is_plottable(self, frame_name):
    #     # return True
    #     return self.doesnt_have_wrong_string(frame_name)
    #     return self.does_have_right_string(frame_name)
    #
    #
    # def does_have_right_string(self, frame_name, right_string=None):
    #
    #     right_string = right_string or ["lev1p5(t_int)", "final(rhe)", "rht(lev1p5)", "rht(final)"]
    #
    #     for goods in right_string:
    #         if frame_name.casefold() == goods:
    #             return True
    #     return False
    #
    #
    # def doesnt_have_wrong_string(self, frame_name, wrong_string=None):
    #     bads = wrong_string or ["lev1p0", "t_int(lev1p0)", "t_int(primary)", "lev1p5(lev1p0)", "compressed_image",
    #                             "final(rhe)"]
    #     if True:
    #         bads.append("primary")
    #         bads.append("lev1p5")
    #
    #     if self.params.multiplot_all:
    #         bads = []
    #
    #     for nam in bads:
    #         # if nam in frame_name:
    #         if nam.casefold() == frame_name:
    #             return False
    #     return True
    #
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

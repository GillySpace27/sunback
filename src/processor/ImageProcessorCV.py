import os
import shutil
import time

import cv2
from tqdm import tqdm

from processor.ImageProcessor import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from science.color_tables import aia_color_table
from utils.stretch_intensity_module import norm_stretch


class ImageProcessorCV(ImageProcessor):
    filt_name = 'CV Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.shrink_factor = 1
        self.frame_name = None
        self.img_frame = None
        self.out_path = None
        self.in_name = -1
    
    def do_fits_function(self, fits_path, in_name=None):
        """ Main Call on the Fits Path """
        self.init_frame(fits_path, self.params.png_frame_name)
        self.render_all()
        return self
    
    def do_img_function(self):
        """ Main Call on the Fits Path """
        if False:
            self.plot_two()
            self.plot_two("Less Zoomed", True)
            # self.display_all()
        
        # self.init_image_frame()
        raise NotImplementedError
    
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
        """Render one image_path"""
        
        if reference:
            self.plot_aia_raw()
            self.plot_aia_log()
        
        self.plot_aia_changed()
        # self.save_concatinated()
        
        # self.do_shortcut()
    
    def do_shortcut(self):
        cat_png_path = self.cat_path
        root_folder = os.path.dirname(self.params.base_directory())
        fits_folder = os.path.dirname(self.params.use_image_path())
        cat_png_filename = os.path.basename(cat_png_path)
        shorts_folder = os.path.join(root_folder, "shorts")
        # short_path = os.path.join(shorts_folder, cat_png_filename.replace(".png", ".lnk"))
        
        timestamp = self.image_data[2]
        short_path = os.path.join(shorts_folder, "{}_{}.png".format(self.params.current_wave(), timestamp.split('.')[0]))
        os.makedirs(shorts_folder, exist_ok=True)
        
        src_file = cat_png_path
        dest_file = os.path.normpath(short_path)
        shutil.copyfile(src_file, dest_file, follow_symlinks=True)
        # self.make_shortcut(src_file,dest_file , False)
    
    def plot_aia_raw(self):
        """Plot the raw_image data from AIA"""
        # Get the Frame and Path
        # self.frame_name = "t_int"
        self.frame_name = self.params.master_frame_list_oldest
        
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)
        # frame = np.log10(frame)
        # frame = frame / np.nanpercentile(frame, 50)/2
        self.frame = np.flipud(frame)
        # self.frame = np.flipud(self.params.raw_image)
        self.out_path = self.get_raw_path()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.vignette()
        self.prep_save()
        self.img_save(self.out_path)
    
    def plot_aia_log(self):
        """Plot the raw_image data from AIA"""
        # Get the Frame and Path
        # self.frame_name = "t_int"
        self.frame_name = self.params.master_frame_list_newest
        
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)
        frame = np.log10(frame)
        frame = frame / np.nanpercentile(frame, 50) / 2
        self.frame = np.flipud(frame)
        # self.frame = np.flipud(self.params.raw_image)
        self.out_path = self.get_raw_path(mod='log')
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.vignette()
        self.prep_save()
        self.img_save(self.out_path)
    
    def init_radius_array(self, vignette_radius=1.19, s_radius=400, t_factor=1.28, force=False):
        """Build an r-coordinate array of shape(in_object)"""
        if self.params.modified_image is None:
            self.params.modified_image = np.zeros_like(self.params.raw_image)
        
        self.params.rez = self.header["NAXIS1"]
        self.found_limb_radius = self.fit_limb_radius = self.header["R_SUN"]
        self.params.center = (self.header["X0_MP"], self.header["Y0_MP"])
        
        nn = 1
        while self.found_limb_radius > self.params.rez / 2:
            nn *= 2
            self.found_limb_radius = self.fit_limb_radius = self.header["R_SUN"] / nn
            self.params.center = [self.header["X0_MP"] / nn, self.header["Y0_MP"] / nn]
        
        self.shrink_factor = nn
        self.output_abscissa = np.arange(self.params.rez)
        
        xx, yy = np.meshgrid(np.arange(self.params.rez), np.arange(self.params.rez))
        xc, yc = xx - self.params.center[0], yy - self.params.center[1]
        self.radius = np.sqrt(xc * xc + yc * yc)
        self.rad_flat = self.radius.flatten()
        self.vcut = int(vignette_radius * self.params.rez // 2)
        self.vrad = self.n2r(self.vcut)
        self.vignette_mask = np.asarray(self.radius > self.vcut, dtype=bool)
    
    # def plot_aia_changed(self):
    #     """Plot the modified_image data from AIA"""
    #     # Get the Frame and Path
    #     self.frame_name = self.params.png_frame_name #.hdu_name_list[-1]
    #     self.frame = np.flipud(self.params.modified_image)
    #     self.out_path = self.get_changed_path()
    #     out_dir = os.path.dirname(self.out_path)
    #     os.makedirs(out_dir, exist_ok=True)
    #     print("Saving to {}".format(self.out_path))
    #     self.vignette()
    #
    #     self.prep_save()
    #     self.img_save(self.out_path)
    #
    def plot_aia_changed(self):
        """Plot the raw_image data from AIA"""
        # Get the Frame and Path
        # self.frame_name = "t_int"
        self.frame_name = self.params.png_frame_name  # .hdu_name_list[-1]
        
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)
        
        self.frame = np.flipud(frame)
        
        self.out_path = self.get_changed_path()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        # print("Saving to {}".format(self.out_path))
        
        self.vignette()
        self.prep_save()
        self.img_save(self.out_path)
    
    # def plot_aia_changed(self):
    #     """Plot the modified_image data from AIA"""
    #     # Get the Frame and Path
    #     self.frame_name = self.params.png_frame_name #.hdu_name_list[-1]
    #     self.frame = np.flipud(self.params.modified_image)
    #     self.out_path = self.get_changed_path()
    #     out_dir = os.path.dirname(self.out_path)
    #     os.makedirs(out_dir, exist_ok=True)
    #     print("Saving to {}".format(self.out_path))
    #     self.vignette()
    #
    #     self.prep_save()
    #     self.img_save(self.out_path)
    
    def prep_save(self):
        self.make_image()
        self.label_plot()
        self.path_box.append(self.out_path)
    
    def make_image(self):
        out = self.frame + 0
        maxmax = np.nanpercentile(out, 99)
        minmin = np.nanpercentile(out, 1)
        themax = np.nanmax(out)
        
        if themax > 100 or themax < 0.8:
            out = (self.frame - minmin) / (maxmax - minmin)
            # print("\nRenormalizing", maxmax, minmin, np.max(out), np.min(out))
        
        self.img_frame = (self.params.cmap(out)[:, :, :3] * 255).astype(np.uint8)
        b, g, r = cv2.split(self.img_frame)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        self.params.rbg_image = rgb_img
        self.vignette()
    
    def img_save(self, path, save=True):
        if save:
            cv2.imwrite(path, self.params.rbg_image)
        else:
            # cv2.imshow(mat=self.params.rbg_image)
            plt.imshow(self.params.rbg_image)
            plt.show()
    
    def label_plot(self):
        """Annotate with Text"""
        # img = self.img_frame
        img = self.params.rbg_image
        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(time_string_raw)
        time_list = time_string.split()
        
        inst = 'AIA'
        _, wave = self.clean_name_string(full_name)
        clock = time_list[1].lower()
        day = time_list[0][:-5]
        year = time_list[0][-4:]
        
        x0 = 3900
        x1 = 3875
        scale = 3 if self.shrink_factor == 1 else 2 if self.shrink_factor == 2 else 1
        h1 = 100
        h2 = 200
        h3 = 300
        if shape[0] < 3000:
            x0 = x0 // 4
            x1 = x1 // 4
            # scale = 1
            h1 = 40
            h2 = 80
            h3 = 120
        scale2 = scale
        # if self.params.alpha is not None:
        #     cv2.putText(img, "a={:0.3f}".format(self.params.alpha), (int(x0*0.95), h3), 0,   scale, (255, 255, 255), 3)
        
        if type(self.frame_name) is list:
            frame_name = [x for x in self.frame_name if x.casefold() in self.hdu_name_list][0]
        else:
            frame_name = self.frame_name
        
        cv2.putText(img, frame_name.casefold(), (int(x0 * 0.92), h1), 0, scale, (255, 255, 255), scale2)
        
        reticle = False
        if reticle:
            cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                       int(self.params.header["R_SUN"]), (255, 255, 255), 3)
            cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                       int(10), (255, 0, 0), 10)
        
        cv2.putText(img, inst, (x0, h2), 0, scale, (255, 255, 255), scale2)
        cv2.putText(img, wave, (x1, h3), 0, scale, (255, 255, 255), scale2)
        cv2.putText(img, clock, (0, h1), 0, scale, (255, 255, 255), scale2)
        cv2.putText(img, day, (0, h2), 0, scale, (255, 255, 255), scale2)
        cv2.putText(img, year, (0, h3), 0, scale, (255, 255, 255), scale2)
    
    def cleanup(self):
        # self.make_intermediate_videos()
        pass
    
    def make_intermediate_videos(self):
        try:
            print("Writing Video...", end='')
            radial_hist_path = "analysis\\radial_hist_post"
            hist_path_0 = os.path.join(self.params.base_directory(), radial_hist_path)
            hist_path_1 = hist_path_0[:-5]
            
            n_hist_0 = len(os.listdir(hist_path_0))
            n_hist_1 = len(os.listdir(hist_path_1))
            
            if n_hist_0:
                self.write_video_in_directory(directory=hist_path_0, fps=15, destroy=False)
            if n_hist_1:
                self.write_video_in_directory(directory=hist_path_1, fps=15, destroy=False)
            if self.params.do_cat:
                self.write_video_in_directory(directory=self.params.cat_directory, file_name="concatinated.avi", fps=15, destroy=False)
            print("Success!")
        except (FileNotFoundError, AttributeError) as e:
            print("ImageProcessorCV")
            raise (e)
        
        # destroy = False
        # if destroy:
        #     shutil.rmtree(self.orig_directory)
    
    @staticmethod
    def peek_frame(img):
        shrink = 5
        cv2.imshow("win2", img[::shrink, ::shrink, ::shrink])
        cv2.waitKey(0)


class MultiImageProcessorCv(ImageProcessorCV):
    filt_name = 'MultiImage Plotter'
    description = "Look at the different methods compared"
    progress_verb = "Writing"
    progress_unit = "Images"
    
    # list_of_inputs = ["lev1p5", "t_int", "lev1p0"]
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.max_width = 20
        self.main_save_path = None
        self.last_frame_name = None
        self.base_image = None
        self.n_plots = None
        self.n_cols = None
        self.n_rows = None
        self.init = False
        self.count = 1
        self.frame_names = []
        self.frames = []
        self.good_frames = []
        self.good_frame_stems = []
        self.fig = None
    
    def do_fits_function(self, fits_path, in_name=None, doBar=False):
        """ Main Call on the Fits Path """
        self.tic()
        self.init_frame(fits_path)
        self.init_plot()
        self.init_radius_array()
        
        self.max_width = np.max([len(x) for x in self.good_frames])
        iterable = tqdm(self.good_frames, desc="") if doBar else self.good_frames
        for self.count, frame_name in enumerate(iterable):
            if frame_name == "jpeg":
                self.plot_jpeg(fits_path, frame_name, doBar, iterable)
                
            else:
                self.handle_one_frame(fits_path, frame_name, doBar, iterable)
        if doBar: iterable.set_description(" *    Plots Complete", refresh=True)
        
        self.finalize_and_save_plots()
        self.reinit_constants()
        self.toc()
        self.open_folder(self.main_save_path)
        return None
    
    def plot_jpeg(self, fits_path, frame_name, doBar, iterable):
        import PIL
        from PIL import Image
        j_directory = os.path.join(self.params.imgs_top_directory(), "jpeg")
        paths = os.listdir(j_directory)
        full_paths = [os.path.join(j_directory, pat) for pat in paths]
        wavenum = int(''.join(i for i in fits_path if i.isdigit()))
        correct = [x for x in full_paths if str(wavenum) in x][0]
        image = Image.open(correct)
        # from astropy.nddata import block_reduce
        # image = block_reduce(image, self.shrink_factor/2)
        # image=  image.rotate(180)
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        ax = self.axArray[0]
        ax.imshow(image, origin='lower', interpolation="None")
        ax.set_title('"The Sun Today"')
        
        return
        
    
    def handle_one_frame(self, fits_path, frame_name, doBar, iterable):
        if doBar: iterable.set_description(" *     Plotting {}".format(frame_name.ljust(self.max_width)), refresh=True)
        frame1, wave1, t_rec1, center1, int_time, name = self.load_this_fits_frame(fits_path, frame_name)
        frame1[self.vignette_mask] = np.nan
        self.add_to_plot(name, frame1)
    
    def init_frame(self, fits_path=None, in_name=-1):
        """Load the fits file from disk and get a in_name or two"""
        
        self.fits_path = fits_path or self.fits_path
        self.params.fits_path = self.fits_path
        
        self.params.raw_image, _, _, _, _, self.raw_name = \
            self.load_this_fits_frame(fits_path, self.params.master_frame_list_newest)
        
        self.params.modified_image, wave1, t_rec1, _, _, self.mod_name = \
            self.load_this_fits_frame(fits_path, in_name)
        
        # self.peek_frames()
        self.image_data = str(wave1), fits_path, t_rec1, self.params.modified_image.shape
        self.params.make_file_paths(self.image_data)
        self.name, self.wave = self.clean_name_string(str(wave1))
    
    def open_folder(self, path):
        import webbrowser
        webbrowser.open('file:///' + path)
    
    def init_plot(self):
        self.good_frames = [x for x in self.hdu_name_list if ("lev1_" not in x)]  # TODO I removed a [1:] here, not sure what it was doing...
        
        use_cmap = True
        if use_cmap:
            self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
        else:
            from matplotlib import cm
            self.params.cmap = cm.gray
        
        # try:
        #     lev1p5_mask = ['lev1p5' in x for x in self.good_frames]
        #     lev1p5_loc = np.where(lev1p5_mask)[0][0]
        #     repeat_frame = lev1p5_loc
        # except IndexError as e:
        #     print('\r' + str(e))
        #     repeat_frame = 0
        #
        # self.good_frames.insert(repeat_frame, self.good_frames[repeat_frame])
        self.good_frames.pop(0)
        self.good_frames.insert(0, "jpeg")
        
        self.good_frame_stems = [x.split('(')[0] for x in self.good_frames]
        
        self.n_plots = len(self.good_frames)
        self.n_rows = 2
        self.n_cols = max((self.n_plots // self.n_rows, 1))
        self.n_slots = self.n_rows * self.n_cols
        while self.n_slots < self.n_plots:
            self.n_cols += 1
            self.n_slots = self.n_rows * self.n_cols
        
        self.fig, self.axArray = plt.subplots(self.n_rows, self.n_cols, sharex="all", sharey="all")
        
        try:
            t_rec = self.header["T_REC"]
        except KeyError as e:
            t_rec = self.header["T_OBS"]
        
        self.fig.suptitle("{}  at  {}".format(self.wave, t_rec))
        self.axArray = self.axArray.flatten()
        
        self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
        blank = np.zeros_like(self.params.raw_image)
        
        for ax in self.axArray:
            ax.imshow(blank, interpolation="None")
            ax.set_title(" ")
    
    def add_to_plot(self, frame_name_in, frame):
        # print("\r * Adding Plot  {}".format(frame_name_in))
        if 'primary' in frame_name_in:
            suffix = "_mod"
        else:
            suffix = ""
        frame_name = frame_name_in + suffix
        self.last_frame_name = frame_name_in
        frame, dont = self.frame_touchup(frame_name, frame)
        vmin = None if dont else 0.
        vmax = None if dont else 1.
        self.axArray[self.count].imshow(frame, origin="lower", vmin=vmin, vmax=vmax,
                                        cmap=self.params.cmap, interpolation="None")
        self.axArray[self.count].set_title(frame_name)
        self.frame_names.append(frame_name)
        self.frames.append(frame)
    
    def finalize_and_save_plots(self, dpi=500):
        
        inches = 4
        colWid = self.n_cols * inches
        rowWid = self.n_rows * inches
        
        self.fig.set_size_inches(w=colWid, h=rowWid)
        plt.tight_layout()
        
        save_path = os.path.join(self.params.imgs_top_directory(), "{}_compare.png".format(self.wave))
        self.main_save_path = save_path
        self.fig.savefig(save_path, dpi=dpi)
        if False:
            # plt.show(block=True)
            self.plot_zooms()
    
    def plot_zooms(self, dpi=500):
        zooms = os.path.join(self.params.imgs_top_directory(), "zooms")
        os.makedirs(zooms, exist_ok=True)
        
        save_path = os.path.join(zooms, "{}_compare.png".format(self.wave))
        self.fig.savefig(save_path, dpi=dpi)
        
        save_path = os.path.join(zooms, "1_zoom_{}_compare.png".format(self.wave))
        plt.xlim((3250 / self.shrink_factor, 4000 / self.shrink_factor))
        plt.ylim((2250 / self.shrink_factor, 3000 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)
        
        save_path = os.path.join(zooms, "2_zoom_{}_compare.png".format(self.wave))
        plt.xlim((2404 / self.shrink_factor, 3500 / self.shrink_factor))
        plt.ylim((3000 / self.shrink_factor, 4096 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)
        
        # plt.close(self.fig)
        # print("Done - Files Saved in {}".format(self.params.imgs_top_directory()))
    
    def reinit_constants(self):
        self.count = 1
        self.last_frame_name = None
        plt.close(self.fig)
    
    def frame_touchup(self, frame_name, frame):
    
        short_circuit = False
        if short_circuit:
            return frame, True
    
        # Frame Cleanup
        frame = frame.astype(np.float32)
        frame[~np.isfinite(frame)] = np.nan
        
        
        if "primary" in frame_name:
            frame *= 2
    
        basic_scrunch = True
        if basic_scrunch:
            frame = self.scrunch(frame)
    
        ## Perform Nonlinear Transforms
        # Power
        for name in ["_mod", "nrgf"]:  # , "int_enhance"]:
            if name in frame_name:
                frame = self.power_mod(frame)
    
        # Maxima Stretching
        do_maxima_scrunch = True
        if do_maxima_scrunch:
            frame = self.maxima_scrunch(frame)
            
        # Norm Stretching
        stretch = True
        if stretch:
            if "qrn" in frame_name:
                frame = norm_stretch(frame)
        
        dont_vminmax = False
        for name in ["RHT"]:
            if name in frame_name:
                dont_vminmax = True

    
        return frame, dont_vminmax
    # if frame_name == "nrgf":
    #     # Replace the Disk
    #     self.init_radius_array()
    #     mask = self.radius < self.found_limb_radius*0.5
    #     frame[mask] = 0.5 #self.base_image[mask]
    
    # darken_rfilt = 1.2
    # darken_quant = 1.1
    #
    # if frame_name == "int_enhance":
    #     # Save the Disk
    #     # self.base_image = frame
    #     # frame = np.sqrt(frame)
    #     minx = np.nanpercentile(frame, 0.1)
    #     maxx = np.nanpercentile(frame, 99.9)
    #     frame = (frame - minx) / (maxx - minx)
    #     frame /= darken_rfilt
    #
    # if frame_name == "quantile":
    #     frame /= darken_quant
    
    # self.vignette_mask = np.asarray(self.radius > self.vcut, dtype=bool)
    # frame[self.vignette_mask] = np.nan
    
    def power_mod(self, frame):
        frame *= 10.
        pow = 1/4
        np.power(frame, pow, out=frame)
        frame *= pow
        return frame
    
    def scrunch(self, frame, n_exclude=50):
        # lowlow = np.nanmin(frame)
        # highigh = np.nanmax(frame)
        
        total = self.params.rez ** 2
        perc = n_exclude / total
    
        low = np.nanpercentile(frame, perc)
        high = np.nanpercentile(frame, 100-perc)
    
        frame = self.norm_formula(frame, low, high)
        return frame
    
    def maxima_scrunch(self, frame, num=1.0):
        mask = frame > num
        frame[mask] = np.nan
        frame = self.scrunch(frame)
        frame[mask] = num
        return frame
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


class ImageProcessorCV(ImageProcessor):
    filt_name = 'CV Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
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
        self.frame_name = ["lev1p5(t_int)", "lev1p5(L)", "t_int(lev1p0)", "lev1p0"]
        
        frame, wave, t_rec, center, int_time = self.load_a_fits_field(self.fits_path, self.frame_name)
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
        self.frame_name = ["lev1p5(t_int)", "lev1p5(L)", "t_int(lev1p0)", "lev1p0"]
        
        frame, wave, t_rec, center, int_time = self.load_a_fits_field(self.fits_path, self.frame_name)
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
        
        self.params.center = [self.header["X0_MP"], self.header["Y0_MP"]]
        self.found_limb_radius = self.fit_limb_radius = self.header["R_SUN"]
        self.params.rez = self.header["NAXIS1"]
        
        self.output_abscissa = np.arange(self.params.rez)
        
        xx, yy = np.meshgrid(np.arange(self.params.rez), np.arange(self.params.rez))
        xc, yc = xx - self.params.center[0], yy - self.params.center[1]
        self.radius = np.sqrt(xc * xc + yc * yc)
        self.rad_flat = self.radius.flatten()
        
        self.binfactor = binfactor = 2
        self.binInds = np.asarray(binfactor * np.floor(self.rad_flat // binfactor), dtype=np.int32)
        # self.make_annular_rings()
        # self.binInds = np.digitize(self.rad_flat, self.RN)
        
        self.binXX = xx.flatten()
        self.binYY = yy.flatten()
        self.binII = np.arange(len(self.rad_flat))
        self.vcut = int(vignette_radius * self.params.rez // 2)
        self.vrad = self.n2r(self.vcut)
        self.vignette_mask = np.asarray(self.radius > self.vcut, dtype=bool)
        self.s_radius = s_radius
        self.tRadius = self.s_radius * t_factor
    
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
        # self.frame_name = "t_int(lev1p0)"
        self.frame_name = self.params.png_frame_name  # .hdu_name_list[-1]
        
        frame, wave, t_rec, center, int_time = self.load_a_fits_field(self.fits_path, self.frame_name)
        
        self.frame = np.flipud(frame)
        
        self.out_path = self.get_changed_path()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        print("Saving to {}".format(self.out_path))
        
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
            print("\nRenormalizing", maxmax, minmin, np.max(out), np.min(out))
        
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
        scale = 3
        h1 = 100
        h2 = 200
        h3 = 300
        if shape[0] < 3000:
            x0 = x0 // 4
            x1 = x1 // 4
            scale = 1
            h1 = 40
            h2 = 80
            h3 = 120
        
        # if self.params.alpha is not None:
        #     cv2.putText(img, "a={:0.3f}".format(self.params.alpha), (int(x0*0.95), h3), 0,   scale, (255, 255, 255), 3)
        
        if type(self.frame_name) is list:
            frame_name = [x for x in self.frame_name if x.casefold() in self.hdu_name_list][0]
        else:
            frame_name = self.frame_name
        
        cv2.putText(img, frame_name.casefold(), (int(x0 * 0.92), h1), 0, scale, (255, 255, 255), 3)
        
        reticle = False
        if reticle:
            cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                       int(self.params.header["R_SUN"]), (255, 255, 255), 3)
            cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                       int(10), (255, 0, 0), 10)
        
        cv2.putText(img, inst, (x0, h2), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, wave, (x1, h3), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, clock, (0, h1), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, day, (0, h2), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, year, (0, h3), 0, scale, (255, 255, 255), 3)
    
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
    filt_name = 'MultiImageProcessorCv'
    description = "Look at the different methods compared"
    progress_verb = "Writing"
    progress_unit = "Images"
    # list_of_inputs = ["lev1p5(t_int)", "lev1p5(L)", "t_int(lev1p0)", "lev1p0"]
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.main_save_path = None
        self.last_frame_name = None
        self.base_image = None
        self.n_plots = None
        self.n_cols = None
        self.n_rows = None
        self.init = False
        self.count = 0
        self.frame_names = []
        self.frames = []
        self.good_frames = []
        self.fig = None
    
    def do_fits_function(self, fits_path, in_name=None):
        """ Main Call on the Fits Path """
        self.tic()
        self.init_frame(fits_path, None)
        self.wave = self.params.current_wave()
        self.init_plot()
        # print("\r ** Beginning to Plot...")
        pbar = tqdm(self.good_frames, desc="")
        max_width = np.max([len(x) for x in self.good_frames])
        for ii, frame_name in enumerate(pbar):
            pbar.set_description(" *     Plotting {}".format(frame_name.ljust(max_width)), refresh=True)
            frame1, wave1, t_rec1, center1, int_time = self.load_a_fits_field(fits_path, frame_name)
            self.add_to_plot(frame_name, frame1)
            last = ii == len(self.good_frames)-1
            if last:
                pbar.set_description(" *    Plots Complete", refresh=True)
        self.finalize_and_save_plots()
        self.toc()
        self.open_folder(self.main_save_path)
        return None
    
    def open_folder(self, path):
        import webbrowser
        webbrowser.open('file:///' + path)
    
    def init_plot(self):
        if not self.fig:
            self.good_frames = [x for x in self.hdu_name_list if ("lev1_" not in x)][1:]
            self.good_frames.insert(0, self.good_frames[0])
            self.n_plots = len(self.good_frames)
            self.n_rows = 2
            self.n_cols = self.n_plots // self.n_rows
            self.n_slots = self.n_rows * self.n_cols
            while self.n_slots < self.n_plots:
                self.n_cols += 1
                self.n_slots = self.n_rows * self.n_cols
            
            self.fig, self.axArray = plt.subplots(self.n_rows, self.n_cols, sharex="all", sharey="all")
            self.fig.suptitle("Comparison of {} at {}".format(self.wave, self.header["T_REC"]))
            self.axArray = self.axArray.flatten()
            self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
            blank = np.zeros_like(self.params.raw_image)
            for ax in self.axArray:
                ax.imshow(blank, interpolation="None")
                ax.set_title(" ")
    
    def add_to_plot(self, frame_name_in, frame):
        
        suffix = "_mod" if frame_name_in == self.last_frame_name else ""
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
        self.count += 1
    
    def finalize_and_save_plots(self):
        print(" *    Exporting Files...", end="")
        self.fig.set_size_inches(10,8)
        plt.tight_layout()
        # plt.show(block="True")
        # now = int(np.round(time.time() - 1645314148 -107115))
        wave = self.wave
        plt.tight_layout()
        zooms = os.path.join(self.params.imgs_top_directory(), "zooms")
        os.makedirs(zooms, exist_ok=True)
        
        dpi = 500
        
        save_path = os.path.join(self.params.imgs_top_directory(), "{}_compare.png".format(wave) )
        self.main_save_path = save_path
        self.fig.savefig(save_path, dpi=dpi)
        
        save_path = os.path.join(zooms, "{}_compare.png".format(wave) )
        self.fig.savefig(save_path, dpi=dpi)
        
        save_path = os.path.join(zooms, "1_zoom_{}_compare.png".format(wave) )
        plt.xlim((3250,4000))
        plt.ylim((2250,3000))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)
        
        save_path = os.path.join(zooms, "2_zoom_{}_compare.png".format(wave) )
        plt.xlim((2404,3500))
        plt.ylim((3000,4096))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)
        
        plt.close(self.fig)
        print("Done")
    
    
    def frame_touchup(self, frame_name, frame):
        from utils.stretch_intensity_module import norm_stretch
        frame = frame.astype(np.float32)
        frame[~np.isfinite(frame)] = np.nan
    
        small = np.nanpercentile(frame, 0.01)
        for name in ["_mod", "nrgf"]: #, "int_enhance"]:
            if name in frame_name:
                frame -= small
                frame = np.power(frame, 1/2)
                
        small = np.nanpercentile(frame, 0.01)
        big = np.nanpercentile(frame, 99.99)
    
        dont = False
        for name in ["RHT"]:
            if name in frame_name:
                dont = True
        if not dont:
            rng = 3
            if big > rng or small < -rng:
                frame = (frame - small) / (big - small)
        
            # Norm Stretching
            frame = norm_stretch(frame)

        return frame, dont
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
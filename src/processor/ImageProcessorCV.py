import os
import shutil
import time

import cv2
from astropy.io import fits
from tqdm import tqdm

from processor.ImageProcessor import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from science.color_tables import aia_color_table
from utils.stretch_intensity_module import norm_stretch



"""This class holds the code for the Quantile Radial Norm Processor"""
name = filt_name = "Quantile Norm"
description = "Apply the single-shot Quantile Norm to images"
progress_verb = 'Filtering'
finished_verb = "Radially Filtered"
out_name = "QRN"




class ImageProcessorCV(ImageProcessor):
    filt_name = 'CV Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"
    finished_verb = "Written to Disk"
    out_name = "FINAL"
    
    frame_name = None
    img_frame = None
    out_path = None
    in_name = -1
    
    def __init__(self, params=None, quick=False, rp=None):
        self.shrink_factor = 1
        super().__init__(params, quick, rp)
        self.frame_name = self.params.png_frame_name
        self.save_to_fits = False
        
    
    def do_fits_function(self, fits_path, in_name=None):
        """ Main Call on the Fits Path """
        self.init_frame(fits_path, self.params.png_frame_name)
        self.render_all(reference=True)
     
        return self
    
    # def do_img_function(self):
    #     """ Main Call on the Fits Path """
    #     if False:
    #         self.plot_two()
    #         self.plot_two("Less Zoomed", True)
    #         # self.display_all()
    #
    #     # self.init_image_frame()
    #     raise NotImplementedError
    
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
            self.plot_aia_orig()
            # self.plot_aia_log()
        
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
    
    def plot_aia_orig(self):
        """Plot the raw_image data from AIA"""
        # Get the Frame and Path
        if True: #self.params.raw_image is None:
            self.frame_name = self.params.master_frame_list_oldest
            frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)
            self.params.raw_image = self.frame = np.flipud(frame)
    
            self.out_path = self.get_orig_path(mod='orig')
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        # self.out_path = self.params.orig_path
        self.do_save()

    def plot_aia_log(self):
        """Plot the raw_image data from AIA"""
        # Get the Frame and Path
        self.frame_name = self.params.master_frame_list_oldest
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)
        
        frame = np.log10(frame)
        frame = frame / np.nanpercentile(frame, 50) / 2
        
        self.frame = np.flipud(frame)
        self.out_path = self.get_orig_path(mod='log')
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        self.do_save()
        
    def do_save(self):
        self.vignette()
        self.prep_save()
        self.img_save(self.out_path)
        

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
        
        if True: #self.params.modified_image is None:
            frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(self.fits_path, self.frame_name)
            self.out_path = self.get_changed_path()
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        else:
            frame = self.params.modified_image
            self.out_path = self.params.mod_path
            
        self.frame = np.flipud(frame)
        

        # print("Saving to {}".format(self.out_path))
        
        self.do_save()
        # self.save_frame(self.frame, self.fits_path)
        

    
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
        out = self.frame_touchup(self.frame_name, self.frame + 0)
        
        self.img_frame = (self.params.cmap(out)[:, :, :3] * 255).astype(np.uint8)
        b, g, r = cv2.split(self.img_frame)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        self.params.rbg_image = rgb_img
        self.vignette()
    
    def img_save(self, path, save=True, stamp=False):
        
        if stamp:
            aH = self.params.alpha_high
            aL = self.params.alpha_low
            path = path.replace(".png", "_ah-{:0.2f}_aL-{:0.2f}.png".format(aH, aL))
        if save:
            cv2.imwrite(path, self.params.rbg_image)
        else:
            # cv2.imshow(mat=self.params.rbg_image)
            plt.imshow(self.params.rbg_image)
            plt.show()
    
    def label_plot(self):
        """Annotate with Text"""
        ## GET LABELS
        # Get frame
        # img = self.img_frame
        img = self.params.rbg_image
        
        # Get Time
        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(time_string_raw,targetZone="US/Mountain", out_fmt="%m-%d-%Y %I:%M%p %Z")
        zone = "MT"
        time_list = time_string.split()
        clock = time_list[1].lower()
        day = time_list[0][:-5]
        year = time_list[0][-4:]
        
        # Get Wavelength
        inst = 'AIA'
        _, wave = self.clean_name_string(full_name)
        
        # Get Stretch Params
        
        # Get Frame Name
        if type(self.frame_name) is list:
            frame_name = [x for x in self.frame_name if x.casefold() in self.hdu_name_list][0]
        else:
            frame_name = self.frame_name
        fname = frame_name.casefold()
        f_length = len(fname)
        name_split = fname.split('(')
        name = name_split[0]
        if len(name_split)>1:
            prev = name_split[1][:-1]
        else:
            prev = '-'
        
        # Scale to Image Size
        if self.shrink_factor == 1:
            # Rez is 4k
            scale = 4
            siz = 4
            h_spacing = 100
            thickness = 3
            h0 = 25 * scale
            rez = img.shape[0]-25*scale
            
        elif self.shrink_factor == 2:
            # rez is 2k
            scale = 3
            siz = 2
            h_spacing = 70
            thickness = 2
            h0 = 25 * scale
            rez = img.shape[0]-25*scale
            
        else:
            # rez is 1K
            scale = 2
            siz = 1
            thickness = 1
            h_spacing = 30
            h0 = 25
            rez = img.shape[0]
            
        # Calculate Locations of Labels
        h1 = h0 + h_spacing
        h2 = h1 + h_spacing
        h3 = h2 + h_spacing

        wid_of_char = 18
        x0  = rez - wid_of_char*len(name) - 5
        x1  = rez - wid_of_char*len(prev)
        x2  = rez - wid_of_char*len(inst) + 2
        x3  = rez - wid_of_char*len(wave) - 7
        
        ## APPLY LABELS
        font = 1
        # Right Side
        cv2.putText(img, name, (x0, h0-5), font, scale, (255, 255, 255), thickness)
        cv2.putText(img, prev, (x1, h1-5), font, scale, (255, 255, 255), thickness)
        cv2.putText(img, inst, (x2, h2-5), font, scale, (255, 255, 255), thickness)
        cv2.putText(img, wave, (x3, h3-5), font, scale, (255, 255, 255), thickness)
        
        # Left Side
        cv2.putText(img, clock, (0, h0), font, scale, (255, 255, 255), thickness)
        cv2.putText(img, day,   (0, h1), font, scale, (255, 255, 255), thickness)
        cv2.putText(img, year,  (0, h2), font, scale, (255, 255, 255), thickness)
        cv2.putText(img, zone,  (0, h3), font, scale, (255, 255, 255), thickness)


        # Bottom Corners
        try:
            aH = "aH: {}".format(self.params.alpha_high)
            aL = "aL: {}".format(self.params.alpha_low)
            cv2.putText(img, aH, (0, 990*siz), font, scale, (255, 255, 255), thickness)
            cv2.putText(img, aL, (0, 1015*siz), font, scale, (255, 255, 255), thickness)
        except SystemError as e:
            print(e)
            
        reticle = False
        if reticle:
            self.draw_reticle(img)
 
        # if self.params.alpha is not None:
        #     cv2.putText(img, "a={:0.3f}".format(self.params.alpha), (int(x0*0.95), h3), 0,   scale, (255, 255, 255), 3)
 
 
    def draw_reticle(self, img):
        cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                   int(self.params.header["R_SUN"]), (255, 255, 255), 3)
        
        cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                   int(10), (255, 0, 0), 10)
    
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
        """ Main Call on the Fits Path """
        # self.tic()
            
        self.init_frame_from_fits(fits_path)
        self.init_quad_figure()
        self.init_radius_array()
        
        self.collect_frames(fits_path, doBar)
        
        self.finalize_and_save_plots()
        self.reinit_constants()
        
        self.toc()
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
                
        if doBar: iterable.set_description(" *    Plots Complete", refresh=True)
    
    def image_is_plottable(self, frame_name):
        # return True
        return self.doesnt_have_wrong_string(frame_name)
        return self.does_have_right_string(frame_name)
        
    
    def does_have_right_string(self, frame_name, right_string=None):
        
        right_string = right_string or ["lev1p5(t_int)", "final(qrn)", "rht(lev1p5)", "rht(final)"]
        
        for goods in right_string:
            if frame_name.casefold() == goods:
                return True
        return False
        
        
    def doesnt_have_wrong_string(self, frame_name, wrong_string=None):
        bads = wrong_string or ["lev1p0", "t_int(lev1p0)", "t_int(primary)", "lev1p5(lev1p0)"]
        if True:
            bads.append("primary")
            bads.append("lev1p5")
        
        if self.params.multiplot_all:
            bads = []
            
        for nam in bads:
            # if nam in frame_name:
            if nam.casefold() == frame_name:
                return False
        return True
    
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
        wavenum = int(''.join(i for i in fits_path if i.isdigit()))
        wave_path = [x for x in full_paths if str(wavenum) in x]
        if len(wave_path):
            correct = wave_path[0]
        else:
            rr = self.params.rez
            correct = 0.75*np.ones((rr,rr))
        image = Image.open(correct)
        # from astropy.nddata import block_reduce
        # frame = block_reduce(frame, self.shrink_factor/2)
        # frame=  frame.rotate(180)
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        ax = self.axArray[self.count_frames]
        ax.imshow(image, origin='lower', interpolation="None")
        ax.set_title('"The Sun Today"')
        
        return
        
    
    def handle_one_frame(self, fits_path, frame_name, doBar, iterable):
        if doBar: iterable.set_description(" *     Plotting {}".format(frame_name.ljust(self.max_width)), refresh=True)
        frame1, wave1, t_rec1, center1, int_time, name = self.load_this_fits_frame(fits_path, frame_name)
        # frame1[self.vignette_mask] = np.nan
        self.add_to_plot(name, frame1)
    
    def init_frame_from_fits(self, fits_path=None, in_name=-1):
        """Load the fits file from disk and get a in_name or two"""
        
        self.fits_path = fits_path or self.fits_path
        self.params.fits_path = self.fits_path
        
        self.params.raw_image, _, _, _, _, self.raw_name = \
            self.load_this_fits_frame(fits_path, self.params.master_frame_list_oldest)
        
        self.params.modified_image, wave1, t_rec1, _, _, self.mod_name = \
            self.load_this_fits_frame(fits_path, in_name)
        
        # self.peek_frames()
        self.image_data = str(wave1), fits_path, t_rec1, self.params.modified_image.shape
        self.params.make_file_paths(self.image_data)
        self.name, self.wave = self.clean_name_string(str(wave1))
    
    def open_folder(self, path):
        import webbrowser
        webbrowser.open('file:///' + path)
    
    def init_quad_figure(self):
        self.good_frames = [x for x in self.hdu_name_list if self.image_is_plottable(x)]
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
        # self.good_frames.pop(0)
        self.params.doing_jpeg = self.params.doing_jpeg
        if self.params.doing_jpeg:
            self.good_frames.insert(0, "jpeg")
        
        self.good_frame_stems = [x.split('(')[0] for x in self.good_frames]
        
        self.n_plots = len(self.good_frames)
        self.n_rows = 2 if self.n_plots > 2 else 1
        self.n_cols = max((self.n_plots // self.n_rows, 1)) if self.n_plots > 2 else 2
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
            self.axArray[self.count_frames].imshow(frame, cmap='hsv', origin='lower', interpolation="None")
        else:
            vmin = None if self.dont_vminmax else 0.
            vmax = None if self.dont_vminmax else 1.
            self.axArray[self.count_frames].imshow(frame, origin="lower", vmin=vmin, vmax=vmax,
                                                   cmap=self.params.cmap, interpolation="None")
            
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
    
        save_path = os.path.join(self.params.imgs_top_directory(), "compare", "{:04}_compare.png".format(int(self.wave)))
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
        
        save_path = os.path.join(zooms, "1_zoom_{:04}_compare.png".format(int(self.wave)))
        plt.xlim((3250 / self.shrink_factor, 4000 / self.shrink_factor))
        plt.ylim((2250 / self.shrink_factor, 3000 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)
        
        save_path = os.path.join(zooms, "2_zoom_{:04}_compare.png".format(int(self.wave)))
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
    filt_name = 'MultiHistogram Plotter'
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
        """ Main Call on the Fits Path """
        # self.tic()
        
        self.init_frame_from_fits(fits_path)
        self.init_radius_array()
        # self.init_bin_array()
        self.init_statistics()

        self.init_quad_figure()
        
        self.collect_frames(fits_path, doBar)
        
        # self.finalize_and_save_plots()
        # self.reinit_constants()
        
        self.toc()
        # self.open_folder(self.main_save_path)
        return None

    def collect_frames(self, fits_path, doBar=False):
        # with open(fits_path) as fp:
        with fits.open(fits_path, cache=False, reprocess_mode="update") as hdul:
            self.hdu_name_list = self.list_hdus(hdul)
        self.good_frames = [x for x in self.hdu_name_list if self.image_is_plottable(x)]
        
        self.max_width = np.max([len(x) for x in self.good_frames])
        iterable = tqdm(self.good_frames, desc="") if doBar else self.good_frames
        
        images, names = [], []
        for frame_name in iterable:
            if "rht" in frame_name:
                continue
            frame, wave1, t_rec1, _, _, mod_name = self.load_this_fits_frame(fits_path, frame_name)
            if False:
                frame = self.resize_image(frame, prnt=False)
            images.append(frame), names.append(frame_name)
            
            # if "lev1p5(t_int)" in frame_name:
            #     images.append(np.sqrt(frame)), names.append("sqrt(lev1p5)")
            if "qrn(lev1p5)" in frame_name:
                images.append(norm_stretch(frame)), names.append("upsilon(qrn)")
                
        self.do_compare_histogramplot(images, names)
        self.do_compare_histogramplot_qrnonly(images, names)
        
        # for frame_name in iterable:
            
            # self.handle_one_frame(fits_path, frame_name, doBar, iterable)
            # self.count_frames += 1
                
        if doBar: iterable.set_description(" *    Plots Complete", refresh=True)
    

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
            self.params.raw_image, _, _, _, _, self.raw_name = \
                self.load_this_fits_frame(fits_path, self.params.master_frame_list_oldest)
        
        self.params.modified_image, wave1, t_rec1, _, _, self.mod_name = \
            self.load_this_fits_frame(fits_path, in_name)
        
        if False:
            self.params.modified_image = self.resize_image(self.params.modified_image, prnt=False)
            self.params.raw_image = self.resize_image(self.params.raw_image, prnt=False)
        
        
        # self.peek_frames()
        self.image_data = str(wave1), fits_path, t_rec1, self.params.modified_image.shape
        self.params.make_file_paths(self.image_data)
        self.name, self.wave = self.clean_name_string(str(wave1))
        return self.params.modified_image, self.name
    
    def open_folder(self, path):
        import webbrowser
        webbrowser.open('file:///' + path)
    
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
        
        save_path = os.path.join(zooms, "1_zoom_{:04}_compare.png".format(int(self.wave)))
        plt.xlim((3250 / self.shrink_factor, 4000 / self.shrink_factor))
        plt.ylim((2250 / self.shrink_factor, 3000 / self.shrink_factor))
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=dpi)
        
        save_path = os.path.join(zooms, "2_zoom_{:04}_compare.png".format(int(self.wave)))
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
    
    
    def image_is_plottable(self, frame_name):
        # return True
        return self.doesnt_have_wrong_string(frame_name)
        return self.does_have_right_string(frame_name)
        
    
    def does_have_right_string(self, frame_name, right_string=None):
        
        right_string = right_string or ["lev1p5(t_int)", "final(qrn)", "rht(lev1p5)", "rht(final)"]
        
        for goods in right_string:
            if frame_name.casefold() == goods:
                return True
        return False
        
        
    def doesnt_have_wrong_string(self, frame_name, wrong_string=None):
        bads = wrong_string or ["lev1p0", "t_int(lev1p0)", "t_int(primary)", "lev1p5(lev1p0)", "compressed_image",
                                "final(qrn)"]
        if True:
            bads.append("primary")
            bads.append("lev1p5")
        
        if self.params.multiplot_all:
            bads = []
            
        for nam in bads:
            # if nam in frame_name:
            if nam.casefold() == frame_name:
                return False
        return True
    
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
        wavenum = int(''.join(i for i in fits_path if i.isdigit()))
        wave_path = [x for x in full_paths if str(wavenum) in x]
        if len(wave_path):
            correct = wave_path[0]
        else:
            rr = self.params.rez
            correct = 0.75*np.ones((rr,rr))
        image = Image.open(correct)
        # from astropy.nddata import block_reduce
        # frame = block_reduce(frame, self.shrink_factor/2)
        # frame=  frame.rotate(180)
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        ax = self.axArray[self.count_frames]
        ax.imshow(image, origin='lower', interpolation="None")
        ax.set_title('"The Sun Today"')
        
        return
        

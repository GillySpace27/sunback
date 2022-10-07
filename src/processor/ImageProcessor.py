import os
# import sys
# from copy import copy
from datetime import datetime, timedelta
from os import listdir
# from os.path import dirname
from os.path import join
# from time import strftime
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
from processor.Processor import Processor
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

from science.color_tables import aia_color_table

from utils.stretch_intensity_module import norm_stretch

class ImageProcessor(Processor):
    filt_name = 'Image Writer'
    out_name = ""
    wave = None
    # progress_stem = "    Exporting Pngs {}"
    progress_text = ""
    video_name_stem = ""
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"
    # do_png = False
    changed = None
    raw = None
    image_data = None
    show = False
    inches = 10
    # vmax_plot = 0.8
    # vmin_plot = 0.5
    dpi = None
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.save_to_fits = False
        self.mod_name = None
        self.raw_name = None
        self.frame_name0 = None
        self.params.cmap = None
        self.fig, self.frame_ax = None, None
        self.plot_formatted = False
        self.path_box = []
        self.figure_box = []
        self.skipped = 0
        self.frame = None
        self.load_curves()
        self.save_to_fits = True
        
        
        try:
            pass
        except AttributeError as e:
            print(e)
            print("I failed in ImageProceesor 55")
            
    def do_fits_function(self, fits_path, in_name=None):
        """This is the do_fits_function for this """
        self.init_frame(fits_path, self.params.png_frame_name)
        if self.render():
            self.export_files()
        return self
    
    def init_frame(self, fits_path=None, in_name=None):
        """Load the fits file from disk and get a in_name or two"""
        # self.load_curves()
        if in_name is not None:
            self.frame_name = in_name
        self.fits_path = fits_path or self.fits_path
        self.params.fits_path = self.fits_path
        if True: #self.params.raw_image is None:
            list_of_inputs = self.params.master_frame_list_oldest
            frame0, _, _, _, _, name0 = self.load_this_fits_frame(fits_path, list_of_inputs)
            self.raw_name = self.frame_name + ''
            frame1, wave1, t_rec1, center1, int_time, name1 = self.load_this_fits_frame(fits_path, in_name.casefold())
            self.wave1 = wave1
            self.t_rec1 = t_rec1
            self.mod_name = self.frame_name + ''
            self.params.raw_image, self.params.modified_image = frame0, frame1
            self.frame = np.zeros_like(self.params.raw_image)
            
        # self.peek_frames()
        try:
            shp = frame1.shape
        except AttributeError:
            shp = (self.params.rez, self.params.rez)
        self.image_data = str(self.wave1), fits_path, self.t_rec1, shp
        self.params.make_file_paths(self.image_data)
        self.figure_box = []
        self.path_box = []
        self.name, self.wave = self.clean_name_string(self.image_data[0])
        use_cmap=True
        if use_cmap and self.wave:
            self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
        else:
            from matplotlib import cm
            self.params.cmap = cm.gray
    def image_is_plottable(self, frame_name):
        # return True
        return self.doesnt_have_wrong_string(frame_name)
        return self.does_have_right_string(frame_name)
        
    
    def does_have_right_string(self, frame_name, right_string=None):
        
        right_string = right_string or ["lev1p5(t_int)", "final(rhe)", "rht(lev1p5)", "rht(final)"]
        
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
    def init_image_frame(self):
        """Load the fits file from disk and get a in_name or two"""

        # self.raw, self.changed = self.params.raw_image, self.params.modified_image
        self.frame = np.zeros_like(self.params.raw_image)
        # self.peek_frames()
        try:
            shape = self.frame.shape
        except:
            shape = 4096
            
        self.image_data = wave1, fits_path, t_rec1, shape = self.params.image_data

        self.name, self.wave = self.clean_name_string(self.image_data[0])
        self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)

    def peek_frames(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', sharey='all')
        ax1.imshow(self.params.raw_image)
        ax2.imshow(self.params.modified_image)
        ax3.imshow(np.abs(self.params.modified_image-self.params.raw_image))
        plt.tight_layout()
        plt.show(block=True)
        

        

    
    def render(self):
        """Render the raw and changed plots"""
        # Which plots to make?
        if self.skip():
            return False
        if self.params.do_orig:
            trials = [False, True]
        else:
            trials = [True]
        
        # Make them
        for processed in trials:
            self.render_one(processed)
        
        return True
    
    def skip(self):
        if self.params.overwrite_pngs() or self.reprocess_mode():
            # If you do want to overwrite
            return False  # Don't Skip
        else:
            # If you don't want to overwrite
            if self.params.png_save_path in self.params.local_imgs_paths():
                # Make images you don't already have
                self.skipped += 1
                return True  # do skip
            else:
                return False  # don't skip
    
    def render_one(self, processed):
        raise NotImplementedError
        
    def export_files(self):
        raise NotImplementedError
    
    def save_concatinated(self, destroy=False):
        # print("Saving Concatinated!!")
        """Make the side by side concatinated images"""
        cat_command_stem = 'ffmpeg -i "{}" -i "{}" -y -filter_complex hstack "{}" -hide_banner -loglevel error'
        orig_path, mod_path, cat_path = self.check_concat_readiness()
        
        if cat_path:
            cat_command = cat_command_stem.format(orig_path, mod_path, cat_path)
            os.system(cat_command)
            if destroy:
                os.remove(orig_path)
                os.remove(mod_path)
                
    def check_concat_readiness(self):
        # Select Directories
        mod_dir = self.params.mods_directory()
        orig_dir = self.params.orig_directory
        cat_path = self.params.cat_path
        
        # Select Paths
        orig_path = self.path_box[0] if self.path_box else self.get_orig_path()
        mod_path = self.path_box[1] if self.path_box else self.get_changed_path()

        # Confirm raw
        raw_paths  = [join(orig_dir, x) for x in listdir(orig_dir)]
        have_orig = orig_path in raw_paths
        
        # Confirm Modified
        processed_paths = [join(mod_dir, x) for x in listdir(mod_dir)]
        have_mod = mod_path in processed_paths
        
        do_cat = self.params.do_cat
        if do_cat and have_mod and have_orig:
            return orig_path, mod_path, cat_path
        else:
            return None, None, None
                
    def get_orig_path(self, mod=False):
        if self.params.do_single:
            return self.params.orig_path.replace("orig\\","").replace("image_lev1p0", mod if mod else "raw")
        else:
            if not mod:
                return self.params.orig_path
            else:
                return self.params.orig_path.replace(".png", "_{}.png".format(mod))
            
    def get_changed_path(self):
        if self.params.do_single:
            return self.params.mod_path.replace("mod\\","").replace("image_lev1p0", "mod")
        else:
            return self.params.mod_path + ''

        
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
        if name:
            digits = "{:04d}".format(int(name))
        else:
            digits = "none"
        # while name[0] == '0':
        #     name = name[1:]
        return digits, name
    

        
        # tz = timezone(timedelta(range_hours=-1))
        # import pdb; pdb.set_trace()
        # cleaned = time_string.replace(tzinfo=timezone.utc).astimezone(tz=None)
        # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime("%I:%M%p, %b-%d, %Y")
        # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=tz).strftime("%I:%M%p, %b-%d, %Y")
        # cleaned = Time(time_string).datetime.strftime("%I:%M%p, %b-%d, %Y")
        # print("----------->", cleaned)
        # import pdb; pdb.set_trace()
        # return
        # name = full_name + ''
        # while name[0] == '0':
        #     name = name[1:]
        # return name
    
    @staticmethod
    def absqrt(image, **kwargs):
        return np.sqrt(np.abs(image, **kwargs))
    
    def frame_touchup(self, frame_name, frame):
        # print("Touchup on {}".format(frame_name))
        # print("I RAN on {}, {}".format(frame_name, self.params.current_wave()))
        # maxmax = np.nanpercentile(frame, 99)
        # minmin = np.nanpercentile(frame, 1)
        # themax = np.nanmax(frame)
        # if themax > 100 or themax < 0.8:
        #     # out = (self.frame - minmin) / (maxmax - minmin)
        #     pass
        # self.params.short_circuit=True
        if self.params.short_circuit or "legacy" in frame_name:
            return frame

        # Frame Cleanup
        frame = frame.astype(np.float32)
        frame[~np.isfinite(frame)] = np.nan
        
        if '(' in frame_name:
            frame_name2 = frame_name.split('(')[0]
        else:
            frame_name2 = frame_name
            
        if "primary" in frame_name2:
            frame = np.sqrt(frame)
    
        basic_scrunch = True
        if basic_scrunch:
            frame = self.scrunch(frame)
    
        ## Perform Nonlinear Transforms
        # Power
        for name in ["lev1p5", "_mod", "nrgf", "primary"]:  # , "int_enhance"]:
            if name in frame_name2:
                frame = self.power_mod(frame)

        
        # Maxima Stretching
        do_maxima_scrunch = True
        if do_maxima_scrunch:
            if 'rhe' in frame_name2:
                frame = self.maxima_scrunch(frame, num2=0.)
            elif "msgn(rhe)" in frame_name:
                frame = self.maxima_scrunch(frame, num=0.95, num2=0.1)
                # frame *= 1.05
            elif "lev1p5" in frame_name:
                if self.params.current_wave() in ['94', '0094', '0131', '131']:
                    frame = np.sqrt(np.abs(frame))
                    frame = self.maxima_scrunch(frame, num=np.nanmin(frame.flatten()), num2=np.nanmax(frame.flatten()))
                    
                    pass
                else:
                    num = 0.88
                    num2 = 0.075
                    frame = self.maxima_scrunch(frame, num=num, num2=num2)
                    # frame *= 1.05
            
            # else:
            #     pass
            #     frame = self.maxima_scrunch(frame)
 
        # Norm Stretching (only runs on rhe)
        frame = self.do_norm_stretch(frame, frame_name)
        

        
        dont_vminmax = False
        for name in ["RHT", 'legacy']:
            if name in frame_name:
                dont_vminmax = True

        if not dont_vminmax:
            frame[frame>1.0] = 1.0
            frame[frame<0.0] = 0.0
            
        self.dont_vminmax = dont_vminmax
        return frame
    
    
    def do_norm_stretch(self, frame, frame_name, do=True):
        if do and "rhe" in frame_name:
            aL, aH = self.get_alphas()
            frame = norm_stretch(frame, alpha=aL, alpha_high=aH)
        return frame
        
    def get_alphas(self):
        wave = self.params.current_wave(self.image_data[0])
        wave = "{:04}".format(int(wave))
        
        wave_list = [{"wave": "0094", "aL": 0.50, "aH": 0.35},
                     {"wave": "0131", "aL": 0.50, "aH": 0.30},
                     {"wave": "0171", "aL": 0.50, "aH": 0.40},
                     {"wave": "0193", "aL": 0.50, "aH": 0.45},
                     {"wave": "0211", "aL": 0.50, "aH": 0.40},
                     {"wave": "0304", "aL": 0.50, "aH": 0.40},
                     {"wave": "0335", "aL": 0.50, "aH": 0.40},
                     {"wave": "1600", "aL": 0.50, "aH": 0.40},
                     {"wave": "1700", "aL": 0.50, "aH": 0.40}]

        dictdict = {}
        for wv in wave_list:
            dictdict[wv["wave"]] = wv
            
        self.params.alpha_low  = dictdict[wave]['aL']
        self.params.alpha_high = dictdict[wave]['aH']
        return self.params.alpha_low, self.params.alpha_high
        
                        # frame = 0.95 * frame
    # if frame_name == "nrgf":
    #     # Replace the Disk
    #     self.init_radius_array()
    #     mask = self.radius < self.limb_radius_from_header*0.5
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
    # if frame_name == "rhe":
    #     frame /= darken_quant
    
    # self.vignette_mask = np.asarray(self.radius > self.vig_radius_pix, dtype=bool)
    # frame[self.vignette_mask] = np.nan
    
    def power_mod(self, frame):
        frame *= 10.
        pow = 1/2.5
        np.power(frame, pow, out=frame)
        frame *= pow
        
        # frame = np.log10(frame)
        # frame = frame / np.nanpercentile(frame, 50) / 2
        
        return frame
    
    def scrunch(self, frame, n_exclude=50, perc_exclude=0.01):
        # lowlow = np.nanmin(frame)
        # highigh = np.nanmax(frame)
        
        # total = self.params.rez ** 2
        # perc_exclude = n_exclude / total
        
    
    
        low = np.nanpercentile(frame, perc_exclude)
        high = np.nanpercentile(frame, 100-perc_exclude)
    
        frame = self.norm_formula(frame, low, high)
        return frame
    
    def maxima_scrunch(self, frame, num=1.0, num2=0.06):
        mask1 = (frame > num)
        mask2 = (frame < num2)
        frame[mask1] = np.nan
        frame[mask2] = np.nan
        frame = self.scrunch(frame)
        frame[mask1] = 1.0
        frame[mask2] = 0.
        return frame
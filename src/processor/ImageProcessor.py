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


class ImageProcessor(Processor):
    filt_name = 'Image Writer'
    out_name = ""
    do_png = False
    save_to_fits = False
    wave = None
    # progress_stem = "    Exporting Pngs {}"
    progress_text = ""
    video_name_stem = ""
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"
    
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
    
        self.params.cmap = None
        self.fig, self.frame_ax = None, None
        self.plot_formatted = False
        self.path_box = []
        self.figure_box = []
        self.skipped = 0
        self.frame = None
        self.load_curves()
        
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
    
    def init_frame(self, fits_path=None, in_name=-1):
        """Load the fits file from disk and get a field or two"""
        # self.load_curves()
        self.fits_path = fits_path or self.fits_path
        self.params.fits_path = self.fits_path
        if True: #self.params.raw_image is None:
            list_of_inputs = ["lev1p5_T", "lev1p5_L", "lev1_t_int", "lev1_Single"]
            frame0, _, _, _, _ = self.load_a_fits_field(fits_path, list_of_inputs)
            frame1, self.wave1, self.t_rec1, center1, int_time = self.load_a_fits_field(fits_path, in_name)
            self.params.raw_image, self.params.modified_image = frame0, frame1
            self.frame = np.zeros_like(self.params.raw_image)
        # self.peek_frames()
        self.image_data = str(self.wave1), fits_path, self.t_rec1, frame1.shape
        self.params.make_file_paths(self.image_data)
        self.figure_box = []
        self.path_box = []
        self.name, self.wave = self.clean_name_string(self.image_data[0])
        use_cmap=True
        if use_cmap:
            self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)

        else:
            from matplotlib import cm
            self.params.cmap = cm.gray

    def init_image_frame(self):
        """Load the fits file from disk and get a field or two"""

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
        if self.params.do_orig or True: #TODO shouldn't be permanent
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
        orig_path = self.path_box[0] if self.path_box else self.get_raw_path()
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
                
    def get_raw_path(self, mod=False):
        if self.params.do_single:
            return self.params.orig_path.replace("orig\\","").replace("image_lev1", mod if mod else "raw")
        else:
            return self.params.orig_path
    
    def get_changed_path(self):
        if self.params.do_single:
            return self.params.mod_path.replace("mod\\","").replace("image_lev1", "mod")
        else:
            return self.params.mod_path

        
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
        digits = "{:04d}".format(int(name))
        # while name[0] == '0':
        #     name = name[1:]
        return digits, name
    
    @staticmethod
    def clean_time_string(time_string):
        # Make the name strings
        
        cleaned = datetime.strptime(time_string[:-4], "%Y-%m-%dT%H:%M:%S")
        cleaned += timedelta(hours=-7)
        
        # tz = timezone(timedelta(range_hours=-1))
        # import pdb; pdb.set_trace()
        # cleaned = time_string.replace(tzinfo=timezone.utc).astimezone(tz=None)
        # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime("%I:%M%p, %b-%d, %Y")
        # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=tz).strftime("%I:%M%p, %b-%d, %Y")
        # cleaned = Time(time_string).datetime.strftime("%I:%M%p, %b-%d, %Y")
        # print("----------->", cleaned)
        # import pdb; pdb.set_trace()
        return cleaned.strftime("%m-%d-%Y %I:%M%p")
        # name = full_name + ''
        # while name[0] == '0':
        #     name = name[1:]
        # return name
    
    @staticmethod
    def absqrt(image, **kwargs):
        return np.sqrt(np.abs(image, **kwargs))

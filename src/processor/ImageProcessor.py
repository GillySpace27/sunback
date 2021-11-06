import os
import sys
from copy import copy
from datetime import datetime, timedelta
from os import listdir
# from os.path import dirname
from os.path import join
from time import strftime
import matplotlib as mpl
mpl.use('Agg')
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
    progress_verb = "Writing Images"
    
    changed = None
    original = None
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
        try:
            self.load_curves()
        except AttributeError as e:
            print(e)
    def do_fits_function(self, fits_path, in_name=None):
        """This is the do_fits_function for this """
        self.init_frame(fits_path, self.params.png_frame_name)
        if self.render():
            self.export_files()
        return self
    
    def init_frame(self, fits_path, in_name=-1):
        """Load the fits file from disk and get a field or two"""
        # self.load_curves()
        if self.params.original_image is None:
            frame0, _, _, _, _ = self.load_first_fits_field(fits_path)
            frame1, wave1, t_rec1, center1, int_time = self.load_a_fits_field(fits_path, in_name)
            # frame1, wave1, t_rec1, center1, int_time = self.load_last_fits_field(fits_path)
            self.params.local_imgs_paths()
            self.params.original_image, self.params.modified_image = frame0, frame1
            self.frame = np.zeros_like(self.params.original_image)
        # self.peek_frames()
        try:
            shape = frame1.shape
        except:
            shape = 4096
        self.image_data = str(wave1), fits_path, t_rec1, shape
        self.make_directories()
        self.figure_box = []
        self.path_box = []
        self.name, self.wave = self.clean_name_string(self.image_data[0])
        self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)

    def init_image_frame(self):
        """Load the fits file from disk and get a field or two"""

        # self.original, self.changed = self.params.original_image, self.params.modified_image
        self.frame = np.zeros_like(self.params.original_image)
        # self.peek_frames()
        try:
            shape = self.frame.shape
        except:
            shape = 4096
            
        self.image_data = wave1, fits_path, t_rec1, shape = self.params.image_data

        self.name, self.wave = self.clean_name_string(self.image_data[0])
        self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)

    def peek_frames(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
        
        
        ax1.imshow(self.params.original_image)
        ax2.imshow(self.params.modified_image)
        ax3.imshow(np.abs(self.params.modified_image-self.params.original_image))
        plt.tight_layout()
        plt.show(block=True)
        

        
    def make_directories(self):
        _, self.fits_save_path, _, _ = self.image_data
        self.png_save_path = self.fits_save_path.replace('fits', 'png')
        self.png_save_stem = self.png_save_path[:-4] + '{}' + ".png"
        self.png_save_directory = os.path.dirname(self.png_save_path)
        # self.clean_directory()

    
        self.orig_directory = join(self.png_save_directory, "orig")
        os.makedirs(self.png_save_directory, exist_ok=True)
        os.makedirs(os.path.dirname(self.orig_directory), exist_ok=True)
        os.makedirs(self.orig_directory, exist_ok=True)
    
    def render(self):
        """Render the original and changed plots"""
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
            if self.png_save_path in self.params.local_imgs_paths():
                # Make images you don't already have
                self.skipped += 1
                return True  # do skip
            else:
                return False  # don't skip
    
    def render_one(self, processed):
        raise NotImplementedError
            
    def export_files(self):
        raise NotImplementedError
    
    def save_concatinated(self, which1="orig", which2="SRN", destroy=False):
        # print("Saving Concatinated!!")
        """Make the side by side concatinated images"""
    

    
        processed_paths = [join(self.png_save_directory, x) for x in listdir(self.png_save_directory)
                           if not os.path.isdir(join(self.png_save_directory, x))]
        original_paths  = [join(self.orig_directory, x) for x in listdir(self.orig_directory)]
        path_list_abs =  processed_paths + original_paths
    
        original = self.path_box[0] if self.path_box else self.get_original_path()
        processed = self.path_box[1] if self.path_box else self.get_changed_path()
    
    
        go_1 = self.params.do_cat
        go_2 = original in original_paths
        go_3 = self.get_changed_path() in processed_paths
    
        self.cat_path = self.png_save_stem.replace("\\png\\","\\png\\cat\\").format("_cat")
    
        fmt_string_stem = 'ffmpeg -i "{}" -i "{}" -y -filter_complex hstack "{}" -hide_banner -loglevel error'
        cat_command = fmt_string_stem.format(original, processed, self.cat_path)
        if go_1 and go_2 and go_3:
            os.makedirs(os.path.dirname(self.cat_path), exist_ok=True)
            os.system(cat_command)
            if destroy:
                os.remove(original)
                
                
    def get_original_path(self):
        return self.png_save_stem.replace("\\png\\", "\\png\\orig\\").format("_orig")
    
    def get_changed_path(self):
        nam = self.params.png_frame_name
        name = nam if type(nam) is str else self.hdu_name_list[nam]
        return self.png_save_stem.format('').replace("aia", name + "_" + "aia")

        
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

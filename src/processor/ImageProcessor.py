import os
from copy import copy
from datetime import datetime, timedelta
from os import listdir
# from os.path import dirname
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
        
        self.fig, self.frame_ax = None, None
        self.plot_formatted = False
        self.pathBox = []
        self.figbox = []
        self.skipped = 0
        
        self.load_curves()
    
    def do_fits_function(self, fits_path, in_name=None):
        """This is the do_fits_function for this """
        self.prep_image(fits_path)
        if self.render():
            self.export_files()
        return self
    
    def prep_image(self, fits_path):
        """Load the fits file from disk and get a field or two"""
        frame0, _, _, _, _ = self.load_first_fits_field(fits_path)
        frame1, wave1, t_rec1, center1, int_time = self.load_last_fits_field(fits_path)
        self.params.local_imgs_paths()
        self.original, self.changed = copy(frame0), copy(frame1)
        self.image_data = str(wave1), fits_path, t_rec1, frame1.shape
        self.make_directories()
        self.figbox = []
        self.pathBox = []
    
    def make_directories(self):
        self.load_curves()
        _, self.fits_save_path, _, _ = self.image_data
        self.png_save_path = self.fits_save_path.replace('fits', 'png')
        self.png_save_stem = self.png_save_path.split(".")[0] + '{}' + ".png"
        self.png_save_directory = os.path.dirname(self.png_save_path)
        os.makedirs(self.png_save_directory, exist_ok=True)
    
    def render(self):
        """Render the original and changed plots"""
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
            if self.png_save_path in self.params.local_imgs_paths():
                # Make images you don't already have
                self.skipped += 1
                return True  # do skip
            else:
                return False  # don't skip
    
    def render_one(self, processed):
        """Render one image"""
        # Pull in the required inputs
        # out_array = self.changed
        # original_image = self.original
        
        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(time_string_raw)
        name, wave = self.clean_name_string(full_name)
        
        # Create the Figure
        if self.fig is None:
            self.fig, self.frame_ax = plt.subplots()
        self.format_plot()
            
        inst, height = self.plot_aia(self.fig, self.frame_ax, wave, processed)
        
        # Format the Plot and Save
        self.label_plot(name, inst, height, wave, time_string)
        self.figbox.append([self.fig, self.frame_ax, processed])
        if self.show:
            plt.show()
            
    def export_files(self):
        try:
            for fig, ax, processed in self.figbox:
                self.execute_plot_save(fig, ax, processed)
        except Exception as e:
            print("Export_Files:", e)
        finally:
            for fig, _, _ in self.figbox:
                pass #plt.close(fig)
                
    def execute_plot_save(self, fig, ax, processed):
        out_path = self.png_save_stem.format('' if processed else "_orig")
        fig.savefig(out_path, facecolor='black', edgecolor='black', dpi=self.dpi)
        ax.cla() # plt.close(fig)
        self.pathBox.append(out_path)
        self.figbox = []
        
        # self.save_concatinated()
    
    def format_plot(self):
        """Make a plot look good"""
        # if not self.plot_formatted:
        # Tweak the Figure Properties
        self.fig.set_facecolor("k")
        self.inches = 10
        self.dpi = self.changed.shape[0] / self.inches
        self.fig.set_size_inches((self.inches, self.inches))
        self.blankAxis(self.frame_ax)
        self.plot_formatted = True
    
    def label_plot(self, name, inst, height, wave, time_string):
        """Annotate with Text"""
        buffer = '' if len(name) == 3 else '  '
        buffer2 = '    ' if len(name) == 2 else ''
        
        title = "{}    {} {}, {}{}".format(buffer2, inst, wave, time_string, buffer)
        self.frame_ax.annotate(title, (0.15, height + 0.02), xycoords='axes fraction', fontsize='large',
                               color='w', horizontalalignment='center')
        
        the_time = strftime("%Z %I:%M%p")
        if the_time[0] == '0':
            the_time = the_time[1:]
        self.frame_ax.annotate(the_time, (0.15, height), xycoords='axes fraction', fontsize='large',
                               color='w', horizontalalignment='center')
        
        self.frame_ax.annotate(the_time, (0.15, height-15), xycoords='axes fraction', fontsize='large',
                               color='w', horizontalalignment='center')
        
        self.frame_ax.annotate("Mode: {}".format(self.params.selection), (0.90, height+15), xycoords='axes fraction', fontsize='large',
                               color='w', horizontalalignment='center')
        
    
    def plot_hmi(self, fig, ax, frame):
        """Plot the data from HMI"""
        inst = ""
        height = 1.05
        ax.imshow(frame, origin='upper', interpolation=None)
        plt.tight_layout(pad=5.5)
        return inst, height
    
    def plot_aia(self, fig, ax, wave, processed):
        """Plot the data from AIA"""
        inst = '  AIA'
        height = 0.95
        cmap = aia_color_table(int(wave) * u.angstrom)
        if processed:
            frame = self.changed #.astype(np.float16)
            vmin = 0.05 #.elf.absolute_min # 0.1 * 65536 # self.vmin_plot * 65536 #2np.max(np.max(out_array))
            vmax = 1.5  #* self.absolute_max # 0.9 * 65536 # self.vmax_plot * 65536 # * np.max(np.max(out_array))
            # print("vin, vmax = ", vmin, vmax)
            ax.imshow(frame, cmap=cmap, origin='lower', interpolation=None, vmin=vmin, vmax=vmax)
        else:
            frame = self.absqrt(self.original, dtype=np.float32)
            ax.imshow(frame, cmap=cmap, origin='lower', interpolation=None)  # ,  vmin=self.vmin_plot, vmax=self.vmax_plot)
            # toprint = self.normalize(self.absqrt(original_image))
            # plt.imshow(toprint, cmap='sdoaia{}'.format(wave), origin='lower', interpolation=None) #,  vmin=self.vmin_plot, vmax=self.vmax_plot)
        plt.tight_layout(pad=0)
        
        return inst, height
    
    def save_concatinated(self):
        """Make the side by side concatinated images"""
        fmt_string_stem = "ffmpeg -i {} -i {} -y -filter_complex hstack {} -hide_banner -loglevel warning"
        path_list = listdir(self.png_save_directory)
        go_1 = self.params.do_cat
        go_2 = self.png_save_stem.format("_orig") in path_list
        go_3 = self.png_save_stem.format("") in path_list
        if go_1 and go_2 and go_3:
            os.system(fmt_string_stem.format(self.pathBox[1], self.pathBox[0], self.png_save_stem.format("_cat")))
    
    # def make_png_path(self, fits_path):
    #     save_path = fits_path.replace("fits", "png")
    #     return basename(save_path)
    #     png_path = self.make_png_path(fits_path)
    #
    #     if png_path.casefold() in self.done_paths:
    #         if not self.params.overwrite_pngs():
    #             return save_path
    
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
    
    # image_meta = str(wave), str(wave), t_rec, data.shape
    
    # self.make_thumbs(img_paths[0])
    # return img_paths
    #      if False:
    #          out_array, center = reduce_array(out_array, center, self.params.resolution())

# def modify_img_series(self):
#     """Processes the img series"""
#     img_paths = []
#
#     # paths = get_paths(self.params.local_fits_paths(),
#     #                self.params.use_wavelengths, self.params.download_path())
#     # self.params.local_fits_paths(paths)
#     # self.params.local_fits_paths()
#
#     for full_path in tqdm(self.params.local_fits_paths()):
#         self.done_paths = find_done_paths(full_path)
#         name = basename(full_path).casefold().replace("fits", "png")
#         if name in self.done_paths and not self.params.overwrite_pngs():
#             one_path = full_path
#         else:
#             with fits.open(full_path) as hdul:
#                 # try:
#                 one_path = self.modify_img(hdul, full_path)
#             # except [TypeError(), IndexError()] as e:
#             #     skipped += 1
#             #     print(e)
#             #     continue
#         if type(one_path) not in [list]:
#             one_path = [one_path]
#         img_paths.extend(one_path)
#         # break
#     self.params.local_img_paths(img_paths)


#
# def modify_img(self, hdul, path=None):
#     """modifies and uploads the in_object"""
#     hdul.verify('silentfix+warn')
#
#     save_path = path.replace("fits", "png")
#     filename = basename(save_path)
#     if filename.casefold() in self.done_paths:
#         if not self.params.overwrite_pngs():
#             return save_path
#     try:
#         hh = 0
#         wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
#     except:
#         hh = 1
#         wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
#         # center = [hdul[hh].header[], hdul[hh].header[]]
#
#     center = [hdul[hh].header['X0_MP'], hdul[hh].header['Y0_MP']]
#
#     data = hdul[hh].data
#
#     # Reduce the size of the array
#     resolution = data.shape[0]
#     desired = self.params.resolution()
#
#     if resolution > desired:
#         reduce_amount = int(resolution / desired)
#         data = block_reduce(data, reduce_amount)
#         center[0] /= reduce_amount
#         center[1] /= reduce_amount
#
#     # while center[0] > 0.9 * desired:
#     #     center[0] /= 2
#     #     center[1] /= 2
#
#
#     # image_meta = str(wave), str(wave), t_rec, data.shape
#     image_meta = str(wave), save_path, t_rec, data.shape
#
#     img_paths = Modify(data, image_meta, center=center).get_paths()
#     return img_paths
# self.make_thumbs(img_paths[0])

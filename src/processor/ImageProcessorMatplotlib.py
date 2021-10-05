import os
from datetime import datetime, timedelta
from os import listdir
from os.path import join
from time import strftime
from processor.ImageProcessor import ImageProcessor
from science.color_tables import aia_color_table
import astropy.units as u
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class ImageProcessorMatplotlib(ImageProcessor):
    filt_name = 'MPL Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing Images"

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        
    def render_one(self, processed):
        """Render one image"""
      
        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(time_string_raw)
        name, wave = self.clean_name_string(full_name)
        
        # Create the Figure
        fig, frame_ax = plt.subplots()
        self.format_plot(fig, frame_ax )
            
        inst, height = self.plot_aia(fig, frame_ax, wave, processed)
        
        # Format the Plot and Save
        self.label_plot(name, inst, height, wave, time_string, frame_ax)
        self.figure_box.append([fig, frame_ax, processed])
        if self.show:
            plt.show()
            
    def export_files(self):
        try:
            for fig, ax, processed in self.figure_box:
                self.execute_plot_save(fig, ax, processed)
            self.save_concatinated()
        except Exception as e:
            print("Export_Files:", e)
        finally:
            for fig, _, _ in self.figure_box:
                plt.close(fig)
                
    def execute_plot_save(self, fig, ax, processed):
        if processed:
            nam= self.params.png_frame_name
            name = nam if type(nam) is str else self.hdu_name_list[nam]
            out_path = self.png_save_stem.format("_"+name)
        else:
            out_path = self.png_save_stem.replace("\\png\\","\\png\\orig\\").format("_orig")
            
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        fig.savefig(out_path, facecolor='black', edgecolor='black', dpi=self.dpi)
        plt.close(fig)
        self.path_box.append(out_path)
        self.figure_box = []
    
    def format_plot(self, fig, frame_ax):
        """Make a plot look good"""
        # if not self.plot_formatted:
        # Tweak the Figure Properties
        fig.set_facecolor("k")
        self.inches = 10
        self.dpi = self.changed.shape[0] / self.inches
        fig.set_size_inches((self.inches, self.inches))
        self.blankAxis(frame_ax)
        self.plot_formatted = True
    
    def label_plot(self, name, inst, height, wave, time_string, frame_ax):
        """Annotate with Text"""
        buffer = '' if len(name) == 3 else '  '
        buffer2 = '    ' if len(name) == 2 else ''
        
        title = "{}    {} {}, {}{}".format(buffer2, inst, wave, time_string, buffer)
        frame_ax.annotate(title, (0.15, height + 0.02), xycoords='axes fraction', fontsize='large',
                               color='w', horizontalalignment='center')
        
        the_time = strftime("%Z %I:%M%p")
        if the_time[0] == '0':
            the_time = the_time[1:]
        frame_ax.annotate(the_time, (0.15, height), xycoords='axes fraction', fontsize='large',
                               color='w', horizontalalignment='center')
        
        frame_ax.annotate(the_time, (0.15, height-15), xycoords='axes fraction', fontsize='large',
                               color='w', horizontalalignment='center')
        
        frame_ax.annotate("Mode: {}".format(self.params.selection), (0.90, height+15), xycoords='axes fraction', fontsize='large',
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
            # frame = self.absqrt(self.changed, dtype=np.float32)
            vmin = 0.0  #self.absolute_min # 0.1 * 65536 # self.vmin_plot * 65536 #2np.max(np.max(out_array))
            vmax = 4  #self.absolute_max # 0.9 * 65536 # self.vmax_plot * 65536 # * np.max(np.max(out_array))
            # print("vin, vmax = ", vmin, vmax)
            ax.imshow(frame, cmap=cmap, origin='lower', interpolation=None, vmin=vmin, vmax=vmax)
            ax.set_title(self.hdu_name_list[-1])
        else:
            frame = self.absqrt(self.original, dtype=np.float32)
            ax.imshow(frame, cmap=cmap, origin='lower', interpolation=None)  # ,  vmin=self.vmin_plot, vmax=self.vmax_plot)
            ax.set_title("original")#, vmin=vmin, vmax=vmax)
            
            # toprint = self.normalize(self.absqrt(original_image))
            # plt.imshow(toprint, cmap='sdoaia{}'.format(wave), origin='lower', interpolation=None) #,  vmin=self.vmin_plot, vmax=self.vmax_plot)
        plt.tight_layout(pad=0)
        
        return inst, height

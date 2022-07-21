import sys
import os

from astropy.io import fits
from numpy import where, power
import numpy as np

from fetcher.Fetcher import Fetcher
import xarray as xr
import matplotlib.pyplot as plt
import shutil
import astropy.units as u
from science.color_tables import aia_color_table

# import urllib
# from datetime import datetime
# from urllib.request import urlretrieve
# import numpy as np
# import requests
# from bs4 import BeautifulSoup
# from utils.file_util import find_root_directory
# from tqdm import tqdm
# from os import listdir
# from time import sleep
# from os.path import join
import os.path as path

default_base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class LocalFetcher(Fetcher):
    description = "Load the images from Disk"
    filt_name = "Local Fetcher"
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        
    def fetch(self, params=None):
        print(" v Loading Local Files...")
        self.load(params)
        num = self.n_fits + self.n_imgs
        print(" ^    Discovered {} fits and {} images\n".format(self.n_fits, self.n_imgs)
              if num>0 else "No Files to Load!")
        if num == 0:
            print("\n    !!Quitting Program!!\n")
            print("Base: ", self.params.base_directory())
            print("Imgs: ", self.params.imgs_top_directory())
            print("Fits: ", self.params.fits_directory())
            
            sys.exit(1)
        self.toc()

class LocalSingleFetcher(Fetcher):
    description = "Load the image_path from Disk"
    filt_name = "Local Single Fetcher"
    
    def fetch(self, params=None):
        print(" v Loading Local File...")
        self.duration = ''
        self.load(params)
        fits_path = self.determine_image_path()
        try:
            hdul = fits.open(fits_path, cache=False, ignore_missing_end=True)
        except ValueError as e:
            print("No Local File Found!")
            raise e
        self.list_hdus(hdul)
        
        for self.params.hdu_name in self.params.master_frame_list_newest:

            if self.params.hdu_name in self.hdu_name_list:
                try:
                    self.load_fits_image(self.params.use_image_path(), self.params.hdu_name)
                    # print(" *   Loaded the '{}' HDU from".format(self.params.hdu_name))
                    print(" *   Loaded".format(self.params.hdu_name))
                    print(" *     ", path.basename(self.params.use_image_path()))
                    print(" *    in\n *     ", path.dirname(self.params.use_image_path()))
                    print(" ^ Success!")
                    break
                except KeyError as e:
                    continue
                print("LocalSingleFetcher")
                raise e
        self.toc()
                
                # self.view_raw()


    

class LocalCdfFetcher(Fetcher):
    description = "Load the image_path from Disk"
    filt_name = "Local Single CDF Fetcher"

    def fetch(self, params=None):
        """Get the correct images prepared"""
        print(" v Loading Local File...")
        self.params = params or self.params
        self.find_paths()
        self.open_cdf()
        self.store_self()
#         self.peek_load()
#         self.select_frame(gen=True)
#         self.peek_selection()
    
    def store_self(self):
        """Store the fetcher into the parameters"""
        self.params.cdf_fetcher = self
        
    def find_paths(self):
        """Figure out the paths"""
        img_path = self.params.use_image_path()
        new_img_path = img_path.replace('.nc', '_filtered.nc')
        file_name = os.path.basename(img_path)
        dir_path = os.path.dirname(img_path)
        self.time_stamp = file_name[3:-3]
        
        pstem = "   Looking in: \n     {}\n     for {}  at  {}"
        print(pstem.format(dir_path, file_name, self.time_stamp))
#         if not os.path.exists(new_img_path):
#             self.copy_cdf(img_path, new_img_path)
        self.params.new_img_path = new_img_path
        return img_path
        
    def open_cdf(self, img_path=None, verb=False):
        """Load all the frames out of the CDF file"""
        # Open the Image
        use_img_path = img_path or self.params.use_image_path()
        dss = xr.open_dataset(use_img_path)
        self.params.use_cdf = True
        self.selection_counter = 0

        # Extract from CDF
        frames = dss.value
        self.cdf_waves = dss.wave_len.values
        self.frames_numpy = frames.to_numpy()+0
        self.n_frames = len(frames)
        color_frames_in = zip(self.frames_numpy, self.cdf_waves)
        self.color_frames = []
        for frame, wave in color_frames_in:
            self.color_frames.append([frame, int(wave)])
        if verb: print("       Found {} frames in the CDF file and loaded {}!".format(self.n_frames, len(self.color_frames)))
        self.params.color_frames = self.color_frames
        self.params.n_frames = self.n_frames
        dss.close()
        
    def save_cdf(self, new_img_path, frame_list, do_plot=False):
        """Load all the frames out of the CDF file"""
        orig_img_path = self.params.use_image_path()
        self.write_to_cdf(orig_img_path, new_img_path, frame_list)

        if do_plot:
            self.confirm_save(orig_img_path, new_img_path)
    
    
    def confirm_save(self, orig_img_path, new_img_path):
            # Load the raw file for reference
            
            print("\n   V Plotting Confirmation of Reduction:")
           
            print("      raw:")
            self.open_cdf(orig_img_path)
            self.peek_load(title="raw: {}".format(os.path.basename(orig_img_path)))

            print("      Modified:")
            self.open_cdf(new_img_path)
            self.peek_load(filt=False,title="MODIFIED: {}".format(os.path.basename(new_img_path)))
            print("   ^ We Plotted!\n")
        
        
    def write_to_cdf(self, orig_img_path, new_img_path, frame_list, do_plot=False):
        # Load + Legacy_SRN_Kernal the netCDF File
        dss = xr.open_dataset(orig_img_path)
#         frames = dss.value
#         print("Writing!")
       
        for index, (frame, wave) in enumerate(frame_list):
            if do_plot:
                # Display a before and after image_path
                fig, (ax1,ax2) = plt.subplots(1,2)
                ax1.imshow(self.quick_filter(dss.value[index]), origin="lower")
                ax2.imshow(frame, origin="lower")
                plt.show()
        
            # Store after_image into array
            dss.value[index] = frame
        
        dss.to_netcdf(new_img_path)
        dss.close()
        print("   New CDF saved to \n    {}".format(new_img_path))
        

        
        
    def peek_load(self, filt=True, use_cmap=True, title=None):
        #Prep Plot
        n_frames = len(self.color_frames)
        fig, axArray = plt.subplots(3,3, sharex="all", sharey="all")
        print('')
        fig.suptitle(title or "Plotting Frames Loaded in Memory")
        axArray = axArray.flatten()
        for (frame, wave), ax  in zip(self.color_frames, axArray):
            ax.set_title(wave)
            if filt:
                to_plot = self.quick_filter(frame)
            else:
                to_plot = frame

            if use_cmap:
                cmap = aia_color_table(int(wave) * u.angstrom)
            else:
                cmap = 'gray'
                
                
            ax.imshow(to_plot,cmap=cmap, origin="lower")

        fig.set_size_inches((8,8))
        plt.tight_layout()
        plt.show(block=True)
        
    def quick_filter(self, image, pow=1/3):
        return power(np.abs(image), pow)
        
    def peek_selection(self):
        """ Plot the loaded image_path"""
        
        print("")
        fig, (ax1, ax) = plt.subplots(1,2)
        fig.suptitle("Plotting Selected Frame: {}".format(self.current_wave))

        to_plot = self.quick_filter(self.params.modified_image)
        ax.imshow(to_plot, origin="lower")
        ax.set_title("Modified")

        to_plot1 = self.quick_filter(self.params.raw_image)
        ax1.imshow(to_plot1, origin="lower")
        ax1.set_title("lev1p0")
        
        fig.set_size_inches(10,5)
        plt.show()
        
    def peek_cdf(self, path):
        print("\n\n          Plotting the frames on input from CDF")
        # Open the Image
        dss = xr.open_dataset(path)

        #Prep Plot
        n_frames = len(dss.value)
        fig, axArray = plt.subplots(3,3, sharex="all", sharey="all")
        axArray = axArray.flatten()

        for frame, ax, wave in zip(dss.value, axArray, dss.wave_len.values):
            ax.set_title(int(wave))
            ax.imshow(frame, origin="lower")

        fig.set_size_inches((4,20))
        plt.tight_layout()
        plt.show(block=Trues)

    def selection_logic(self, get_ind=None, get_wave=None, gen=False):
        """Select which frame to use
                get_ind = int  :  will select frames by index
                get_wave = int :  will select frames by wavelength name
                gen = True     :  will select sequential frames eac call
        """

        if gen:
            select_ind = self.selection_counter+0
            self.selection_counter += 1
        else:
            if get_wave:
                wave_ind = where(self.cdf_waves == get_wave)
                select_ind = int(wave_ind)
            elif get_ind or get_ind==0:
                select_ind = int(get_ind)
            elif type(self.params.do_one()) is int:
                select_ind = self.params.do_one()
            elif self.params.current_wave():
                wave_ind = where(self.cdf_waves == self.params.current_wave())
                select_ind = int(wave_ind)
            else:
                select_ind = 0
        return select_ind
        
    def select_frame(self, get_ind=None, get_wave=None, gen=False, peek=False):
        """Select which frame to use, then load it
                get_ind = int  :  will select frames by index
                get_wave = int :  will select frames by wavelength name
                gen = True     :  will select sequential frames eac call
        """
        # Selection logic
        select_ind = self.selection_logic(get_ind, get_wave, gen)
#         print("          I'm loading frame {}".format(select_ind), flush=True)

        # Load the Frame
        this_frame =    self.color_frames[select_ind][0]
        this_wave = int(self.color_frames[select_ind][1])
        frame_shape = this_frame.shape
        # Set the Frame
        self.params.modified_image = this_frame+0
        self.params.raw_image = this_frame+0
        
#         stem = "         Loaded {}A frame of size {}, idx={}"
#         print(stem.format(this_wave, frame_shape, select_ind))
#         self.params.set_current_wave(this_wave) # This is too powerful
        
        # Prep the Metadata
        wave1 = self.current_wave = this_wave
        shape = self.params.modified_image.shape
        fits_path = None
        t_rec1 = self.time_stamp
        
        # Store the Metadata
        self.image_data = self.params.image_data = wave1, fits_path, t_rec1, shape
        if peek:
            self.peek_selection()
        
        return self.params.modified_image


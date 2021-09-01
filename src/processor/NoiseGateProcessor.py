from astropy.io import fits
import numpy as np
from processor.Processor import Processor
from os import listdir, makedirs
from os.path import join, abspath
from time import time, strftime
import matplotlib.pyplot as plt
from noisegate import tools as ngt

plt.ion()

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from processor.Processor import Processor

from science.SRNFilter import SRNFilter

norm = SRNFilter.normalize

small_fill = 100


class NoiseGateProcessor(Processor):
    
    def __init__(self, p):
        self.params = p
        
        self.local_wave_directory = None
        self.image_folder = None
        self.movie_folder = None
        self.video_name_stem = None
    
    def process(self):
        """loads fits images and then performs the noise gating on them"""
        for self.wave in self.params.use_wavelengths:
            if not self.wave in self.params.do_one():
                continue
            self.load_fits_images()
            self.noise_gate()
            self.save_cubes()
    
    def load_fits_images(self):
        print("Loading {} Fits Files...".format(self.wave))
        self.prepare_to_load()
        self.allocate_cubes()
        self.fill_cubes()
        print("Cube has been loaded!", flush=True)
    
    def prepare_to_load(self):
        self.build_paths()
        self.sample_frame = self.load_file(self.im_paths[0])
        self.height, self.width = self.sample_frame.shape
        self.number = len(self.im_paths)
    #
    # def build_paths(self):
    #     self.local_wave_directory = join(self.params.imgs_directory(), self.wave)
    #     self.image_folder = join(self.local_wave_directory, 'png')
    #     self.fits_folder = join(self.local_wave_directory, 'fits')
    #     self.movie_folder = abspath(join(self.params.imgs_directory(), "movies\\"))
    #     self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(self.wave, strftime('%m%d_%H%M'), '{}'))
    #     self.im_paths = [join(self.fits_folder, img) for img in listdir(self.fits_folder) if img.endswith(".fits")]
    #     makedirs(self.movie_folder, exist_ok=True)
    #
    def allocate_cubes(self):
        try:
            self.cube
            self.gated_cube
        except AttributeError:
            self.cube = np.empty((self.number, self.width, self.height), dtype=type(self.sample_frame[0, 0]))
            self.gated_cube = self.cube + 0
        self.cube.fill(np.nan)
        self.gated_cube.fill(np.nan)
    
    def fill_cubes(self):
        for ii, img in enumerate(tqdm(self.im_paths)):
            frame = self.load_file(img)
            if frame is not None and frame.shape == self.sample_frame.shape:
                self.cube[ii] = frame
            
            if ii > small_fill:
                break
    
    def save_cubes(self):
        for ii, img in enumerate(tqdm(self.im_paths)):
            self.save_file(img, self.cube[ii])
            # self.load_file(img)
            if ii > small_fill:
                break
        
    def noise_gate(self):
        print("Beginning Noise Gating Procedure...", end='')
        max_use = 20
        n_chunks = int(self.number // max_use)
        for ii in tqdm(np.arange(n_chunks)):
            start = (ii) * max_use
            end = (ii + 1) * max_use
            cubie = self.cube[start:end]
            out = ngt.noise_gate_batch(cubie, cubesize=12, model='hybrid', factor=2.0)
            self.gated_cube[start:end] = out
            break
            if ii * max_use > small_fill:
                break
            # for jj in np.arange(start=start, stop=end):
            #     # fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
            #     # ax1.imshow(norm(self.cube[jj]))
            #     # ax2.imshow(norm(self.gated_cube[jj]))
            #     # plt.show(block=True)
            #
            #     fig, (ax) = plt.subplots(1, 1)
            #     plt.title(jj)
            #     ax.imshow(self.gated_cube[jj] - self.cube[jj])
            #     plt.show(block=True)
        print("Noise Gating Complete")
        
        #
        #     if len(images) > 0:
        #
        #
        #
        #
        #         in_object = cv2.imread(join(self.image_folder, images[0]))
        #         height, width, layers = in_object.shape
        #         final_name = self.video_name_stem.format("_raw.avi")
        #         print(final_name)
        #         video_avi = cv2.VideoWriter(final_name, 0, self.params.frames_per_second(), (width, height))
        #
        #         for in_object in tqdm(images, desc=">Noise Gating {}".format(current_wave), unit="in_object"):
        #             # print(join(self.image_folder, in_object))
        #             im = cv2.imread(join(self.image_folder, in_object))
        #             video_avi.write(im)
        #
        #         cv2.destroyAllWindows()
        #         video_avi.release()
        #
        #     else:
        #         print("No png Images Found")
        # except FileNotFoundError:
        #     print("Images Not Found")

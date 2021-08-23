from os import listdir, makedirs
from os.path import join, abspath
from time import time, strftime

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from processor.Processor import Processor


class VideoProcessor(Processor):
    def __init__(self, p):
        self.params = p
        
        self.local_wave_directory = None
        self.image_folder = None
        self.movie_folder = None
        self.video_name_stem = None
        
        # name = "{}_{}".format(self.this_name, "max")
        # self.soni = Sonifier(self.params, self.save_path, name, self.video_name_stem, frames_per_second=self.params.frames_per_second())
        
        # print("\nMovie: {}".format(self.this_name))
        
    def process(self):
        """Combines all png files into an avi movie"""
        
        for wave in self.params.use_wavelengths:
            if not wave in self.params.do_one():
                continue
            self.build_paths(wave)
            try:
                images = [img for img in listdir(self.image_folder) if img.endswith(".png")] # and self.check_valid_png(img)]
                if len(images) > 0:
                    frame = cv2.imread(join(self.image_folder, images[0]))
                    height, width, layers = frame.shape
                    final_name = self.video_name_stem.format("_raw.avi")
                    print(final_name)
                    video_avi = cv2.VideoWriter(final_name, 0, self.params.frames_per_second(), (width, height))
                    
                    for image in tqdm(images, desc=">Writing Movie {}".format(wave), unit="frame"):
                        # print(join(self.image_folder, image))
                        im = cv2.imread(join(self.image_folder, image))
                        video_avi.write(im)
                    
                    cv2.destroyAllWindows()
                    video_avi.release()
                
                else:
                    print("No png Images Found")
            except FileNotFoundError:
                print("Images Not Found")
    
    def build_paths(self, wave):
        self.local_wave_directory = join(self.params.img_directory(), wave)
        self.image_folder = join(self.local_wave_directory, 'png')
        self.movie_folder = abspath(join(self.params.img_directory(), "movies\\"))
        self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(wave, strftime('%m%d_%H%M'), '{}'))
        # print(self.video_name_stem)
        makedirs(self.movie_folder, exist_ok=True)
        
        # try:
        #     videoclip_full = VideoFileClip(self.video_name_stem.format("_raw.avi"))
        #     invalid_movie=False
        # except:
        #     invalid_movie=True
        #
        # if True: #  self.new_images or invalid_movie:
        #     # logger = open(join(self.params.local_directory, 'log.txt'), 'w+')

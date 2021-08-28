from os import makedirs
from os.path import join, dirname
from time import strftime
import cv2
from tqdm import tqdm
from processor.Processor import Processor
"""This Processor is used to turn a set of images into a video"""


class VideoProcessor(Processor):
    
    filt_name = '  Video Writer'
    out_name = "_raw.avi"
    do_png = True
    wave = None
    progress_stem = "    Writing Movie {}"
    progress_text = ""
    video_name_stem = ""
    description = "Turn all the imgs into an AVI video"
    
    def process_one_wavelength(self, wave):
        """Prepare and execute the video writer"""
        video_avi = self.prep_video_writer(wave)
        if video_avi:
            self.run_video_writer(video_avi)
            print("   Done\n")

    def prep_video_writer(self, wave):
        """Build all the paths and initialize everything"""
        self.load(self.params)
        self.wave = wave
        if len(self.params.local_imgs_paths()) > 0:
            frame = cv2.imread(self.params.local_imgs_paths()[0])
            height, width, layers = frame.shape
            video_name_stem = join((self.params.movs_directory()), '{}_{}_movie{}'.format(wave, strftime('%m%d_%H%M'), '{}'))
            final_name = video_name_stem.format(self.out_name)
            makedirs(dirname(final_name), exist_ok=True)
            video_avi = cv2.VideoWriter(final_name, 0, self.params.frames_per_second(), (width, height))
            self.progress_text = self.progress_stem.format(self.wave)
            return video_avi
        else:
            print("    No Files Found \n")
            return False
    
    def run_video_writer(self, video_avi):
        """Generate the video file"""
        for image in tqdm(self.params.local_imgs_paths(), desc=self.progress_text, unit="frame"):
            im = cv2.imread(image)
            video_avi.write(im)
        cv2.destroyAllWindows()
        video_avi.release()
        print("      Complete!")

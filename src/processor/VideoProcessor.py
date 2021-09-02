from os import makedirs, listdir
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
    progress_stem = " *    Writing Movie {}"
    progress_text = ""
    video_name_stem = ""
    description = "Turn all the imgs into an AVI video"
    
    def __init__(self):
        super().__init__()
    
    def process_one_wavelength(self, wave):
        """Prepare and execute the video writer"""
        if self.params.write_video():
            video_avi = self.prep_video_writer(wave)
            self.run_video_writer(video_avi)
        else:
            print(" ^    Skipped")

    # def skip_video(self):
    #     if :
    #         # If you do want to overwrite
    #         return False # Don't Skip
    #     else:
    #         # If you don't want to overwrite
    #         if self.final_name in listdir(self.params.movs_directory()):
    #             # Make images you don't already have
    #             return True # do skip
    #         else:
    #             return False # don't skip
    
    def prep_video_writer(self, wave):
        """Build all the paths and initialize everything"""
        fits_paths, imgs_paths = self.load(self.params)
        self.wave = wave
        if len(self.params.local_imgs_paths()) > 0:
            frame = cv2.imread(self.params.local_imgs_paths()[0])
            height, width, layers = frame.shape
            video_name_stem = join((self.params.movs_directory()),
                                   '{}_{}_movie{}'.format(wave, strftime('%m%d_%H%M'), '{}'))
            final_name = video_name_stem.format(self.out_name)
            
            makedirs(dirname(final_name), exist_ok=True)
            video_avi = cv2.VideoWriter(final_name, 0, self.params.frames_per_second(), (width, height))
            self.progress_text = self.progress_stem.format(self.wave)
            self.final_name = final_name
            return video_avi
        else:
            print("    No Files Found \n")
            return False
    
    def run_video_writer(self, video_avi):
        """Generate the video file"""
        
        good_paths = [pp for pp in self.params.local_imgs_paths() if ('orig' not in pp and 'cat' not in pp)]
        
        if len(good_paths) == 0:
            print("    No Files Found \n")
            return False
        
        ii = 0
        for img_path in tqdm(good_paths, desc=self.progress_text, unit="frames"):
            if 'orig' not in img_path and 'cat' not in img_path:
                video_avi.write(cv2.imread(img_path))
                ii += 1
        cv2.destroyAllWindows()
        video_avi.release()
        print(" ^    Successfully Wrote Movie from {} images!".format(ii))
    

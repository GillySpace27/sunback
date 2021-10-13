import os
from os import makedirs, listdir
from os.path import join, dirname, abspath
from time import strftime
import cv2
from tqdm import tqdm
from processor.Processor import Processor

"""This Processor is used to turn a set of images into a video"""


class VideoProcessor(Processor):
    mov_suffix = "raw"
    mov_type = "avi"
    filt_name = 'Video Writer'
    destroy = False
    do_png = True
    wave = None
    progress_stem = " *    {}"
    progress_verb = 'Writing Movie'
    progress_string = progress_stem.format(progress_verb)
    finished_verb = "Wrote Movie"
    progress_unit = "imgs"
    progress_text = progress_string
    
    video_name_stem = ""
    description = "Turn all the imgs into an AVI video"
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.frame_shape = None
        self.good_paths = []
        self.skipped = 0
        self.final_output_path = None

    
    def process_one_wavelength(self, wave):
        """Prepare and execute the video writer"""
        video_avi =     self.prep_video_writer(wave)
        if video_avi:   self.run_video_writer(video_avi)
    
    def prep_video_writer(self, wave):
        """Build all the paths and initialize everything"""
        self.load(self.params, wave=wave)
        if self.n_fits:
            self.build_output_paths()

            if not self.should_continue():
                return False
            
            # Make the Directory
            makedirs(dirname(self.final_output_path), exist_ok=True)
            
            # Make the VideoWriter and return it
            video_avi = cv2.VideoWriter(self.final_output_path, 0, self.params.frames_per_second(), self.frame_shape)
            return video_avi
        
        else:  # If there are no files then sad
            print("    No Files Found \n")
            return False
    
    def build_output_paths(self, path_box=None):
        """Build the Path to the Video"""
        # Parse Inputs
        if path_box is None:
            path_box = self.params.local_imgs_paths()
        height, width, _ = cv2.imread(path_box[0]).shape
        self.frame_shape = (width, height)
        self.good_paths = [path for path in path_box if ('orig' not in path and 'cat' not in path)]
        
        # Build File Name
        batch_name = self.params.config['name']
        wave = self.params.current_wave()
        time_now = strftime('%m%d_%H%M')
        file_name = '{}_{}_video_{}.{}'.format(batch_name, wave, time_now, self.mov_suffix, self.mov_type)
        self.final_output_path = join(self.params.movs_directory(), file_name)
        self.progress_text = self.progress_stem.format(self.wave)
        
        
    def should_continue(self):
        """Skip the video writing if indicated"""
        if os.path.exists(self.final_output_path) and \
                not (self.params.write_video() or self.reprocess_mode()):
            print(" ^    Skipped \n")
            return False
        
        # Find the Good Frames
        if len(self.good_paths) == 0:
            print(" ^    No Good Files Found \n")
            return False
        return True
    
    def run_video_writer(self, video_avi):
        """Generate the video file"""
        ii = 0
        self.skipped = 0
        for img_path in tqdm(self.good_paths, desc=self.progress_text, unit="frames"):
            if 'orig' not in img_path and 'cat' not in img_path:
                if self.reprocess_mode() or True:  # TODO THis is a like truth
                    video_avi.write(cv2.imread(img_path))
                    ii += 1
                else:
                    self.skipped += 1
            if self.destroy:
                os.remove(img_path)
        cv2.destroyAllWindows()
        video_avi.release()
        print(" ^    Successfully {} from {} images! ({} skipped)".format(self.finished_verb, ii, self.skipped))



        
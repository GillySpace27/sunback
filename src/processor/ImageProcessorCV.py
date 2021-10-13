import os
import shutil
import cv2
from processor.ImageProcessor import ImageProcessor
import numpy as np


class ImageProcessorCV(ImageProcessor):
    filt_name = 'CV Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing Images"
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.img_frame = None
        self.out_path = None
    
    def do_fits_function(self, fits_path, in_name=None):
        """ Main Call on the Fits Path """
        self.init_frame(fits_path, self.params.png_frame_name)
        self.render_all()
        return self
    
    def render_all(self):
        """Render one image"""
        self.plot_aia_original()
        self.plot_aia_changed()
        self.save_concatinated(destroy=True)
    
    def plot_aia_original(self):
        """Plot the original data from AIA"""
        # Get the Frame and Path
        self.frame = np.flipud(self.original)
        self.out_path = self.get_original_path()
        self.execute_save()
    
    def plot_aia_changed(self):
        """Plot the changed data from AIA"""
        # Get the Frame and Path
        self.frame = np.flipud(self.changed)
        self.out_path = self.get_changed_path()
        self.execute_save()
    
    def execute_save(self):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.make_image()
        self.label_plot()
        self.img_save(self.out_path)
        self.path_box.append(self.out_path)
    
    def make_image(self):
        self.img_frame = (self.cmap(self.frame)[:, :, :3] * 255).astype(np.uint8)
    
    def img_save(self, path):
        b, g, r = cv2.split(self.img_frame)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        cv2.imwrite(path, rgb_img)
    
    def label_plot(self):
        """Annotate with Text"""
        img = self.img_frame
        full_name, fits_path, time_string_raw, shape = self.image_data
        time_string = self.clean_time_string(time_string_raw)
        time_list = time_string.split()
        
        inst = 'AIA'
        _, wave = self.clean_name_string(full_name)
        clock = time_list[1].lower()
        day = time_list[0][:-5]
        year = time_list[0][-4:]
        
        cv2.putText(img, inst, (3900, 100), 0, 3, (255, 255, 255), 3)
        cv2.putText(img, wave, (3875, 200), 0, 3, (255, 255, 255), 3)
        cv2.putText(img, clock,   (0, 100), 0, 3, (255, 255, 255), 3)
        cv2.putText(img, day,     (0, 200), 0, 3, (255, 255, 255), 3)
        cv2.putText(img, year,    (0, 300), 0, 3, (255, 255, 255), 3)
    
    def cleanup(self):
        destroy = False
        try:
            self.write_video_in_directory(fullpath=self.cat_path, file_name="concatinated.avi", fps=5, destroy=destroy)
        except FileNotFoundError as e:
            print(e)
        if destroy:
            shutil.rmtree(self.orig_directory)
            
    @staticmethod
    def peek_frame(img):
        shrink = 5
        cv2.imshow("win2", img[::shrink, ::shrink, ::shrink])
        cv2.waitKey(0)

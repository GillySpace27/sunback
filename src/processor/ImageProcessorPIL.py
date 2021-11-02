import os
from processor.ImageProcessor import ImageProcessor
import numpy as np
from PIL import Image


class ImageProcessorPIL(ImageProcessor):
    filt_name = 'PIL Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing Images"
    cat_path = None
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.out_path = None
    
    def cleanup(self):
        self.write_video_in_directory(fullpath=self.cat_path, file_name="concatinated.avi", fps=5, destroy=False)
        # self.write_video_in_directory(fullpath=self.png_save_path, file_name="concatinated.avi", fps=5, destroy=True)
        
    def do_fits_function(self, fits_path, in_name=None):
        """This is the do_fits_function for this """
        self.init_frame(fits_path, self.params.png_frame_name)
        self.render_all()
        return self
    
    def render_all(self):
        """Render one image"""
        self.plot_aia_original()
        self.plot_aia_changed()
        self.save_concatinated(destroy=False)
        
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
        self.path_box.append(self.out_path)
        self.execute_save()
    
    def execute_save(self):
        self.path_box.append(self.out_path)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.get_img().save(self.out_path)
    
    def get_img(self):
        colored_array = (self.cmap(self.frame)[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(colored_array)


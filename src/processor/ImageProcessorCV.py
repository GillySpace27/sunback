import os
import shutil
import cv2
from processor.ImageProcessor import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessorCV(ImageProcessor):
    filt_name = 'CV Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing Images"
    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.img_frame = None
        self.out_path = None
        # self.write_video_in_directory(fullpath=r"D:\sunback_images\MultiRange\Liftoff 0171_l_2013_09_29_000001\0171\png\cat", file_name="concatinated.avi", fps=15, destroy=False)
        
    
    def do_fits_function(self, fits_path, in_name=None):
        """ Main Call on the Fits Path """
        self.init_frame(fits_path, self.params.png_frame_name)
        self.render_all()
        return self
    
    def do_img_function(self):
        """ Main Call on the Fits Path """

        self.init_image_frame()
        self.plot_two()
        
        # self.display_all()
        return self
    
    def display_all(self):
        self.display_original()
        self.display_changed()
        
    def display_original(self):
        print("Original")
        self.frame = np.flipud(self.params.original_image)
        self.prep_save()
        plt.imshow(self.frame)
        plt.title("Original")
        plt.show(block=True)
        
    def display_changed(self):
        print("Changed")
        self.frame = np.flipud(self.params.modified_image)
        self.prep_save()
        plt.imshow(self.frame)
        plt.title("Changed")
        plt.show(block=True)
        
    def render_all(self):
        """Render one image"""
        self.plot_aia_original()
        self.plot_aia_changed()
        self.save_concatinated(destroy=True)
    
    def plot_aia_original(self):
        """Plot the original_image data from AIA"""
        # Get the Frame and Path
        self.frame = np.flipud(self.params.original_image)
        self.out_path = self.get_original_path()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        
        self.prep_save()
        self.img_save(self.out_path)
        
    
    def plot_aia_changed(self):
        """Plot the modified_image data from AIA"""
        # Get the Frame and Path
        self.frame = np.flipud(self.params.modified_image)
        self.out_path = self.get_changed_path()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        
        self.prep_save()
        self.img_save(self.out_path)
        
    
    def prep_save(self):
        self.make_image()
        self.label_plot()
        self.path_box.append(self.out_path)
    
    def make_image(self):
        out = self.frame + 0
        maxmax = np.nanpercentile(out, 99)
        minmin = np.nanpercentile(out, 1)
        if maxmax > 100:
            out = (self.frame-minmin)/(maxmax-minmin)
            # print("\nRenormalizing", maxmax, minmin, np.max(out), np.min(out))
            
        self.img_frame = (self.params.cmap(out)[:, :, :3] * 255).astype(np.uint8)
        b, g, r = cv2.split(self.img_frame)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        self.params.rbg_image = rgb_img
        
    
    def img_save(self, path, save=True):
        if save:
            cv2.imwrite(path, self.params.rgb_img)
        else:
            cv2.imshow(self.params.rbg_image)
            
        
    
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

        x0 = 3900
        x1 = 3875
        scale = 3
        h1=100
        h2=200
        h3=300
        if shape[0] < 3000:
            x0 = x0 // 4
            x1 = x1 // 4
            scale = 1
            h1=40
            h2=80
            h3=120
        
        cv2.putText(img, inst, (x0, h1), 0,   scale, (255, 255, 255), 3)
        cv2.putText(img, wave, (x1, h2), 0,   scale, (255, 255, 255), 3)
        cv2.putText(img, clock,   (0, h1), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, day,     (0, h2), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, year,    (0, h3), 0, scale, (255, 255, 255), 3)
    
    def cleanup(self):
        destroy = False
        try:
            self.write_video_in_directory(fullpath=self.cat_path, file_name="concatinated.avi", fps=5, destroy=destroy)
        except (FileNotFoundError, AttributeError) as e:
            print(e)
        if destroy:
            shutil.rmtree(self.orig_directory)
            
    @staticmethod
    def peek_frame(img):
        shrink = 5
        cv2.imshow("win2", img[::shrink, ::shrink, ::shrink])
        cv2.waitKey(0)

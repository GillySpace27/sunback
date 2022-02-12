import os
import shutil
import cv2
from processor.ImageProcessor import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessorCV(ImageProcessor):
    filt_name = 'CV Image Writer'
    description = "Turn all the fits files into png files"
    progress_verb = "Writing"
    progress_unit = "Images"

    
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.frame_name = None
        self.img_frame = None
        self.out_path = None
        self.in_name = -1
        
    
    def do_fits_function(self, fits_path, in_name=None):
        """ Main Call on the Fits Path """
        self.init_frame(fits_path, self.params.png_frame_name)
        self.render_all()
        return self
    
    def do_img_function(self):
        """ Main Call on the Fits Path """
        if False:
            self.plot_two()
            self.plot_two("Less Zoomed", True)
            # self.display_all()
    
        # self.init_image_frame()
        raise NotImplementedError
    
    def display_all(self):
        self.display_raw()
        self.display_changed()
        
    def display_raw(self):
        print("LEV1")
        self.frame = np.flipud(self.params.raw_image)
        self.prep_save()
        plt.imshow(self.frame)
        plt.title("LEV1")
        plt.show(block=True)
        
    def display_changed(self):
        print("Changed")
        self.frame = np.flipud(self.params.modified_image)
        self.prep_save()
        plt.imshow(self.frame)
        plt.title("Changed")
        plt.show(block=True)
        
    def render_all(self):
        """Render one image_path"""
        self.plot_aia_raw()
        self.plot_aia_changed()
        # self.save_concatinated()
        
        # self.do_shortcut()
        
    def do_shortcut(self):
        cat_png_path = self.cat_path
        root_folder = os.path.dirname(self.params.base_directory())
        fits_folder = os.path.dirname(self.params.use_image_path())
        cat_png_filename = os.path.basename(cat_png_path)
        shorts_folder =  os.path.join(root_folder, "shorts")
        # short_path = os.path.join(shorts_folder, cat_png_filename.replace(".png", ".lnk"))
        
        
        timestamp = self.image_data[2]
        short_path = os.path.join(shorts_folder, "{}_{}.png".format(self.params.current_wave(), timestamp.split('.')[0]))
        os.makedirs(shorts_folder, exist_ok=True)

        src_file  =  cat_png_path
        dest_file =  os.path.normpath(short_path)
        shutil.copyfile(src_file, dest_file, follow_symlinks=True)
        # self.make_shortcut(src_file,dest_file , False)
    
    def plot_aia_raw(self):
        """Plot the raw_image data from AIA"""
        # Get the Frame and Path
        # self.frame_name = "t_integrated"
        self.frame_name = ["LEV1p5_T", "LEV1p5_L", "T_Integrated", "LEV1"]
        
        frame, wave, t_rec, center, int_time = self.load_a_fits_field(self.fits_path, self.frame_name)
        self.frame = np.flipud(frame)
        # self.frame = np.flipud(self.params.raw_image)
        self.out_path = self.get_raw_path()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        
        self.prep_save()
        self.img_save(self.out_path)
        
    
    def plot_aia_changed(self):
        """Plot the modified_image data from AIA"""
        # Get the Frame and Path
        self.frame_name = self.params.png_frame_name #.hdu_name_list[-1]
        self.frame = np.flipud(self.params.modified_image)
        self.out_path = self.get_changed_path()
        out_dir = os.path.dirname(self.out_path)
        os.makedirs(out_dir, exist_ok=True)
        print("Saving to {}".format(self.out_path))
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
        themax = np.nanmax(out)
        
        if themax > 100 or themax < 0.8:
            out = (self.frame-minmin)/(maxmax-minmin)
            print("\nRenormalizing", maxmax, minmin, np.max(out), np.min(out))
            
        self.img_frame = (self.params.cmap(out)[:, :, :3] * 255).astype(np.uint8)
        b, g, r = cv2.split(self.img_frame)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        self.params.rbg_image = rgb_img
        
    
    def img_save(self, path, save=True):
        if save:
            cv2.imwrite(path, self.params.rbg_image)
        else:
            # cv2.imshow(mat=self.params.rbg_image)
            plt.imshow(self.params.rbg_image)
            plt.show()
            
    def label_plot(self):
        """Annotate with Text"""
        # img = self.img_frame
        img = self.params.rbg_image
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

        # if self.params.alpha is not None:
        #     cv2.putText(img, "a={:0.3f}".format(self.params.alpha), (int(x0*0.95), h3), 0,   scale, (255, 255, 255), 3)

        
        if type(self.frame_name) is list:
            frame_name = [x for x in self.frame_name if x.casefold() in self.hdu_name_list][0]
        else:
            frame_name = self.frame_name
            
        
        cv2.putText(img, frame_name, (int(x0*0.94), h3), 0,   scale, (255, 255, 255), 3)
        
        reticle = False
        if reticle:
            cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                   int(self.params.header["R_SUN"]), (255,255,255), 3)
            cv2.circle(img, (int(self.params.header["X0_MP"]), int(self.params.header["Y0_MP"])),
                   int(10), (255,0,0), 10)
            
        cv2.putText(img, inst, (x0, h1), 0,   scale, (255, 255, 255), 3)
        cv2.putText(img, wave, (x1, h2), 0,   scale, (255, 255, 255), 3)
        cv2.putText(img, clock,   (0, h1), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, day,     (0, h2), 0, scale, (255, 255, 255), 3)
        cv2.putText(img, year,    (0, h3), 0, scale, (255, 255, 255), 3)
    
    def cleanup(self):
        # self.make_intermediate_videos()
        pass
    
    def make_intermediate_videos(self):
        try:
            print("Writing Video...", end='')
            radial_hist_path = "analysis\\radial_hist_post"
            hist_path_0 = os.path.join(self.params.base_directory(), radial_hist_path)
            hist_path_1 = hist_path_0[:-5]
            
            n_hist_0 = len(os.listdir(hist_path_0))
            n_hist_1 = len(os.listdir(hist_path_1))
            
            if n_hist_0:
                self.write_video_in_directory(directory=hist_path_0, fps=15, destroy=False)
            if n_hist_1:
                self.write_video_in_directory(directory=hist_path_1, fps=15, destroy=False)
            if self.params.do_cat:
                self.write_video_in_directory(directory=self.params.cat_directory, file_name="concatinated.avi", fps=15, destroy=False)
            print("Success!")
        except (FileNotFoundError, AttributeError) as e:
            print("ImageProcessorCV")
            raise(e)
        
          # destroy = False
          # if destroy:
          #     shutil.rmtree(self.orig_directory)
    
        
    @staticmethod
    def peek_frame(img):
        shrink = 5
        cv2.imshow("win2", img[::shrink, ::shrink, ::shrink])
        cv2.waitKey(0)

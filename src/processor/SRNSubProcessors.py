import shutil
from time import time

from processor.SRNProcessor import SRNProcessor
import os

##########################
##########################
## Child Classes of SRN ##
##########################
##########################
from processor.VideoProcessor import VideoProcessor


class SRNSingleShotProcessor(SRNProcessor):
    out_name = 'SRN'
    name = filt_name = 'SRN Single Shot Processor'
    description = "Create and Apply the Radial SRN Curves"
    progress_verb = 'Processing'
    finished_verb = "Modified"
    show_plots = True
    
    def __init__(self, fits_path=None, in_name=-1, orig=False,
                 show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
        self.first = True
        self.go_ahead = True
        self.params.current_wave('rainbow')
        self.params.Force_init = True
        self.can_use_keyframes = False
        self.can_initialize = False
    
    def setup(self):
        # self.load(self.params, quietly=True, wave=self.params.current_wave('rainbow'))
        # self.params.current_wave('rainbow')

        # self.params.fits_directory()
        pass
    
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        verb =True
        if verb:
            print(" v ", self.progress_verb, "Image...")
        self.image_learn()  # Analyze the input to help make normalization curves
#         self.plot_norm_curves(save=False, show=True, extra=True)
        self.image_modify()  # Actually Normalize This Image
        self.first=False
        self.plot_full_normalization(save=True, show=False, do=True)
        print(" ^ Success!\n")
        return self.params.modified_image
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        # print("Save/load!")
        self.save_curves(banner=False)
        self.load_curves()
        pass


class SRNpreProcessor(SRNProcessor):
    """Analyzes the whole dataset and builds curves"""
    out_name = None
    name = filt_name = 'SRN Pre-Processor'
    description = "Create the Radial SRN Curves"
    progress_verb = 'Analyzing'
    finished_verb = "Analyzed"
    show_plots = True
    
    def __init__(self, fits_path=None, in_name=-1, orig=False,
                 show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path=fits_path, in_name=in_name, orig=orig, show=show, verb=verb, quick=quick, rp=rp, params=params)
        self.first = True
        self.go_ahead = True
    
    def setup(self):
        self.load()
        self.print_keyframes()
        self.skipped = 0
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.should_run():
            self.image_learn()
            self.plot_norm_curves(save=True)
        return None
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        if self.should_run():
            self.skipped -= 1
            self.make_save_smoothed_curves(banner=False)  # Build smooth curves based on the statistics
        self.render_pre_hist_video()
        # print("Curves Saved!")
        
        
    def render_pre_hist_video(self):
        fps = 8
        os.makedirs(self.params.base_directory(), exist_ok=True)
        print("Rendering pre-processor video...", end='')
        path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist_pre\\a-pre-hist.avi")
        self.write_video_in_directory(fullpath=path1, fps=fps, key_string="inner", destroy=False, pop=2)
        
        # path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist_pre\\{}_inner_outer_{}.avi".format(self.params.current_wave(), time()))
        # path2 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\zoom\\{}_zoom_{}.avi".format(self.params.current_wave(), time()))
        
        # directory1 = os.path.dirname(path1)
        # name1 = os.path.basename(path1)
        # directory2 = os.path.dirname(path2)
        # name2 = os.path.basename(path2)
        
        # self.write_video_in_directory(fullpath=path2, fps=fps, key_string="zoom" , destroy=False)
        
        # self.delete_temp_folder_items(os.path.dirname(path1))
        # self.delete_temp_folder_items(os.path.dirname(path1))
        print("Success!")
    
    def should_run(self):
        """Decide of the processor should run on this file"""
        self.can_use_keyframes = True
        not_dark = self.header["IMG_TYPE"] == "LIGHT"
        not_weak = self.header["EXPTIME"] > 1.0
        set_to_make = self.params.remake_norm_curves() or self.reprocess_mode()
        not_made_yet = not os.path.exists(self.params.curve_path()) or self.outer_min is None
        frame_is_not_loaded = self.params.original_image is None
        self.go_ahead = not_weak & not_dark and (set_to_make or not_made_yet or frame_is_not_loaded)
        return self.go_ahead


    # def delete_temp_folder(self, folder):
    #     if os.path.isdir(folder):
    #         shutil.rmtree(folder)
    #
    # def delete_temp_folder_items(self, folder):
    #     for root, dirs, files in os.walk(folder):
    #         for file in files:
    #             self.force_delete(file, root)

    @staticmethod
    def force_delete(file, root='', do=True):
        if do:
            if not os.path.isdir(file):
                os.remove(os.path.join(root, file))
            else:
                shutil.rmtree(file)
    
    
    
    
    
    
    
class SRNradialFiltProcessor(SRNProcessor):
    """Uses radial curves to normalize images"""
    name = out_name = 'SRN'
    filt_name = 'SRN Radial Filter'
    description = "Filter the Images Radially with SRN"
    progress_verb = 'Filtering'
    progress_unit = 'Images'
    finished_verb = "Filtered"
    
    def __init__(self, fits_path=None, in_name=-1, orig=False,
                 show=False, verb=False, quick=False, rp=None, params=None):
    
        super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
        self.show_norm = False
        self.first = True
        self.go_ahead = True
        self.can_use_keyframes = False
    
    def setup(self):
        self.super_flush()
        self.load_curves()
    
    def do_work(self):
        self.image_modify()
        # self.peek_norm()
        self.show_norm=False
        self.plot_full_normalization(True, show=self.show_norm, save=True)
        self.percentilize()
        return self.params.modified_image
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        self.render_post_hist_video()
        print(" ^ Filter Applied Successfully", flush=True)


    def render_post_hist_video(self):
        print("Rendering post-processor video...", end='')
        fps = 8
        os.makedirs(self.params.base_directory(), exist_ok=True)
        path1 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_post\\b-post-hist.avi")
        self.write_video_in_directory(fullpath=path1, fps=fps, destroy=False, pop=2)
        
        
        # path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist_pre\\{}_inner_outer_{}.avi".format(self.params.current_wave(), time()))
        # path2 = os.path.join(self.params.base_directory(), "analysis\\radial_hist_pre\\zoom\\{}_zoom_{}.avi".format(self.params.current_wave(), time()))
        
        # directory1 = os.path.dirname(path1)
        # name1 = os.path.basename(path1)
        # directory2 = os.path.dirname(path2)
        # name2 = os.path.basename(path2)
        
        # self.write_video_in_directory(fullpath=path2, fps=fps, key_string="zoom" , destroy=False)
        
        # self.delete_temp_folder_items(os.path.dirname(path1))
        # self.delete_temp_folder_items(os.path.dirname(path1))
        print("Success!")
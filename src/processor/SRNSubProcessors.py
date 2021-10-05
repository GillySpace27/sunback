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
    finished_verb = "Applied"
    show_plots = True
    
    def __init__(self, fits_path=None, in_name=-1, orig=False,
                 show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
        self.first = True
        self.go_ahead = True
    
    def setup(self):
        pass
    
    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        self.image_learn()  # Analyze the input to help make normalization curves
        self.image_modify()  # Actually Normalize This Image
        return self.changed
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        # print("Save/load!")
        # self.save_curves()
        # self.load_curves()
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
            
            self.plot_inner_outer(save=True)
            # self.plot_radial_norm_keyframes(do=True, show=False, save=True)
        return None
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        if self.should_run():
            self.skipped -= 1
            self.save_curves()
            self.make_smoothed_curves()  # Build smooth curves based on the statistics
            self.save_curves()
        self.render_inner_outer_video()
        
        
    def render_inner_outer_video(self):
        fps = 8
        os.makedirs(self.params.base_directory(), exist_ok=True)
        
        path1 = os.path.join(self.params.base_directory(),"analysis\\radial_hist\\{}_inner_outer_{}.avi".format(self.params.current_wave(), time()))
        path2 = os.path.join(self.params.base_directory(), "analysis\\radial_hist\\zoom\\{}_zoom_{}.avi".format(self.params.current_wave(), time()))
        
        # directory1 = os.path.dirname(path1)
        # name1 = os.path.basename(path1)
        # directory2 = os.path.dirname(path2)
        # name2 = os.path.basename(path2)
        
        self.write_video_in_directory(fullpath=path1, fps=fps, key_string="inner", destroy=True)
        self.write_video_in_directory(fullpath=path2, fps=fps, key_string="zoom" , destroy=True)
        
        # self.delete_temp_folder_items(os.path.dirname(path1))
        # self.delete_temp_folder_items(os.path.dirname(path1))
    
    def should_run(self):
        """Decide of the processor should run on this file"""
        self.can_use_keyframes = True
        set_to_make = self.params.remake_norm_curves() or self.reprocess_mode()
        not_made_yet = not os.path.exists(self.params.curve_path()) or self.outer_min is None
        frame_is_not_loaded = self.original is None
        self.go_ahead = set_to_make or not_made_yet or frame_is_not_loaded
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
    finished_verb = "Filtered"
    
    def __init__(self, fits_path=None, in_name=-1, orig=False,
                 show=False, verb=False, quick=False, rp=None, params=None):
        super().__init__(fits_path, in_name, orig, show, verb, quick, rp, params)
        self.first = True
        self.go_ahead = True
        self.can_use_keyframes = False
    
    def setup(self):
        self.load_curves()
        self.super_flush()
    
    def do_work(self):
        self.image_modify()
        self.plot_radial_norm_keyframes(True, show=False, save=True)
        

        return self.changed
    
    def cleanup(self):
        """Runs after all the images have been modified with do_work"""
        pass
        # folder_name = os.path.abspath(os.path.join(self.params.base_directory(), "analysis\\radial_hist_full"))
        # self.write_video_in_directory(fullpath=folder_name, file_name="full_hist.avi", fps=5, destroy=False)

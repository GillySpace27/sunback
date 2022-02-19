import shutil
from time import time

from processor.SRNProcessor import SRNProcessor
from fetcher.LocalFetcher import LocalCdfFetcher
import os
import matplotlib.pyplot as plt

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
        self.frame_list = []

    
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
        
        if self.params.use_cdf:
            return self.run_cdf()
        else:
            return self.run_single()
            
    def run_single(self):
        """Run the program on a single loaded frame"""
        self.image_learn()  # Analyze the input to help make normalization curves
        self.image_modify()  # Actually Normalize This Image
        self.image_plot()
        self.first=False
        self.plot_full_normalization(save=True, show=False, do=True)
        print(" ^ Success!\n")
        return self.params.modified_image

        
    def run_cdf(self, do_plot=False):
        """Run the program on a single loaded frame"""

        for index in range(self.params.n_frames):
#             print("Frame {}".format(index))
            self.params.cdf_fetcher.select_frame(get_ind=index)
            self.image_learn()  # Analyze the input to help make normalization curves
            self.image_modify()  # Actually Normalize This Image
            self.image_store_cdf()
            if do_plot:
                self.params.cdf_fetcher.peek_selection()
                self.image_plot()
            self.first=False
        self.image_save_cdf()
        print(" ^ Success!\n")

#             return self.params.modified_image
#         import pdb; pdb.set_trace()
#         self.params.old_fetchers

#         the_fetcher = [x for x in self.params.old_fetchers if type(x) is LocalCdfFetcher][0]

#         for frame in the_fetcher.select_frame(gen=True):
#             print(frame)

#         pass
    
    def image_store_cdf(self, do_plot=False):
        
        wave = self.params.image_data[0]+0
        frame = self.touchup(self.params.modified_image)+0
        
        self.frame_list.append((frame, wave))
        
#         print("           Storing Frame {}...".format(wave), pointing_end='')
        if do_plot:
            plt.imshow(frame, origin='lower')
            plt.title(wave)
            plt.show()
        
        
        
#         print("Not Yet Implemented")

    def image_save_cdf(self):
        self.params.cdf_fetcher.save_cdf(self.params.new_img_path, self.frame_list, self.params.confirm_save)

    
    def image_plot(self, save=False, show=True, do=True):
        # self.plot_norm_curves(save=False, show=True, extra=True)
        self.plot_full_normalization(save=save, show=show, do=do)
    
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
    
    def __init__(self, fits_path=None, in_name="lev1_t_int", orig=False,
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
            # self.plot_norm_curves(save=True)
        self.out_name = "quantile"
        return self.params.quantile_image
    
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
        frame_is_not_loaded = self.params.raw_image is None
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
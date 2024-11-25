"""This is the script to run on a server somewhere to process the images"""
from src.run import Runner, SingleRunner
from src.science.parameters import Parameters
from src.putter.DesktopPutter import DesktopPutter
from src.putter.AwsPutter import AwsPutter
from src.processor.SunPyProcessor import AIA_PREP_Processor, NRGFProcessor, MSGNProcessor
from src.processor.QRNProcessor import QRNSingleShotProcessor_Legacy
# from src.processor.RHEProcessor import RHEProcessor
from src.processor.SunPyProcessor import RHEFProcessor
from src.fetcher.WebFitsFetcher import WebFitsFetcher
from src.processor.ImageProcessorCV import ImageProcessorCV, MultiImageProcessorCv
from src.processor.CompositeRainbowImageProcessor import RainbowRGBImageProcessor


def run_server_lingon(delay=20, debug=True, do_one="rainbow", stop=True):
    p = Parameters()

    p.is_debug(debug)
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.batch_name("background_server_lingon")
    p.run_type("Web Server Daemon")
    p.do_orig = True
    p.speak_save = False
    p.use_drive = "G"
    p.do_parallel = False
    # p.init_pool(9)

    # Run Flags
    p.download_files(True)
    p.get_fits = True
    p.multiplot_all = False
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    p.upsilon = None

    # These settings might not look like they make sense but they make it work
    p.do_standard_RHE()
    p.do_prep = False
    # p.msgn_targets(['lev1p5'])  # , 'rhe(lev1p5)'
    # p.rhe_targets(["lev1p5"]) #, 'msgn(lev1p5)'])  # "lev1p5",
    # p.png_frame_name = ["lev1p5", "rhe(lev1p5)", 'msgn(lev1p5)', 'rhe(msgn)']  # ['rhe(lev1p5)']
    p.png_frame_name = ["RHEF"]

    # This is the right combination of processors for the server
    if True:
        p.fetchers(WebFitsFetcher,)  # Gets Fits from JSOC Most Recent
        # p.processors([AIA_PREP_Processor], rp=True)
        p.processors([RHEFProcessor],  rp=True)  # Applies the Radial Filtering
        # p.processors([MSGNProcessor], rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        # p.processors([RHEProcessor],  rp=True)  # Applies the Radial Filtering
        pass
        p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
    # p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
    p.putters([RainbowRGBImageProcessor], rp=True)  # Makes the PNGs into a Composite PNG
    p.putters([AwsPutter])  # Uploads the PNGs to AWS
    p.putters([DesktopPutter])  # Sets the PNGs to the Desktop Background

    # Imageprocessor -> get_alphas() to adjust Upsilon

    # # Run the Code
    SingleRunner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server_lingon()
# from sys import path
# #path.append(path[0] + "/..")  # Adds higher directory to python modules path.


# # Create a new list that includes all paths except those that contain 'site-packages'
# new_sys_path = [] #[p for p in path if 'python3' not in p]

# # Update sys.path
# path = new_sys_path

# # Print updated sys.path for verification
# #print(sys.path)
# a = [print(x) for x in path]

# import pdb; pdb.set_trace()

"""This is the script to run on a server somewhere to process the images"""
from run import Runner, SingleRunner
from science.parameters import Parameters
from putter.DesktopPutter import DesktopPutter
from putter.AwsPutter import AwsPutter
from processor.SunPyProcessor import AIA_PREP_Processor, NRGFProcessor, MSGNProcessor
from processor.QRNProcessor import QRNSingleShotProcessor_Legacy
from processor.RHEProcessor import RHEProcessor
from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.ImageProcessorCV import ImageProcessorCV, MultiImageProcessorCv


def run_server_lingon(delay=20, debug=False, do_one='rainbow', stop=True):
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
    p.init_pool(9)

    # Run Flags
    p.download_files(True)
    p.get_fits = True
    p.multiplot_all = False
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'

    # These settings might not look like they make sense but they make it work
    p.do_standard_RHE()
    # p.msgn_targets(['lev1p5'])  # , 'rhe(lev1p5)'
    # p.rhe_targets(["lev1p5", 'msgn(lev1p5)'])  # "lev1p5",
    # p.png_frame_name = ['rhe(msgn)']  # ['rhe(lev1p5)']

    # This is the right combination of processors for the server
    if True:
        p.fetchers(WebFitsFetcher,)  # Gets Fits from JSOC Most Recent
        p.processors([MSGNProcessor], rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        p.processors([RHEProcessor],  rp=True)  # Applies the Radial Filtering
        p.processors([RHEProcessor],  rp=True)  # Applies the Radial Filtering
        p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
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

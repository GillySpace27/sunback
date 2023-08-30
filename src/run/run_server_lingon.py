"""This is the script to run on a server somewhere to process the images"""
from sys import path
path.append(path[0] + "/..")  # Adds higher directory to python modules path.
# a = [print(x) for x in path]

from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.ImageProcessorCV import ImageProcessorCV, MultiImageProcessorCv
from processor.RHEProcessor import RHEProcessor
from processor.QRNProcessor import QRNSingleShotProcessor_Legacy
from processor.SunPyProcessor import AIA_PREP_Processor, NRGFProcessor, MSGNProcessor
from putter.AwsPutter import AwsPutter
from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
from run import Runner, SingleRunner


def run_server(delay=80, debug=False, do_one='rainbow', stop=True):
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
    p.init_pool(4)

    # Run Flags
    p.download_files(True)
    p.get_fits = True
    p.multiplot_all = False
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'

    # These settings might not look like they make sense but they make it work
    p.msgn_targets(['lev1p5']) #, 'rhe(lev1p5)'
    p.rhe_targets(["lev1p5", 'msgn(lev1p5)']) #"lev1p5",
    p.png_frame_name = ['rhe(msgn)'] #['rhe(lev1p5)']

    # This is the right combination of processors for the server
    if True:
        p.fetchers(WebFitsFetcher,           )  # Gets Fits from JSOC Most Recent
        p.processors([MSGNProcessor], rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        p.processors([RHEProcessor],  rp=True)  # Applies the Radial Filtering
        p.processors([RHEProcessor],  rp=True)  # Applies the Radial Filtering
        p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
    p.putters([AwsPutter])  # Uploads the PNGs to AWS

    #Imageprocessor -> get_alphas() to adjust Upsilon

    # # Run the Code
    SingleRunner(p).start()

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server()

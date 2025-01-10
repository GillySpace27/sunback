"""This is the script to run on a server somewhere to process the images"""
from sunback.run import SingleRunner
from sunback.science.parameters import Parameters
from sunback.putter.DesktopPutter import DesktopPutter
from sunback.putter.AwsPutter import AwsPutter
from sunback.processor.SunPyProcessor import RHEFProcessor
from sunback.fetcher.WebFitsFetcher import WebFitsFetcher
from sunback.processor.ImageProcessorCV import ImageProcessorCV
from sunback.processor.CompositeRainbowImageProcessor import RainbowRGBImageProcessor


def run_server_lingon(delay=60, debug=False, do_one="rainbow", stop=True):
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
    # Run Flags
    p.download_files(True)
    p.get_fits = True
    p.multiplot_all = False
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    p.upsilon = None
    p.do_prep = False

    p.do_standard_RHE()

    # This is the right combination of processors for the server
    if True:
        p.fetchers(WebFitsFetcher,)  # Gets Fits from JSOC Most Recent
        p.processors([RHEFProcessor],  rp=True)  # Applies the Sunpy Radial Filtering
        # p.processors([MSGNProcessor], rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
    p.putters([RainbowRGBImageProcessor], rp=True)  # Makes the PNGs into a Composite PNG
    p.putters([AwsPutter])  # Uploads the PNGs to AWS
    p.putters([DesktopPutter])  # Sets the PNGs to the Desktop Background

    # Imageprocessor -> get_alphas() to adjust Upsilon

    # # Run the Code
    SingleRunner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server_lingon()
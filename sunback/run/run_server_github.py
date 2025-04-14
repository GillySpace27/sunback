"""This is the script to run on a server somewhere to process the images"""

from sunback.run import SingleRunner
from sunback.science.parameters import Parameters
from sunback.putter.DesktopPutter import DesktopPutter
from sunback.putter.AwsPutter import AwsPutter
from sunback.processor.SunPyProcessor import RHEFProcessor, UpsilonProcessor, NRGFProcessor
from sunback.fetcher.WebFitsFetcher import WebFitsFetcher
from sunback.processor.ImageProcessorCV import ImageProcessorCV, ImageProcessorHDR
from sunback.processor.CompositeRainbowImageProcessor import RainbowRGBImageProcessor
from sunback.processor.ScienceProcessor import DEMReconstructionProcessor


import logging

# Set the logging level for boto3 and botocore to INFO
logging.getLogger("root").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("s3transfer").setLevel(logging.INFO)
logging.getLogger("sunback").setLevel(logging.DEBUG)

# Optional: Reduce the verbosity of other logs (e.g., urllib3)
logging.getLogger("urllib3").setLevel(logging.INFO)

# Ensure root logger is also set appropriately
logging.basicConfig(level=logging.INFO)


def run_server_github(delay=180, debug=True, do_one="rainbow", stop=True):
    p = Parameters()

    p.is_debug(debug)
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.batch_name("background_server_github")
    p.run_type("Web Server Daemon")
    p.do_orig = True
    p.speak_save = False
    p.use_drive = "C"
    p.do_parallel = False
    # Run Flags
    p.download_files(True)
    p.get_fits = True
    p.multiplot_all = False
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    p.do_vignette = True
    p.do_upsilon = True
    p.do_upsilon_together = False
    p.do_prep = False
    p.visualization_style = "threshold"

    # p.do_standard_RHE()
    # p.msgn_targets(["lev1p5"])
    p.rhe_targets(["lev1p5"])
    p.png_frame_name = ["ups(rhef)"]  # ["rhef(lev1p5)"]
    p.rgb_frame = "rhef(lev1p5)"
    if True:
        p.fetchers(WebFitsFetcher,)  # Gets Fits from JSOC Most Recent
        # # p.processors([AIA_PREP_Processor],)
        p.processors([RHEFProcessor], rp=True)  # Applies the Sunpy Radial Filtering
    #     # p.processors([NRGFProcessor], rp=True)  # Applies the Sunpy Radial Filtering
    #     # # p.processors([MSGNProcessor], rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        p.processors([UpsilonProcessor], rp=True)
    p.processors([DEMReconstructionProcessor])

    p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
    # p.putters([ImageProcessorHDR], rp=True)  # Turns Fits into Pngs

    p.putters([RainbowRGBImageProcessor], rp=True)

    p.putters([AwsPutter])  # Uploads the PNGs to AWS
    # p.putters([DesktopPutter])  # Sets the PNGs to the Desktop Background

    # Imageprocessor -> get_alphas() to adjust Upsilon

    # # Run the Code
    SingleRunner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server_github()

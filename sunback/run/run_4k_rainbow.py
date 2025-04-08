"""This is the script to run on a server somewhere to process the images"""

from sunback.run import SingleRunner
from sunback.science.parameters import Parameters
from sunback.putter.DesktopPutter import DesktopPutter
from sunback.putter.AwsPutter import AwsPutter
from sunback.processor.SunPyProcessor import RHEFProcessor, MSGNProcessor, UpsilonProcessor, AIA_PREP_Processor
from sunback.fetcher.FidoFetcher import FidoFetcher
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

# Optional: Reduce the verbosity of other logs (e.g., urllib3)
logging.getLogger("urllib3").setLevel(logging.INFO)

# Ensure root logger is also set appropriately
logging.basicConfig(level=logging.INFO)


def run_server_4k(delay=60, debug=True, do_one="0171", stop=True):
    p = Parameters()

    p.is_debug(debug)
    p.delay_seconds(delay)
    p.do_one(False, True)
    p.batch_name("4k_rainbow_recent_hdr")
    p.run_type("Single Run Rainbow")
    p.do_recent(True)
    p.do_single = True
    p.do_orig = False
    p.speak_save = False
    p.use_drive = "G"
    p.do_parallel = True
    # Run Flags
    p.download_files(True)
    p.get_fits = True
    p.multiplot_all = False
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    p.do_vignette = True
    p.do_upsilon = True
    p.do_upsilon_together = True
    p.do_prep = True
    p.range(days=6.4)
    # p.do_standard_RHE()
    # p.msgn_targets(["lev1p5"])
    p.rhe_targets(["lev1p5_p"])
    # p.png_frame_name = ["ups(rhef)"]  # ['rhe(lev1p5)']
    p.png_frame_name = ['rhef(lev1p5_p)']
    p.rgb_frame = "rhef(lev1p5_p)"
    # This is the right combination of processors for the 4k server
    if True:
        p.fetchers(FidoFetcher, rp=True)  # Gets Fits from JSOC Most Recent
        p.processors(AIA_PREP_Processor, rp=True)
        # p.processors([MSGNProcessor], rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        p.processors([RHEFProcessor], rp=True)  # Applies the Sunpy Radial Filtering
        # p.processors([UpsilonProcessor], rp=True)  # Applies the Sunpy Radial Filtering
        # p.processors([DEMReconstructionProcessor])
        p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
    # p.putters([ImageProcessorHDR], rp=True)  # Turns Fits into Pngs
    # p.putters(
    #     [RainbowRGBImageProcessor], rp=True
    # )  # Makes the PNGs into a Composite PNG
    p.putters([AwsPutter])  # Uploads the PNGs to AWS
    p.putters([DesktopPutter])  # Sets the PNGs to the Desktop Background

    # Imageprocessor -> get_alphas() to adjust Upsilon

    # # Run the Code
    SingleRunner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    # from tqdm import tqdm
    # for wave in tqdm(known_wavelengths, desc=f"4k Rainbow", unit="img"):
    run_server_4k()

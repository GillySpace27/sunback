from sunback.run import SingleRunner
import os
import subprocess

cwd = os.getcwd()
os.system(f"export PYTHONPATH={cwd}")
os.system("echo $PYTHONPATH")
# result = subprocess.run([f'export PYTHONPATH={cwd}'], capture_output=True, text=True)
# result = subprocess.run(['echo $PYTHONPATH'], capture_output=True, text=True)
# exec(f"export )
# exec(f"echo $PYTHONPATH")

from sunback.science.parameters import Parameters
from sunback.processor.SunPyProcessor import (
    RHEFProcessor,
    SunPyProcessor,
    AIA_PREP_Processor,
    NRGFProcessor,
    FNRGFProcessor,
    IntEnhanceProcessor,
    MSGNProcessor,
)
from sunback.processor.ScienceProcessor import ScienceProcessor
from sunback.processor.QRNProcessor import QRNProcessor, QRNSingleShotProcessor
from sunback.processor.RHTProcessor import RHTProcessor
# from src.processor.RHEProcessor import RHEProcessor

# from src.processor.NoiseGateProcessor import NoiseGateProcessor
from sunback.processor.ImageProcessorCV import (
    ImageProcessorCV,
    MultiImageProcessorCv,
    MultiHistogramProcessorCv,
)
from sunback.fetcher.LocalFetcher import LocalSingleFetcher
from sunback.fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from sunback.fetcher.FidoFetcher import FidoFetcher
import matplotlib.pyplot as plt

# exec("export PYTHONPATH=/src:$PYTHONPATH")


# from src.processor.QRNSubProcessors import QRNSingleShotProcessor


plt.ioff()

# My favorite prominance: "2013-09-29T13:35:00"
# Awesome Plumes: "2019-01-01T00:00:00"
# MC Paper: "2014-11-10T16:00:00"


def run_single(
    wave="0193",
    tstart="2011-08-09T00:00:01",
    duration_seconds=60 * 20,
    frames=None,
    name="newtest193",
):
    """Download a single frame and time-integrate it, then apply RHE
    :type wave: strings
    :type tstart: string
    :type duration_seconds: int or float
    :type frames: int
    """
    # Set the Parameters

    p = default_run_single_params(wave, tstart, duration_seconds, frames, name)

    # p.download_files(False)
    # p.multiplot_all = True
    master = True

    # Set the Processes
    get_images = True and master
    if get_images:
        pass
        # p.fetchers(LocalSingleFetcher)
        p.fetchers(FidoFetcher, rp=True)  # Gets the desired file
        # p.processors([FidoTimeIntProcessor],   rp=True)   # Integrate several frames for S/N
        # p.processors([NoiseGateProcessor],     rp=True)
        p.processors([AIA_PREP_Processor], rp=True)  # Do Sunpy Things

    # 'rhe(lev1p5)' #"msgn(rhe)" #'all' #['rhe', "msgn(rhe)", "rhe(msgn)"] ## I want to be able to call final, but it is made in the processor that save images, so I have to split it out into the touchup processor.
    p.png_frame_name = ["rhef"]
    # p.msgn_targets(["lev1p5", 'rhef'])
    p.msgn_targets(["lev1p5", "rhef"])
    p.rhe_targets(["lev1p5", "nrgf", "msgn"])  # "lev1p5",
    radial_norms = True and master
    if radial_norms:
        pass
        p.processors(
            [NRGFProcessor, RHEFProcessor], rp=True
        )  # Applies the Sunpy NRGF Filter
        # p.processors([RHEFProcessor], rp=True)  # Applies the RHE Filter
        p.processors(
            [MSGNProcessor, RHEFProcessor], rp=True
        )  # Applies the Sunpy Multiscale Gausian Norm
        p.processors([RHEFProcessor], rp=True)  # Applies the RHE Filter
        # p.processors([ImageProcessorCV])

    p.aftereffects_in_name = [
        "rhe(lev1p5)",
    ]
    aftereffects = False and master
    if aftereffects:
        pass
        p.processors([RHTProcessor], rp="redo")  # Applies the Rolling Hough Transform+
        # p.processors([RHTProcessor],            rp="redo")  # Applies the Rolling Hough Transform+
    use_putters = True or master
    if use_putters:
        p.putters(MultiImageProcessorCv, rp=True)  # Makes the PNGs from Fits
        # p.putters(ScienceProcessor,            rp=True)  # Makes the PNGs from Fits
        p.putters(MultiHistogramProcessorCv, rp=True)  # Makes the PNGs from Fits

    # Run the Code
    runner = SingleRunner(p)
    runner.start()


def default_run_single_params(
    wave, tstart, duration_seconds=60, frames=None, name="Single"
):
    """Create the default parameters and parse and set the inputs"""
    p = Parameters()

    # Parse Inputs
    p.do_one(wave, stop=True)
    p.set_time_range_duration(tstart)
    if frames is not None:
        duration_seconds = frames * 12
    p.exposure_time_seconds(duration_seconds)

    # Set Metadata
    p.batch_name(name)
    p.run_type("Process a Single Image Start to Finish")
    p.fetchers(LocalSingleFetcher)
    # Set Flags
    p.do_single = True
    p.doing_jpeg = False
    p.config = None
    p.destroy = False
    p.is_debug(True)
    p.do_cat = True
    p.do_recent(False)
    p.currently_local = True
    p.download_files(True)
    p.do_prep = False  # Won't do AIA prep upon download of each frame
    p.use_drive = ""
    # p.do_one(wave, stop=True)

    # p.processors([FNRGFProcessor],            rp=True)  # Applies the Sunpy FNRGF Filter
    # p.processors([QRNSingleShotProcessor],           rp=True)  # Applies the QRN Filter
    # p.png_frame_name = ['lev1P5_Q', 'RHE']
    # p.putters(ImageProcessorCV,            rp=True)  # Makes the PNGs from Fits

    return p


if __name__ == "__main__":
    # Do something if this file is invoked on its own

    all_wavelengths = ["0171"]  # , '0304', ]  #, '0211', '0193' ]
    # all_wavelengths = ['211', '0094', '0335'] ['0094', '0131'] #,
    # all_wavelengths = ['0171', '0304'] #,  "0304"]

    for wave_to_use in all_wavelengths:
        run_single(wave=wave_to_use)

        # import sys
        # sys.exit()


# p.putters([VideoProcessor],             rp=True)  # Makes the PNGs into a Movie

# p.processors([FidoTimeIntProcessor], rp=True)   # Integrate several frames for S/N

# p.processors([QRNradialFiltProcessor],  rp=True)  # Applies the QRN Filter


# def run_range_multishot_movie(debug=True, do_one='0304', stop=True,
#                               tstart='2016/11/04 01:00:00', tend='2014/11/06 00:00:00',
#                               cadence_minutes=5, fps=10, exposure_time=24,
#                               key_fixed_cadence=3, key_fixed_number=None, time_preset="p"):
#     # Set the Parameters
#     p = Parameters()
#     # tstart, tend = self.params.set_time_range_duration(tstart, duration_seconds=60):
#     time_string = tstart.replace('/', '_').replace(' ', '_').replace(':', '')
#     rng = "MultiRange\\MRange_{}".format(time_string)
#     p.batch_name(rng)
#     p.run_type("Make Movie of Given Time Range, With Time Integration")
#     p.do_one(do_one, stop)
#     p.is_debug(debug)
#
#     # Set the Times
#     if not p.load_preset_time_settings(time_preset):
#         p.cadence_minutes(cadence_minutes)
#         p.exposure_time_seconds(exposure_time)
#         p.frames_per_second(fps)
#     p.fixed_cadence_keyframes(key_fixed_cadence)
#     p.fixed_number_keyframes(key_fixed_number)
#     p.time_period(period=[tstart, tend])
#
#     # p.compare_fits_frames()
#
#     # Set the Processes
#     # p.fetchers(FidoFetcher)                                     # Gets Fits FIDO
#     # p.processors([FidoTimeIntProcessor])                        # Integrate several frames for S/N
#
#     p.processors([QRNpreProcessor], rp=True)  # Learns the bounds of the dataset for QRN
#     p.processors([QRNradialFiltProcessor], rp=True)  # Applies the QRN Filter
#
#     p.putters([ImageProcessor], rp=True)  # Makes the PNGs from Fits
#     p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
#
#     # Run the Code
#     run.Runner(p).pointing_start()

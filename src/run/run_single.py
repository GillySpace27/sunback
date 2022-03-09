from fetcher.FidoFetcher import FidoFetcher
from fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from fetcher.LocalFetcher import LocalSingleFetcher
from processor.ImageProcessorCV import ImageProcessorCV, MultiImageProcessorCv
from processor.NoiseGateProcessor import NoiseGateProcessor
from processor.QRNProcessor import QRNProcessor
from processor.RHTProcessor import RHTProcessor
from processor.SRNProcessor import SRNProcessor
from processor.SRNSubProcessors import SRNSingleShotProcessor
from processor.SunPyProcessor import SunPyProcessor, AIA_PREP_Processor, NRGFProcessor, FNRGFProcessor, IntEnhanceProcessor, MSGNProcessor
from science.parameters import Parameters
from run import SingleRunner
import matplotlib.pyplot as plt

plt.ioff()


def run_single(wave="0304", tstart="2013-09-29T13:35:00", duration_seconds=60*4, frames=None):
    """Download a single image and time-integrate it, then apply QRN
        :type wave: strings
        :type tstart: string
        :type duration_seconds: int or float
        :type frames: int
    """
    # Set the Parameters
    name = "Single_Test"
    p = default_run_single_params(wave, tstart, duration_seconds, frames, name)
    
    master = True
    
    # Set the Processes
    get_images = True and master
    if get_images:
        p.fetchers(FidoFetcher,                rp=True)   # Gets the desired file
        p.processors([FidoTimeIntProcessor],   rp=True)   # Integrate several frames for S/N
        # p.processors([NoiseGateProcessor],     rp=True)
        p.processors([AIA_PREP_Processor],     rp=True)   # Do Sunpy Things
    
    radial_norms = False and master
    if radial_norms:
        p.processors([QRNProcessor],            rp=True)  # Applies the QRN Filter
        p.processors([NRGFProcessor],           rp=True)  # Applies the Sunpy NRGF Filter
        p.processors([IntEnhanceProcessor],     rp=True)  # Applies the Sunpy IntEnhance Filter
    
    p.aftereffects_in_name = "quantile"
    aftereffects = False and master
    if aftereffects:
        p.processors([MSGNProcessor],           rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        p.processors([RHTProcessor],            rp=True)  # Applies the Rolling Hough Transform
    
    p.putters(MultiImageProcessorCv,            rp=True)  # Makes the PNGs from Fits
    
    # Run the Code
    runner = SingleRunner(p)
    runner.start()


def default_run_single_params(wave, tstart, duration_seconds=60, frames=None, name="Single"):
    """ Create the default parameters and parse and set the inputs"""
    p = Parameters()
    
    # Parse Inputs
    p.do_one(wave, True)
    p.set_time_range_duration(tstart)
    if frames is not None:
        duration_seconds = frames*12
    p.exposure_time_seconds(duration_seconds)
    
    # Set Metadata
    p.batch_name(name)
    p.run_type("Process a Single Image Start to Finish")
    p.fetchers(LocalSingleFetcher)
    # Set Flags
    p.do_single = True
    p.config = None
    p.destroy = False
    p.is_debug(True)
    p.do_cat = True
    p.do_recent(False)
    p.currently_local = True
    p.download_files(True)
    p.do_prep = False # Won't do AIA prep upon download of each frame
    p.use_drive = "G"

    # p.processors([FNRGFProcessor],            rp=True)  # Applies the Sunpy FNRGF Filter
    # p.processors([SRNSingleShotProcessor],           rp=True)  # Applies the SRN Filter
    # p.png_frame_name = ['lev1P5_Q', 'Quantile']
    # p.putters(ImageProcessorCV,            rp=True)  # Makes the PNGs from Fits

    
    return p

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    
    all_wavelengths = ['0304', '0171', '0211', '0193', '0131', '0335', '0094', ]
    # all_wavelengths = ['0171', '0304'] #,  "0304"]
    
    for wave_to_use in all_wavelengths:
        run_single(wave=wave_to_use)
        
        # import sys
        # sys.exit()
























    # p.putters([VideoProcessor],             rp=True)  # Makes the PNGs into a Movie


    # p.processors([FidoTimeIntProcessor], rp=True)   # Integrate several frames for S/N


    # p.processors([SRNradialFiltProcessor],  rp=True)  # Applies the SRN Filter



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
#     p.processors([SRNpreProcessor], rp=True)  # Learns the bounds of the dataset for SRN
#     p.processors([SRNradialFiltProcessor], rp=True)  # Applies the SRN Filter
#
#     p.putters([ImageProcessor], rp=True)  # Makes the PNGs from Fits
#     p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
#
#     # Run the Code
#     run.Runner(p).pointing_start()

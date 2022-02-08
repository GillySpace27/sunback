from fetcher.FidoFetcher import FidoFetcher
from fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from fetcher.LocalFetcher import LocalSingleFetcher
from processor.ImageProcessorCV import ImageProcessorCV
from processor.QRNProcessor import QRNProcessor
from science.parameters import Parameters
from run import SingleRunner
import matplotlib.pyplot as plt

plt.ioff()


def run_single(tstart="2022-01-01T00:00:00", duration_seconds=60):
    
    # Set the Parameters
    p = default_run_single_params()
    p.set_time_range_duration(tstart)
    p.exposure_time_seconds(duration_seconds)
    
    
    # Set the Processes
    p.fetchers(FidoFetcher,                rp=True)  # Gets the desired file
    p.processors([FidoTimeIntProcessor],  rp=True)   # Integrate several frames for S/N
    p.processors([QRNProcessor],           rp=True)  # Applies the SRN Filter
    p.putters(ImageProcessorCV,           rp=True)  # Makes the PNGs from Fits
    
    # Run the Code
    aa = SingleRunner(p)
    aa.start()

def default_run_single_params():
    p = Parameters()
    p.do_single = True
    p.config = None
    p.destroy = False
    p.batch_name("Single")
    p.png_frame_name = 'QRN'
    p.run_type("Process a Single Image Start to Finish")
    p.do_one("0304", True)
    p.is_debug(True)
    p.do_cat = True
    p.do_recent(False)
    p.currently_local = True
    p.use_drive = "G"
    
    
    
    # p.fetchers(LocalSingleFetcher,                rp=True)  # Gets the desired file
    
    # # Set the Times
    # # if not p.load_preset_time_settings(config["time_preset"]):
    # p.cadence_minutes(config["cadence_minutes"])
    # p.frames_per_second(config["fps"])
    # p.fixed_cadence_keyframes(config["key_fixed_cadence"])
    # p.fixed_number_keyframes(config["key_fixed_number"])
    
    
    
    return p

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    # test_image = r"D:\sunback_images\Single\aia.lev1_euv_12s.2013-09-29T120009Z.304.image_lev1.fits"
    # test_image = r"D:\sunback_images\Single\aia.lev1_euv_12s.2013-10-02T162012Z.171.image_lev1.fits"
    run_single()


























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
#     run.Runner(p).start()

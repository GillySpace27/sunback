from sunback.fetcher.LocalFetcher import LocalSingleFetcher
from sunback.processor.ImageProcessorCV import ImageProcessorCV
from sunback.science.parameters import Parameters
from sunback.run import SingleRunner
import matplotlib.pyplot as plt
from sunback.processor.RHEProcessor import RHEProcessor
from sunback.processor.SunPyProcessor import MSGNProcessor

plt.ioff()


def run_single_in_memory(image, center=(900, 1200)):
    # Set the Parameters
    p = make_params()
    p.use_image_path(image)
    p.center = center

    # Set the Processes
    p.fetchers(LocalSingleFetcher,                rp=True)  # Gets the desired file
    p.processors([RHEProcessor],        rp=True)
    p.processors([MSGNProcessor],        rp=True)  # Makes the PNGs from Fits
    p.putters(ImageProcessorCV,           rp=True)  # Makes the PNGs from Fits


    # Run the Code
    aa = SingleRunner(p)
    aa.start()

def make_params():
    p = Parameters()
    p.config = None
    p.destroy = False
    p.batch_name("Single")
    p.png_frame_name = 'RHE'
    p.run_type("Process a Single Image")
    p.do_single = True
    p.do_one(True, True)
    p.is_debug(True)
    p.do_cat = True
    # p.frames_per_second(12)
    # p.do_recent(False)
    return p

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    # test_image = r"D:\sunback_images\Single\aia.lev1_euv_12s.2013-09-29T120009Z.304.image_lev1.fits"
    # test_image = r"D:\sunback_images\Single\aia.lev1_euv_12s.2013-10-02T162012Z.171.image_lev1.fits"
    # test_image = "sunback_data/PXL_20240408_184406528.jpg"
    # test_image, center = "sunback_data/eclipse/Photos-001 (3)/PXL_20240408_184359057.jpg",  (1210, 706)
    # test_image, center = "sunback_data/eclipse/Photos-001 (3)/PXL_20240408_184406528.jpg", (2080, 1820),
    test_image, center = "sunback_data/cosmic_background_studios_1.jpg", (720, 899, ),



    # test_image = "sunback_data/eclipse.fits"
    # run_single_in_memory(test_image, (1234, 1086))
    run_single_in_memory(test_image, center)
    # run_single_in_memory(test_image, None)


























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

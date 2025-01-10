from sunback.fetcher.FidoFetcher import FidoFetcher
from sunback.fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from sunback.processor.ImageProcessorCV import ImageProcessorCV
from sunback.processor.SunPyProcessor import RHEFProcessor
from sunback.processor.VideoProcessor import VideoProcessor
from sunback.science.parameters import Parameters
import run
import matplotlib.pyplot as plt
plt.ioff()


def run_recent_movie(delay=10, debug=True, do_one="304", stop=True, cadence_minutes=5, fps=24, range_days=7, exposure=12*10):
    # Set the Parameters
    p = Parameters()
    # p.delay_seconds(delay)
    p.batch_name("Recent_Movie_304_11_24")
    p.run_type("Generate Recent Movie")
    p.do_one(do_one, stop)
    p.verb = False
    p.do_orig = False
    p.do_cat = False
    # p.stop_after_one(stop)
    p.is_debug(debug)

    p.download_files(True)
    # p.overwrite_pngs(True)
    # p.delete_old(True)

    # Set the Times
    debug_hours = 36 # Range in Hours
    debug_cadence = 60 # Cadence in Minutes
    # p.set_time_range_duration()

    p.range(days=range_days, hours=None)
    p.cadence_minutes(cadence_minutes)
    p.frames_per_second(fps)
    p.exposure_time_seconds(exposure)
    p.do_parallel = True
    p.init_pool(20)
    p.png_frame_name =["-1"]

    # Set the Processes
    # p.fetchers(FidoFetcher, rp=True)                                     # Gets Fits FIDO
    # p.processors([FidoTimeIntProcessor], rp=None)                        # Integrate several frames for S/N

    # p.processors([RHEFProcessor],            rp=True)  # Applies the Radial Filtering

    #
    # p.putters([ImageProcessorCV], rp=False)  # Makes the PNGs from Fits
    p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
    p.do_recent(True)

    # Run the Code
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_recent_movie()


    #, VideoProcessor(p)])  #

    # p.processors([RadialFiltProcessor(p), NoiseGateProcessor(p), VideoProcessor(p)])  #

    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    # p.putter(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    # p.putters(NullPutter(p))       # Does Nothing with the PNGS


    # p.sonify_limit(False)
    # p.remove_old_images(False)
    # p.make_compressed(True)
    # p.sonify_images(True, True)
    # p.sonify_images(False, False)
    # p.do_171(True)
    # p.do_304(True)

    # # p.bpm(150)
    #

    # # Set the Processes
    # # if p.download_files():
    # p.fetchers(FidoFetcher())      # Gets Fits FIDO
    #
    #
    # p.putters([ImageProcessor])
    # p.putters([VideoProcessor])













































































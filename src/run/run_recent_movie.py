from src.fetcher.FidoFetcher import FidoFetcher
from src.fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from src.processor.ImageProcessorCV import ImageProcessorCV
# from src.processor.ImageProcessor import ImageProcessor
# from src.processor.QRNProcessor import QRNProcessor, QRNSingleShotProcessor
# from src.processor.QRNProcessor import QRNSingleShotProcessor #, QRNpreProcessor, QRNradialFiltProcessor
from src.processor.RHEProcessor import RHEProcessor
from src.processor.VideoProcessor import VideoProcessor
from src.science.parameters import Parameters
import run
import matplotlib.pyplot as plt
plt.ioff()


def run_recent_movie(delay=10, debug=True, do_one="171", stop=True, cadence_minutes=5, fps=30, range_days=7, exposure=60*100):
    # Set the Parameters
    p = Parameters()
    # p.delay_seconds(delay)
    p.batch_name("Recent_Movie_171")
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

    # Set the Processes
    p.fetchers(FidoFetcher, rp=True)                                     # Gets Fits FIDO
    p.processors([FidoTimeIntProcessor], rp=None)                        # Integrate several frames for S/N

    p.processors([RHEProcessor],            rp=True)  # Applies the Radial Filtering
    # p.processors([QRNpreProcessor],     rp=True)  # Learns the bounds of the dataset for QRN
    # p.processors([QRNradialFiltProcessor], rp=True)  # Applies the QRN Filter
    #
    p.putters([ImageProcessorCV], rp=True)  # Makes the PNGs from Fits
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













































































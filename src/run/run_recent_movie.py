from fetcher.FidoFetcher import FidoFetcher
from processor.ImageProcessor import ImageProcessor
from processor.RadialFiltProcessor import RadialFiltProcessor
from processor.VideoProcessor import VideoProcessor
from science.parameters import Parameters
import run
import matplotlib.pyplot as plt
plt.ioff()


def run_recent_movie(delay=10, debug=True, do_one="0304", stop=True, cadence_minutes=10, fps=23, range_days=2, range_hours=12):
    # Set the Parameters
    p = Parameters()
    # p.delay_seconds(delay)
    p.batch_name("Recent_Movie")
    p.run_type("Generate Recent Movie")
    p.do_one(do_one, stop)
    p.verb = False
    p.do_orig = False
    p.do_cat = False
    # p.stop_after_one(stop)
    p.is_debug(debug)
    p.do_recent(True)
    
    p.download_images(False)
    # p.overwrite_pngs(True)
    # p.delete_old(True)
    
    # Set the Times
    debug_hours = 36 # Range in Hours
    debug_cadence = 60 # Cadence in Minutes
    p.range(days=None if debug else range_days, hours=debug_hours if debug else range_hours)
    p.cadence_minutes(debug_cadence if debug else cadence_minutes)
    p.frames_per_second(fps)

    # Set the Processes
    # if p.download_images():
    p.fetchers(FidoFetcher())      # Gets Fits FIDO

    p.processors([RadialFiltProcessor()])

    p.putters([ImageProcessor()])
    p.putters([VideoProcessor()])

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
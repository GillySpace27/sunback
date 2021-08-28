# from fetcher.LocalFitsFetcher import LocalFitsFetcher
# from processor.RadialFiltProcessor import RadialFiltProcessor
# from fetcher.FidoFetcher import FidoFetcher
# from processor.VideoProcessor import VideoProcessor
from fetcher.FidoFetcher import FidoFetcher
from fetcher.LocalFetcher import LocalFetcher
from processor.RadialFiltProcessor import RadialFiltProcessor
from processor.VideoProcessor import VideoProcessor
from science.parameters import Parameters
# from putter.NullPutter import NullPutter
import run
import matplotlib.pyplot as plt
plt.ioff()

def run_recent_movie(delay=10, debug=True, do_one="0171", stop=True):
    p = Parameters()
    p.delay_seconds(delay)
    p.batch_name("Recent")
    p.run_type("Run Recent Movie")
    p.do_one(do_one, True)
    p.stop_after_one(stop)
    p.is_debug(debug)
    p.do_recent(True)
    
    p.range(days=1)
    p.cadence_minutes(30)
    p.frames_per_second(18)

    p.download_images(True)
    # p.overwrite_pngs(True)
    # p.delete_old(True)
    #
    # # p.bpm(150)
    #
    
    
    p.fetchers(FidoFetcher())      # Gets Fits FIDO

    p.processors([RadialFiltProcessor()])
    p.processors([VideoProcessor()])
    
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
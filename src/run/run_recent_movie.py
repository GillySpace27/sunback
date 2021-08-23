from fetcher.LocalFetcher import LocalFetcher
from processor.RadialFiltProcessor import RadialFiltProcessor
from fetcher.FidoFetcher import FidoFetcher
from processor.VideoProcessor import VideoProcessor
from science.parameters import Parameters
from putter.NullPutter import NullPutter
import run
import matplotlib.pyplot as plt
plt.ioff()

def run_recent_movie(delay=10, debug=False, do_one=False, stop=False):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.is_debug(debug)
    p.do_recent(True)
    p.resolution(1024)
    # p.do_upload_to_S3(False)
    
    # p.time_period(period=['2013/12/21 04:00', '2013/12/24 08:00'])
    # p.resolution(1024)
    p.range(days=1)
    p.download_images(True)
    p.overwrite_pngs(True)
    p.delete_old(True)
    p.cadence_minutes(10)
    p.frames_per_second(18)
    # p.bpm(150)
    
    # p.sonify_limit(False)
    # p.remove_old_images(False)
    # p.make_compressed(True)
    # p.sonify_images(True, True)
    # p.sonify_images(False, False)
    # p.do_171(True)
    # p.do_304(True)
    
    # p.fetcher(WebFetcher(p))      # Gets Fits from JSOC Most Recent
    p.fetchers(FidoFetcher(p))      # Gets Fits FIDO
    # p.fetcher(LocalFetcher(p))      # Gets Fits from Disk
    # p.fetcher(AwsFetcher(p))        # Gets PNGs from S3 Daemon
    
    p.processors([RadialFiltProcessor(p)]) #, VideoProcessor(p)])  #
    
    # p.processors([RadialFiltProcessor(p), NoiseGateProcessor(p), VideoProcessor(p)])  #

    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    # p.putter(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    p.putters(NullPutter(p))       # Does Nothing with the PNGS
    
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_recent_movie(do_one='0304', stop=True, debug=True)

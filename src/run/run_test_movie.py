import time

from executor.LocalExecutor import LocalExecutor
from executor.ModifyExecutor import ModifyExecutor
from fetcher.AwsFetcher import AwsFetcher
from fetcher.FidoFetcher import FidoFetcher
from fetcher.WebFetcher import WebFetcher
from fetcher.LocalFetcher import LocalFetcher
from post_processor.VideoPostProcessor import VideoPostProcessor
from putter.LocalPutter import LocalPutter
from science.parameters import Parameters
from putter.AwsPutter import AwsPutter
from putter.NullPutter import NullPutter
import run


def run_movie(delay=10, debug=False, do_one=False, stop=False):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.is_debug(debug)
    p.do_recent(False)
    # p.do_upload_to_S3(False)
    
    p.time_period(period=['2013/10/31 12:00', '2013/11/01 12:00'])
    p.resolution(1024)
    # p.range(days=0.25)
    p.download_images(False)
    p.overwrite_pngs(True)
    p.cadence_minutes(60 * 12)
    p.frames_per_second(10)
    # p.bpm(150)
    
    # p.sonify_limit(False)
    # p.remove_old_images(False)
    # p.make_compressed(True)
    # p.sonify_images(True, True)
    # p.sonify_images(False, False)
    # p.do_171(True)
    # p.do_304(True)
    
    # p.fetcher(WebFetcher(p))      # Gets Fits from JSOC Most Recent
    p.fetcher(FidoFetcher(p))      # Gets Fits FIDO
    # p.fetcher(AwsFetcher(p))        # Gets PNGs from S3 Daemon
    # p.fetcher(LocalFetcher(p))      # Gets Fits from Disk
    
    p.executor(ModifyExecutor(p))  # Makes the PNGs from Fits
    # p.executor(LocalExecutor(p))    # Gets the PNGs from Disk
    
    p.post_processor([VideoPostProcessor(p),])  # Makes the PNGs into a Movie
    
    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    # p.putter(LocalPutter(p))        # Runs the Desktop Background Sequence on PNGs
    p.putter(NullPutter(p))       # Does Nothing with the PNGS
    
    tt = time.time()
    run.Runner(p).start()
    ttt =time.time() - tt
    print("Program took {} seconds, or {} minutes".format(ttt, ttt/60))

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_movie(debug=True)

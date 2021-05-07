from executor.LocalExecutor import LocalExecutor
from executor.ModifyExecutor import ModifyExecutor
from fetcher.AwsFetcher import AwsFetcher
from fetcher.WebFetcher import WebFetcher
from fetcher.LocalFetcher import LocalFetcher
from putter.LocalPutter import LocalPutter
from science.parameters import Parameters
from putter.AwsPutter import AwsPutter
from putter.NullPutter import NullPutter
import run


def run_all_locally(delay=10, debug=False, do_one=False, stop=False):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.is_debug(debug)
    
    p.fetcher(WebFetcher(p))      # Gets Fits from JSOC Most Recent
    # p.fetcher(AwsFetcher(p))        # Gets PNGs from S3 Daemon
    # p.fetcher(LocalFetcher(p))      # Gets Fits from Disk
    
    p.executor(ModifyExecutor(p)) # Makes the PNGs from Fits
    # p.executor(LocalExecutor(p))    # Gets the PNGs from Disk
    
    # p.processor(MovieProcessor(p)) # Makes the PNGs into a Movie
    
    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    p.putter(LocalPutter(p))        # Runs the Desktop Background Sequence on PNGs
    # p.putter(NullPutter(p))       # Does Nothing with the PNGS
    
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_all_locally(debug=True)

from Executor.LocalExecutor import LocalExecutor
from Executor.ModifyExecutor import ModifyExecutor
from Fetcher.WebFetcher import WebFetcher
from Fetcher.LocalFetcher import LocalFetcher
from Putter.LocalPutter import LocalPutter
from science.parameters import Parameters
from Putter.AwsPutter import AwsPutter
from Putter.NullPutter import NullPutter
import run


def run_aws(delay=10, debug=False, do_one=False, stop=True):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.is_debug(debug)
    
    # p.fetcher(WebFetcher(p))      # Gets Fits from Space
    p.fetcher(LocalFetcher(p))      # Gets Fits from Disk
    
    # p.executor(ModifyExecutor(p)) # Makes the PNGs from Fits
    p.executor(LocalExecutor(p))    # Gets the PNGs from Disk
    
    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    p.putter(LocalPutter(p))        # Runs the Desktop Background Sequence on PNGs
    # p.putter(NullPutter(p))       # Does Nothing with the PNGS
    
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_aws(debug=True)

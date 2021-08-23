from processor.RadialFiltProcessor import RadialFiltProcessor
from fetcher.WebFetcher import WebFetcher
from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
import run


def run_all_locally(delay=10, debug=False, do_one=False, stop=False):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.is_debug(debug)
    
    p.fetchers(WebFetcher(p))      # Gets Fits from JSOC Most Recent
    # p.fetcher(AwsFetcher(p))        # Gets PNGs from S3 Daemon
    # p.fetcher(LocalFetcher(p))      # Gets Fits from Disk
    
    p.executor(RadialFiltProcessor(p)) # Makes the PNGs from Fits
    # p.executor(LocalExecutor(p))    # Gets the PNGs from Disk
    
    # p.processor(MovieProcessor(p)) # Makes the PNGs into a Movie
    
    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    p.putters(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    # p.putter(NullPutter(p))       # Does Nothing with the PNGS
    
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_all_locally(debug=True)

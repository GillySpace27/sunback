"""This is the script to run on a server somewhere to process the images"""
from processor.RadialFiltProcessor import RadialFiltProcessor
from fetcher.LocalFetcher import LocalFetcher
from science.parameters import Parameters
from putter.AwsPutter import AwsPutter
from run import Runner

def run_aws(delay=10, debug=False, do_one=False, stop=False):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.is_debug(debug)
    
    # p.fetcher(WebFetcher(p))      # Gets Fits from JSOC Most Recent
    p.fetchers(LocalFetcher(p))      # Gets Fits from Disk
    
    p.executor(RadialFiltProcessor(p)) # Makes the PNGs from Fits
    # p.executor(LocalExecutor(p))    # Gets the PNGs from Disk
    
    # p.processor(MovieProcessor(p)) # Makes the PNGs into a Movie
    
    p.putters(AwsPutter(p))        # Uploads the PNGs to AWS
    # p.putter(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    # p.putter(NullPutter(p))       # Does Nothing with the PNGS
    
    Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_aws(debug=True)

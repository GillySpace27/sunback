"""This is the script to run on a server somewhere to process the images"""
from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.RadialFiltProcessor import RadialFiltProcessor
from putter.AwsPutter import AwsPutter
from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
# from putter.AwsPutter import AwsPutter
from run import Runner

def run_aws(delay=10, debug=False, do_one=False, stop=True):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.is_debug(debug)
    p.batch_name()
    
    p.fetchers(WebFitsFetcher(p))      # Gets Fits from JSOC Most Recent
    
    p.processors(RadialFiltProcessor(p))  # Applies the Radial Filtering
    
    # p.putters(AwsPutter(p))        # Uploads the PNGs to AWS
    p.putters(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    
    Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_aws(debug=True)

"""This is the script to run on a server somewhere to process the images"""

from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.RadialFiltProcessor import RadialFiltProcessor
from putter.AwsPutter import AwsPutter
from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
from run import Runner


def run_server(delay=10, debug=True, do_one=False, stop=False):
    p = Parameters()
    p.is_debug(debug)
    p.delay_seconds(2 if debug else delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    p.batch_name("background_server")
    p.run_type("Web Server Daemon")
    
    p.fetchers(WebFitsFetcher())  # Gets Fits from JSOC Most Recent
    p.processors(RadialFiltProcessor())  # Applies the Radial Filtering
    # p.putters(AwsPutter())  # Uploads the PNGs to AWS
    p.putters(DesktopPutter())        # Runs the Desktop Background Sequence on PNGs
    
    Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server()

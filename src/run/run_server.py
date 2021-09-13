"""This is the script to run on a server somewhere to process the images"""

from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.ImageProcessor import ImageProcessor
from processor.SRNProcessor import SRNProcessor, SRNSingleShotProcessor
from putter.AwsPutter import AwsPutter
from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
from run import Runner


def run_server(delay=10, debug=True, do_one='0171', stop=False):
    p = Parameters()
    p.is_debug(debug)
    p.delay_seconds(3 if debug else delay)
    p.do_one(do_one, True if debug else stop)
    # p.stop_after_one(True)
    p.batch_name("background_server")
    p.run_type("Web Server Daemon")
    
    p.do_orig = True
    
    # Run Flags
    # p.redownload_files(False)
    # p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    # p.overwrite_pngs(True)
    # p.write_video(False)
    p.set_current_wave('rainbow')
    # p.delete_old(True)
    
    p.fetchers(WebFitsFetcher(rp=True))  # Gets Fits from JSOC Most Recent
    
    p.processors(SRNSingleShotProcessor())  # Applies the Radial Filtering
    
    p.putters([ImageProcessor()])  # Turns Fits into Pngs
    
    # if p.is_debug():
    #     p.putters([DesktopPutter()])  # Runs the Desktop Background Sequence on PNGs
    # else:
    p.putters([AwsPutter()])  # Uploads the PNGs to AWS
    
    Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server()

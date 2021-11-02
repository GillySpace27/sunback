"""This is the script to run on a server somewhere to process the images"""

from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.ImageProcessor import ImageProcessor
# from processor.SRNProcessor import SRNProcessor, \
from processor.ImageProcessorCV import ImageProcessorCV
from processor.SRNSubProcessors import SRNSingleShotProcessor, SRNpreProcessor, SRNradialFiltProcessor
# from putter.AwsPutter import AwsPutter
# from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
from run import Runner


def run_server(delay=10, debug=True, do_one='rainbow', stop=True):
    p = Parameters()
    p.is_debug(debug)
    p.delay_seconds(3 if debug else delay)
    p.do_one(do_one, stop)
    # p.stop_after_one(True)
    p.batch_name("background_server")
    p.run_type("Web Server Daemon")
    p.png_frame_name = 'SRN'
    p.do_orig = True
    
    # Run Flags
    # p.download_files(False)
    # p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    # p.overwrite_pngs(True)
    # p.write_video(False)
    # p.set_current_wave('rainbow')
    # # p.delete_old(True)

    # p.fetchers(WebFitsFetcher, rp=True)  # Gets Fits from JSOC Most Recent
    # p.processors(SRNSingleShotProcessor, rp=True)  # Applies the Radial Filtering
    p.putters([ImageProcessorCV], rp=True)  # Turns Fits into Pngs
    
    #
    # p.processors(SRNpreProcessor, rp=True)  # Applies the Radial Filtering
    # p.processors(SRNradialFiltProcessor, rp=True)  # Applies the Radial Filtering
    # if p.is_debug():
    #     p.putters([DesktopPutter()])  # Runs the Desktop Background Sequence on PNGs
    # else:
    # p.putters([AwsPutter()])  # Uploads the PNGs to AWS
    
    # Runner(p).start()
    
    
    # Set the Parameters
    # p = make_params(batch_name, wave, config)
    
    # Set the Processes
    # p.processors([FidoTimeIntProcessor], rp=None)                        # Integrate several frames for S/N
    
    # p.processors([SRNpreProcessor],     rp=True)  # Learns the bounds of the dataset for SRN
    # p.processors([SRNradialFiltProcessor], rp=True)  # Applies the SRN Filter
    # #
    # p.putters([ImageProcessorCV], rp=True)  # Makes the PNGs from Fits
    # p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
    #
    # # Run the Code
    Runner(p).start()

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server()

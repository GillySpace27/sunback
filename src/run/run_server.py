"""This is the script to run on a server somewhere to process the images"""

from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.ImageProcessorCV import ImageProcessorCV, MultiImageProcessorCv
from processor.RHEProcessor import RHEProcessor
from processor.QRNProcessor import QRNSingleShotProcessor_Legacy
from processor.SunPyProcessor import AIA_PREP_Processor, NRGFProcessor, MSGNProcessor
from putter.AwsPutter import AwsPutter
from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
from run import Runner, SingleRunner


def run_server(delay=60, debug=True, do_one='rainbow', stop=True):
    p = Parameters()
    p.is_debug(debug)
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    # p.stop_after_one(True)
    p.batch_name("background_server")
    p.run_type("Web Server Daemon")
    p.do_orig = True
    p.speak_save = False
    p.use_drive = "G"
    p.do_parallel = False
    p.init_pool(4)
    # Run Flags
    p.download_files(True)
    p.get_fits = True
    p.multiplot_all = False
    # p.set_waves_to_do('0171')
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    # p.overwrite_pngs(True)
    # p.write_video(False)
    # p.set_current_wave('rainbow')
    # # p.delete_old(True)
    p.png_frame_name = ['RHE', "lev1p5"]
    p.msgn_targets(['lev1p5', 'rhe(lev1p5)'])
    p.fetchers(WebFitsFetcher,                      )  # Gets Fits from JSOC Most Recent
    p.processors([RHEProcessor],            rp=True)  # Applies the Radial Filtering
    p.processors([MSGNProcessor],           rp=True)  # Applies the Sunpy Multiscale Gausian Norm
    p.processors([MSGNProcessor],           rp=True)  # Applies the Sunpy Multiscale Gausian Norm
    p.putters([ImageProcessorCV],           rp=True)  # Turns Fits into Pngs
    p.putters([MultiImageProcessorCv],      rp=True)  # Makes the PNGs from Fits
    p.putters([AwsPutter])  # Uploads the PNGs to AWS
    p.putters([DesktopPutter], rp=True)  # Runs the Desktop Background Sequence on PNGs
    

    
    # p.processors([AIA_PREP_Processor],      rp=True   )  # Do Sunpy Things
    # p.processors([QRNSingleShotProcessor_Legacy],            rp=True)  # Applies the Radial Filtering
    
    
    # p.processors([QRNSingleShotProcessor_Legacy])
    # p.processors([NRGFProcessor],           rp=True)  # Applies the Sunpy NRGF Filter
    # p.processors([MSGNProcessor],           rp=True)  # Applies the Sunpy Multiscale Gausian Norm
    # p.processors([QRNSingleShotProcessor], rp=True)  # Applies the Radial Filtering
    # p.processors([RHTProcessor],            rp=True)  # Applies the Rolling Hough Transform
    #


    #
    # p.processors(QRNpreProcessor, rp=True)  # Applies the Radial Filtering
    # p.processors(QRNradialFiltProcessor, rp=True)  # Applies the Radial Filtering
    # if p.is_debug():
    # else:
    
    # Runner(p).pointing_start()
    
    
    # Set the Parameters
    # p = default_run_single_params(batch_name, wave, config)
    
    # Set the Processes
    # p.processors([FidoTimeIntProcessor], rp=None)                        # Integrate several frames for S/N
    
    # p.processors([QRNpreProcessor],     rp=True)  # Learns the bounds of the dataset for QRN
    # p.processors([QRNradialFiltProcessor], rp=True)  # Applies the QRN Filter
    # #
    # p.putters([ImageProcessorCV], rp=True)  # Makes the PNGs from Fits
    # p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
    #
    # # Run the Code
    SingleRunner(p).start()

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_server()

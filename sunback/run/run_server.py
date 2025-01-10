"""This is the script to run on a server somewhere to process the images"""
# from sys import path
# path.append(path[0] + "/..")  # Adds higher directory to python modules path.
# a = [print(x) for x in path]

from sunback.fetcher.WebFitsFetcher import WebFitsFetcher
from sunback.processor.ImageProcessorCV import ImageProcessorCV, MultiImageProcessorCv
from sunback.processor.RHEProcessor import RHEProcessor
from sunback.processor.QRNProcessor import QRNSingleShotProcessor_Legacy
from sunback.processor.SunPyProcessor import AIA_PREP_Processor, NRGFProcessor, MSGNProcessor
from sunback.putter.AwsPutter import AwsPutter
from sunback.putter.DesktopPutter import DesktopPutter
from sunback.science.parameters import Parameters
from sunback.run import Runner, SingleRunner


def run_server(delay=10, debug=True, do_one='rainbow', stop=False):
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
    p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    # p.set_waves_to_do('0171')
    # p.overwrite_pngs(True)
    # p.write_video(False)
    # p.set_current_wave('rainbow')
    # # p.delete_old(True)

    # These settings might not look like they make sense but they make it work
    # p.png_frame_name = ['rhe(lev1p5)']
    p.msgn_targets(['lev1p5']) #, 'rhe(lev1p5)'
    p.rhe_targets(["lev1p5", 'msgn(lev1p5)']) #"lev1p5",
    p.png_frame_name = ['rhe(msgn)']

    compute = True
    # This is the right combination of processors for the server
    if compute:
        p.fetchers(WebFitsFetcher,                      )  # Gets Fits from JSOC Most Recent
        p.processors([MSGNProcessor],           rp=True)  # Applies the Sunpy Multiscale Gausian Norm
        p.processors([RHEProcessor],            rp=True)  # Applies the Radial Filtering
        p.processors([RHEProcessor],            rp=True)  # Applies the Radial Filtering
        p.putters([ImageProcessorCV],           rp=True)  # Turns Fits into Pngs

        p.putters([AwsPutter])  # Uploads the PNGs to AWS
    p.putters([DesktopPutter], rp=True)  # Runs the Desktop Background Sequence on PNGs



    # p.putters([MultiImageProcessorCv],      rp=True)  # Makes the PNGs from Fits
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

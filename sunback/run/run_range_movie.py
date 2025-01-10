from os.path import join

from sunback.processor.ImageProcessor import ImageProcessor
from sunback.processor.QRNradialFiltProcessor import QRNradialFiltProcessor
from sunback.fetcher.FidoFetcher import FidoFetcher
from sunback.processor.VideoProcessor import VideoProcessor
from sunback.science.parameters import Parameters
import run
import matplotlib.pyplot as plt
plt.ioff()


def run_range_movie(delay=10, debug=True, do_one="0304", stop=True, tstart='2015/11/05 00:00', tend='2015/11/05 06:00', cadence_minutes=20, fps=10):
    # Set the Parameters
    p = Parameters()
    p.delay_seconds(delay)
    p.run_type("Make Movie of Given Time Range")

    rng = "Range\\Range_{}".format(tstart.replace('/', '_').replace(' ', '_').replace(':', ''))
    p.batch_name(rng)

    p.do_one(do_one, True)
    p.stop_after_one(stop)
    p.is_debug(debug)
    p.do_recent(False)

    # Set the Times
    p.time_period(period=[tstart, tend])
    p.cadence_minutes(60 if debug else cadence_minutes)
    p.frames_per_second(fps)
    # p.resolution(2048)

    # Run Flags
    p.download_files(True)
    # p.overwrite_pngs(True)
    # p.delete_old(True)

    # Set the Processes
    p.fetchers(FidoFetcher())      # Gets Fits FIDO

    p.processors([RadialFiltProcessor()])  # Makes the PNGs from Fits

    p.putters([ImageProcessor()])  # Makes the PNGs from Fits
    p.putters([VideoProcessor()])  # Makes the PNGs into a Movie

    # Run the Code
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_range_movie()






    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    # p.putter(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    # p.putters(NullPutter(p))       # Does Nothing with the PNGS


    # p.sonify_limit(False)
    # p.remove_old_images(False)
    # p.make_compressed(True)
    # p.sonify_images(True, True)
    # p.sonify_images(False, False)
    # p.do_171(True)
    # p.do_304(True)

    # # p.bpm(150)
    #

































































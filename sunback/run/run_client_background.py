"""This is the script to run on a server somewhere to process the images"""
from sunback.run import SingleRunner
from sunback.science.parameters import Parameters
from sunback.putter.DesktopPutter import DesktopPutter
from sunback.fetcher.AwsImgFetcher import AwsImgFetcher
from sunback.fetcher.WebFitsFetcher import WebFitsFetcher

def run_client(delay=60, debug=False, do_one="rainbow", stop=True):
    p = Parameters()

    p.is_debug(debug)
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.batch_name("background_server_lingon")
    p.run_type("Client Sunback Daemon")
    p.do_orig = True
    p.speak_save = False
    p.use_drive = "G"
    p.do_parallel = False
    # Run Flags
    p.download_files(True)
    p.get_fits = True

    p.fetchers(AwsImgFetcher,)  # Gets Fits from www.gilly.space/sun
    p.putters([DesktopPutter])  # Sets the PNGs to the Desktop Background

    # Imageprocessor -> get_alphas() to adjust Upsilon

    # # Run the Code
    SingleRunner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_client()
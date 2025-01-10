"""This is the script to run to get pics from AWS and set them to the desktop background"""

from sunback.fetcher.AwsImgFetcher import AwsImgFetcher
from sunback.putter.DesktopPutter import DesktopPutter
from sunback.science.parameters import Parameters
from sunback.run import Runner


def run_background(delay=60, debug=True, stop=False):
    p = Parameters()
    p.is_debug(debug)
    p.stop_after_one(stop)
    p.delay_seconds(10 if debug else delay)
    p.batch_name("background_client")
    p.run_type("Background Updator")

    p.fetchers(AwsImgFetcher())   # Gets PNGs from S3 Daemon
    p.putters(DesktopPutter())    # Runs the Desktop Background Sequence on PNGs

    Runner(p).start()








if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_background()

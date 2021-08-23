
from fetcher.AwsFetcher import AwsFetcher
from putter.DesktopPutter import DesktopPutter
from science.parameters import Parameters
from run import Runner


def run_background(delay=30, debug=False, do_one=False, stop=True):
    p = Parameters()
    p.is_debug(debug)
    p.delay_seconds(5 if debug else delay)
    p.do_one(do_one, stop)
    p.stop_after_one(stop)
    
    p.fetcher(AwsFetcher(p))        # Gets PNGs from S3 Daemon
    p.putter(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    
    Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_background()

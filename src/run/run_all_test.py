"""This is the script to test all the run cases"""

# from fetcher.WebFitsFetcher import WebFitsFetcher
# from processor.RadialFiltProcessor import SRNradialFiltProcessor
# from putter.AwsPutter import AwsPutter
# from putter.DesktopPutter import DesktopPutter
from run_background import run_background
from run_recent_movie import run_recent_movie
from run_server import run_server
from run_range_movie import run_range_movie
# from science.parameters import Parameters
# from run import Runner
from run_range_multishot_movie import run_range_multishot_movie


def run_all_test(debug=True):
    run_background(stop=True, debug=debug)
    run_server(stop=True, debug=debug)
    run_recent_movie(stop=True, debug=debug)
    run_range_movie(stop=True, debug=debug)
    run_range_multishot_movie(stop=True, debug=debug)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_all_test()

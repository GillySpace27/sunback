"""This is the script to test all the run cases"""

# from src.fetcher.WebFitsFetcher import WebFitsFetcher
# from src.processor.RadialFiltProcessor import QRNradialFiltProcessor
# from src.putter.AwsPutter import AwsPutter
# from src.putter.DesktopPutter import DesktopPutter
from sunback.run.run_background import run_background
from sunback.run.run_recent_movie import run_recent_movie
from sunback.run.run_server import run_server
from sunback.run.run_range_movie import run_range_movie
# from src.science.parameters import Parameters
# from src.run import Runner
from sunback.run.run_range_multishot_movie import run_range_multishot_movie


def run_all_test(debug=True):
    run_background(stop=True, debug=debug)
    run_server(stop=True, debug=debug)
    run_recent_movie(stop=True, debug=debug)
    run_range_movie(stop=True, debug=debug)
    run_range_multishot_movie(stop=True, debug=debug)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_all_test()

"""This is the script to test all the run cases"""

from fetcher.WebFitsFetcher import WebFitsFetcher
from processor.RadialFiltProcessor import RadialFiltProcessor
from putter.AwsPutter import AwsPutter
from putter.DesktopPutter import DesktopPutter
from run_background import run_background
from run_recent_movie import run_recent_movie
from run_server import run_server
from science.parameters import Parameters
from run import Runner


def run_all_test():
    run_background(stop=True)
    run_server(stop=True)
    run_recent_movie(stop=True)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_all_test()

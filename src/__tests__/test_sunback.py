"""
Unit and regression test for the sunback package.
"""

# Import package, test suite, and other packages as needed
from unittest import TestCase

from movie.dep import sunback
import sys


def test_sunback_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sunback" in sys.modules


class TestSunback(TestCase):
    def test_download_image(self):
        local_path = "test_171.jpg"
        web_path = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_512_0171.jpg"
        self.assertFalse(sunback.Sunback().download_image(local_path, web_path))

    def test_update_background(self):
        local_path = "tests/test_171.jpg"
        self.assertFalse(sunback.Sunback().update_background(local_path, test=True))

    # def test_modify_image(self):
    #     self.fail()
    #
    # def test_loop(self):
    #     self.fail()
    #
    # def test_run(self):
    #     self.fail()
    #
    # def test_debug(self):
    #     self.fail()

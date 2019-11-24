"""
Unit and regression test for the sunback package.
"""

# Import package, test suite, and other packages as needed
from unittest import TestCase

import sunback
import pytest
import sys


def test_sunback_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sunback" in sys.modules


# class TestSunback(TestCase):
#     def test_download_image(self):
#         self.fail()
#
#     def test_update_background(self):
#         self.fail()
#
#     def test_modify_image(self):
#         self.fail()
#
#     def test_loop(self):
#         self.fail()
#
#     def test_run(self):
#         self.fail()
#
#     def test_debug(self):
#         self.fail()

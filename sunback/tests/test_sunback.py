"""
Unit and regression test for the sunback package.
"""

# Import package, test suite, and other packages as needed
import sunback
import pytest
import sys

def test_sunback_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sunback" in sys.modules


# def test_download_works():
#     """Try to run the download process"""
#     web_path="https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_{}}.jpg"
#     loc_path="test.jpg"
#     answer = sunback.download_image(web_path, loc_path, '193')
#     print(answer)
#     assert not answer


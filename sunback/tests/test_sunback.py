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

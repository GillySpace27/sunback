"""
SolarBackgroundUpdater
A program that downloads the most current images of the sun from the SDO satellite, then finds the most likely temperature in each pixel. Then it sets each of the images to the desktop background in series.
"""

# Add imports here
from sunback.sunback.sunback import *

# Handle versioneer
from sunback.sunback._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

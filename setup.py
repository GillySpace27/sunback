import sys
from setuptools import setup, find_packages

# If you have a docstring describing the package:
docstring = """Sets your desktop background to the most recent images of the Sun.
Solar Background Updater
A program that downloads the most current images of the sun from the SDO satellite,
then sets each of the images to the desktop background in series.
"""

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except OSError:
    long_description = docstring

setup(
    name="sunback",
    version="0.6.0",
    author="C. R. Gilly",
    author_email="chris.gilly@colorado.edu",
    description=docstring.split("\n")[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    url="https://github.com/GillySpace27/sunback",

    # Tell setuptools to look for packages in src/
    package_dir={"": "sunback"},
    packages=find_packages(where="sunback"),

    include_package_data=True,
    setup_requires=[] + pytest_runner,
    python_requires=">=3.0",

    install_requires=[
        "boto3",
        "matplotlib",
        "twine",
        "pillow",
        "appscript;platform_system=='Darwin'",
        "moviepy",
        "parfive",
        "playsound",
        "opencv-python",
        "numba",
        "beautifulsoup4",
        "sunpy",
        "scipy",
        "astropy",
        "html5lib",
        "numpy",
        "sunkit-image",
        "aiapy",
        "xarray"
    ],

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "Topic :: Desktop Environment",
        "Topic :: Desktop Environment :: Screen Savers",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
    ],

    entry_points={
        "console_scripts": [
            "sunback-run = sunback.run.run_client_background:run_client",
        ]
    },
)
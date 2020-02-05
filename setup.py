""" Sets your desktop background to the most recent images of the Sun.
Solar Background Updater
A program that downloads the most current images of the sun from the SDO satellite, then finds the most likely temperature in each pixel. Then it sets each of the images to the desktop background in series. 
"""
import sys
from setuptools import setup, find_packages
from sunback import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])

setup(
    # Self-descriptive entries which should always be present
    name='Sunback',
    author='Chris R. Gilly',
    author_email='chris.gilly@colorado.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='BSD-3-Clause',
    url='https://github.com/GillySpace27/sunback', # Website

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Required packages, pulls from pip if needed; do not use for Conda deployment
    install_requires=["sunpy[net]", "matplotlib", "twine", "pillow", "appscript;platform_system=='Darwin'"],

    platforms=['Windows', 'Linux', 'Mac OS-X'],            # Valid platforms your code works on, adjust to your flavor
                                        #'Linux','Mac OS-X','Unix',

    python_requires=">=3.0,<3.8",          # Python version restrictions

    classifiers=[
        # How mature is this project? Common values are
        #   1 - Planning
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Topic :: Desktop Environment',
        'Topic :: Desktop Environment :: Screen Savers',
        'Topic :: Multimedia :: Graphics :: Viewers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.2',
        # 'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3 :: Only',

        # Platforms
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
    ]

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)

from setuptools import setup, find_packages

setup(
    name="sunback",
    version="0.6.10",
    author="C. R. Gilly",
    author_email="chris.gilly@colorado.edu",
    description="Sets your desktop background to the most recent images of the Sun.",
    long_description="Sets your desktop background to the most recent images of the Sun.",
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    url="https://github.com/GillySpace27/sunback",

    # Root-level `sunback` contains the package `sunback`
    package_dir={"": "."},  # Project root contains the first `sunback`
    packages=find_packages(where="."),  # Discover packages starting at the project root

    include_package_data=True,
    python_requires=">=3.0",
    install_requires=[
        "boto3",
        "matplotlib",
        "twine",
        "pillow",
        "appscript;platform_system=='Darwin'",
        "moviepy",
        "parfive",
        "opencv-python",
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

    entry_points={
        "console_scripts": [
            "sunback-run = sunback.sunback.run.run_client_background:run_client",
            "sunback-serve = sunback.sunback.run.run_server_lingon:run_server_lingon"
        ]
    },
)
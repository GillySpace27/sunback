"""sunback: the NRT solar imagery pipeline, its research framework and the wallpaper client."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sunback")
except PackageNotFoundError:  # a checkout that was never pip-installed
    __version__ = "0+unknown"

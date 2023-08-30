from astropy.nddata import block_reduce
from PIL import Image
from os.path import dirname, abspath, join, isdir, split
from os import makedirs, getcwd, listdir
from time import time, localtime, strftime
import numpy as np

def reduce_array(frame, center, desired, func=np.nansum):
    # Reduce the size of the array
    resolution = frame.shape[0]
    center = center + 0
    reduce_amount = 1
    if resolution > desired:
        reduce_amount = int(resolution / desired)
        frame = block_reduce(frame, reduce_amount, func=func)
        center[0] /= reduce_amount
        center[1] /= reduce_amount
    return frame, center, reduce_amount


##  THUMBNAILS
def make_thumbs(rtPath):
    smallPath, rtPath, smallAWSpath, bigAWSpath = get_thumblinks(rtPath)
    imgDat = Image.open(rtPath)
    imgDat.thumbnail((512, 512))
    imgDat.save(smallPath)
    return smallPath, rtPath, smallAWSpath, bigAWSpath


def get_thumblinks(rtPath):
    """_summary_

    Parameters
    ----------
    rtPath : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    import os.path
    smallPath = rtPath.replace("/mod/", "/thmb/")
    bigAWSpath = os.path.join("renders", os.path.basename(rtPath))
    smallAWSpath = os.path.join("renders", "thumbs", os.path.basename(rtPath))

    for pth in [smallPath, rtPath]:
        the_dir = os.path.dirname(pth)
        makedirs(the_dir, exist_ok=True)

    return smallPath, rtPath, smallAWSpath, bigAWSpath

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
    smallPath, bigPath, arcPath = get_thumblinks(rtPath)
    imgDat = Image.open(rtPath)
    imgDat.thumbnail((512, 512))
    imgDat.save(smallPath)
    return smallPath, bigPath, arcPath


def get_thumblinks(rtPath):
    import os
    rep = '_mod'
    if rep in rtPath:
        orig = True
    else:
        orig = False
    
    filename = os.path.basename(rtPath)
    
    if "compare" in filename:
        name = filename
    
    elif not orig:
        name = filename #[-8:]
    else:
        name = filename.replace(".png", '_orig.png')
        name = name[-13:]
    
    arcPath = "renders/archive/" + "{}_{}".format(int(time()), name)
    smallPath = "renders/thumbs/" + name
    bigPath = 'renders/' + name
    makedirs("renders/thumbs/", exist_ok=True)
    return smallPath, bigPath, arcPath

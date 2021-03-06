#!/usr/bin/python

from PIL import Image
import numpy as np


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32)

    depth[depth_png == 0] = -1.
    
    depth = np.expand_dims(depth, axis=2)

    return depth

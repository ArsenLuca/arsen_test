# -*- coding: utf-8 -*-

from skimage.io import imread
from skimage.segmentation import felzenszwalb


def pff_graph_seg(img, scale=3.0, sigma=0.95, min_size=5):
    """ pff graph seg """
    if isinstance(img, str):
        img = imread(img)
    segments = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
    return segments

def test_seg():
    """ test pff graph seg """
    from skimage.data import coffee
    img = coffee()
    segments = felzenszwalb(img, scale=3.0, sigma=0.95, min_size=5)

import numpy as np
from astropy.io import fits


def load_mask(mask_path):
    '''
    AIM: Load mask FITS image and convert to boolean mask.
    '''

    mask_image = fits.getdata(mask_path)

    return ~(mask_image > 0)


def create_default_mask(shape):
    '''
    AIM: Create all-True mask for images without masks (i.e. a dimension-matched matrix of 1s).
    '''

    return np.ones(shape, dtype=bool)


def apply_mask(image, mask):
    '''
    AIM: Apply boolean mask to image...relatively straightforward.
    '''

    return image*mask
from astropy.io import fits
from reproject import reproject_interp

#this script is a 'placeholder' for reprojection! I eventually would like to reproject r-band images to the W1 or W3 scaling. the optical imaging contains too many pixels!

def reproject_image_to_reference(source_path, reference_path):
    '''
    AIM: reproject the source image (source_im_path) onto the reference image (ref_im_path) pixel scale.
    '''
    
    source_hdu = fits.open(source_path)[0]
    reference_hdu = fits.open(reference_path)[0]

    array, footprint = reproject_interp(source_hdu, reference_hdu.header)

    return array, footprint
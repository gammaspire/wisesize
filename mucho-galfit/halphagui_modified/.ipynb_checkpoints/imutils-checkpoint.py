#!/usr/bin/env python

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
#import ccdproc

from astropy.io.fits import Header
import numpy as np

def subtract_median_sky(data,getstd=False,getmedian=True,subtract=True):
    ''' subtract median sky from image data '''
    try:
        from photutils import make_source_mask
        mask = make_source_mask(data,nsigma=3,npixels=5,dilate_size=5)
        masked_data = np.ma.array(data,mask=mask)
        #clipped_array = sigma_clip(masked_data,cenfunc=np.ma.mean)

    except ImportError: # using a more recent version of photutils

        from photutils.segmentation import SegmentationImage
        from photutils.segmentation import detect_sources
        from photutils.background import Background2D, MedianBackground

        bkg_estimator = MedianBackground()

        bkg = Background2D(imdat,(50, 50),filter_size=(3, 3), bkg_estimator=bkg_estimator)
        threshold = 3 * bkg.background_rms
        segment_map = detect_sources(imdat, threshold, npixels=10)
        masked_data = np.ma.array(data,mask=mask)
    mean,median,std = sigma_clipped_stats(masked_data,sigma=3.0,cenfunc=np.ma.mean)
    if subtract:
        data -= median


    if getstd:
        return data,median,std
    
    elif getmedian:
        return data,median
    
    else:
        return data


def get_pixel_scale(imheader):
    ''' takes in image header and returns the pixel scale in arcsec  '''
    from astropy.wcs import WCS
    import astropy.units as u
    
    # get pixel scale from image header
    # convert from degrees/pix to arcsec/pix
    
    ## making more general - not all images have CD1_1 keyword
    #self.pscale = abs(float(self.image_header['CD1_1'])*3600)

    # found a better way to get the pixel scale
    
    image_wcs = WCS(imheader)        
    pscalex,pscaley = image_wcs.proj_plane_pixel_scales()
    
    pscale = pscalex.to(u.arcsec).value # convert deg/pix to arcsec/pix

    return pscale

def get_image_size_deg(imagename):
    ''' takes in image and header and returns (sizex,sizey) dimensions in deg  '''
    from astropy.wcs import WCS
    from astropy.io import fits

    image, imheader = fits.getdata(imagename,header = True)
    image_wcs = WCS(imheader)

    # found a better way to get the pixel scale
    
    image_wcs = WCS(imheader)        
    pscalex,pscaley = image_wcs.proj_plane_pixel_scales()
    


    nrow,ncol = image.shape

    sizex = ncol*pscalex
    sizey = nrow*pscaley
    #print()
    #print("in get_image_size_deg, sizex,sizey = ",sizex,sizey)
    #print()
    # return the size without the unit
    return sizex.value,sizey.value


def get_image_center_deg(imagename):
    ''' takes in image and header and returns ra and dec of center in deg  '''
    from astropy.wcs import WCS
    from astropy.io import fits
    
    image, imheader = fits.getdata(imagename,header = True)
    image_wcs = WCS(imheader)

    nrow,ncol = image.shape


    nrow_center = nrow/2
    ncol_center = ncol/2
    pixel_coord = np.array([[nrow_center,ncol_center]])

    radec = image_wcs.wcs_pix2world(pixel_coord,1,ra_dec_order=True)
    ra,dec = radec[0]
    #print("in get_image_center_deg, ra,dec = ",ra,dec)
    return ra,dec

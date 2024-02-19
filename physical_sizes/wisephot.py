#!/usr/bin/env python

'''
Much of this code is adapted from photwrapper.py from https://github.com/rfinn/halphagui
The aim is to perform photometry on W1 and W3 images from the unWISE catalog

'''

import os
homedir=os.getenv("HOME")

from photutils import detect_threshold, detect_sources

from photutils.segmentation import SourceCatalog

from photutils import EllipticalAperture

from photutils.isophote import EllipseGeometry, Ellipse
from photutils import aperture_photometry

#for smoothing the images...I think
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling import models, fitting

from astropy.stats import gaussian_sigma_to_fwhm
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization.mpl_normalize import ImageNormalize

from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats
from astropy.visualization import simple_norm

import scipy.ndimage as ndi

from matplotlib import pyplot as plt
from scipy.stats import scoreatpercentile

import numpy as np
import sys

#use image header information to define object RA, DEC; then this function will find the catalog index with RA, DEC closest to that header information

def getnearpos(array1,value1,array2,value2):
    idx = (np.sqrt((array1-value1)**2 + (array2-value2)**2)).argmin()
    return idx  

class ellipse():
    def __init__(self, obj_catalog_path, w1_image_path, w3_image_path, w1_psf_path, w3_psf_path, mask_path=None, objra=None, objdec=None, napertures=20):
        
        self.obj_cat = Table.read(obj_catalog_path)
        
        self.w1_image_path = w1_image_path
        self.w3_image_path = w3_image_path
        
        self.w1_im, self.w1_head = fits.getdata(self.w1_image_path, header=True)
        self.w3_im, self.w3_head = fits.getdata(self.w3_image_path, header=True)
        
        self.objra = self.w1_head['CENTRA']
        self.objdec = self.w3_head['CENTDEC']
        
        #find galaxy name
        self.galaxy_name = self.objcat['VFID'][getnearpos(self.obj_cat['RA'], self.objra, self.obj_cat['DEC'], self.objdec)]

        # get image dimensions - will use this to determine the max sma to measure
        self.yimage_max, self.ximage_max = self.w1_im.shape

        # check to see if obj position is passed in - need to do this for off-center objects
        if (objra is not None): # unmask central elliptical region around object
            # get wcs from mask image
            wcs = WCS(self.header)
            
            # get x and y coord of galaxy from (RA,DEC) using mask wcs
            #print(f"\nobject RA={self.objra:.4f}, DEC={self.objdec:.4f}\n")
            self.xcenter,self.ycenter = wcs.wcs_world2pix(self.objra,self.objdec,0)
            self.xcenter_ra = self.xcenter
            self.ycenter_dec = self.ycenter            
            # convert sma to pixels using pixel scale from mask wcs
            self.pixel_scale = wcs.pixel_scale_matrix[1][1]
            #self.objsma_pixels = self.objsma/(self.pixel_scale*3600)
            
        try:
            self.gain = self.header['GAIN']
        except KeyError:
            print("WARNING: no GAIN keyword in header. Setting gain=1")
            self.gain = 1.
        
        self.w1_psf_path = w1_psf_path
        self.w3_psf_path = w3_psf_path
        
        self.w1_psf = fits.getdata(self.w1_psf_path)
        self.w3_psf = fits.getdata(self.w3_psf_path)
      
        # the mask should identify all pixels in the cutout image that are not
        # associated with the target galaxy
        # these px will be ignored when defining the shape of the ellipse and when measuring the photometry

        if mask_path is not None:
            self.mask_image, self.mask_header = fits.getdata(mask_path,header=True)
            self.mask_flag = True
            # convert to boolean array, with bad pixels=True
            self.boolmask = np.array(self.mask_image,'bool')
            self.w1_masked = np.ma.array(self.w1_im, mask = self.boolmask)
            self.w3_masked = np.ma.array(self.w3_im, mask = self.boolmask)
        
        else:
            print('Not using a mask...nicht gut.')
            self.mask_flag = False
            self.w1_masked = self.w1_im
            self.w3_masked = self.w3_im

        #for plotting with matplotlib (I guess)
        self.use_mpl = True
        self.napertures = napertures

    def make_gauss2d_kernel(self):
        
        #initiate the fitting function
        fit_t = fitting.LevMarLSQFitter()

        amp_w1 = np.max(self.w1_psf)
        sigma = 3.5     #resulting model STD is too infinitesimal otherwise, so I chose a somewhat arbitrary starting value.
        y_max_w1, x_max_w1 = self.w1_psf.shape

        #set up symmetric W1 PSF model
        g2_w1 = models.Gaussian2D(amp_w1, x_max_w1/2, y_max_w1/2, sigma, sigma)

        #define indices on a grid corresponding to pixel coordinates
        yi_w1, xi_w1 = np.indices(self.w1_psf.shape)

        #fitting of model g2 to the W1 PSF
        g_w1 = fit_t(g2_w1, xi_w1, yi_w1, self.w1_psf)

        #same sigma as above, so no need to redefine variable
        amp_w3 = np.max(self.w3_psf)
        #redefined in case shapes are different
        y_max_w3, x_max_w3 = self.w3_psf.shape

        g2_w3 = models.Gaussian2D(amp_w3, x_max_w3/2, y_max_w3/2, sigma, sigma)

        yi_w3, xi_w3 = np.indices(self.w3_psf.shape)

        #fitting of model g2 to the W3 PSF
        g_w3 = fit_t(g2_w3, xi_w3, yi_w3, self.w3_psf)
        
        #note --> STDDEVs add in quadrature.
        sig_kernelx = np.sqrt(g_w3.x_stddev[0]**2 - g_w1.x_stddev[0]**2)
        sig_kernely = np.sqrt(g_w3.y_stddev[0]**2 - g_w1.y_stddev[0]**2)
        
        #fit new Gaussian kernel to the w1 image, ideally to smooth it in a comparable way to the blurred nature of W3
        self.smoothed_w1 = convolve(self.w1_im, Gaussian2DKernel(x_stddev=sig_kernelx, y_stddev=sig_kernely))
        self.smoothed_w1_masked = convolve(self.masked_w1, Gaussian2DKernel(x_stddev=sig_kernelx, y_stddev=sig_kernely))
        
        #Define FWHM using new kernel's sigma values. I only need the FWHM for the smoothed w1, as FWHM is only used here in the fitting of elliptical apertures (which I fit to smoothed W1).  
        #Rose used 3.5 for FWHM, I think for the H-alpha images. While 3.5 worked well for my galaxies, I will try this more tailored version first.
        
        rms_sigma = np.sqrt(sig_kernelx**2 + sig_kernely**2)/2
        self.fwhm = gaussian_sigma_to_fwhm*rms_sigma
    
    def plot_conv_mosaic(self, savefig=False):
        images = [self.w3_im,self.w1_im,self.smoothed_w1,np.abs(self.smoothed_w1-self.w1_im)]
        titles = ['W3 Image','W1 Unconvolved','W1 Convolved','W1 Residual']

        plt.figure(figsize=(11,9))
        for i, im in enumerate(images):
            plt.subplot(1,4,i+1)
            plt.imshow(im,origin='lower')
            plt.title(titles[i],fontsize=15)        
        
        if savefig:
            plt.savefig(f'{homedir}/Desktop/{self.galaxy_name}_kernel.png', dpi=100, bbox_inches='tight', pad_inches=0.2)
        
        plt.show()
    
    def run_two_image_phot(self,write1=False,savefig=False):

        self.make_gauss2d_kernel()
        self.plot_conv_mosaic(savefig=savefig)
        
        self.detect_objects()
        self.find_central_object() 
        self.get_ellipse_guess()
        self.measure_phot()
        self.get_all_frac_masked_pixels()
        self.calc_sb()
    
    def detect_objects(self, snrcut=1.5, npixels=10):
        ''' 
        run photutils detect_sources to find objects in fov.  
        you can specify the snrcut, and only pixels above this value will be counted.
        
        this also measures the sky noise as the mean of the threshold image
        '''
        # this is not right, because the mask does not include the galaxy
        # updating based on photutils documentation
        # https://photutils.readthedocs.io/en/stable/background.html
        
        # get a rough background estimate

        # I already compute sky sigma and store it in header
        # should look for that and use that as a threshold if it's available

        try:
            
            skystd = self.header['SKYSTD']
            self.sky_noise = skystd
            self.sky = self.header['SKYMED']
        except KeyError:
            print("WARNING: SKYSTD not found in ",self.galaxy_name)
            self.sky_noise = np.nan

        # get the value for halpha
        try:
            if self.header2 is not None:
                self.sky_noise2 = self.header2['SKYSTD']
                self.sky2 = self.header['SKYMED']
            else:
                print("WARNING: SKYSTD not found in ",self.image2_name)
                self.sky_noise2 = np.nan
                self.sky2 = np.nan
        except KeyError:
            print("WARNING: SKYSTD not found in ",self.image2_name)
            self.sky_noise2 = np.nan
            self.sky2 = np.nan
        
        if self.mask_flag:
            if self.sky_noise is not np.nan:
                self.threshold = self.sky_noise
            else:
                self.threshold = detect_threshold(self.image, nsigma=snrcut,mask=self.boolmask)
            self.segmentation = detect_sources(self.image, self.threshold, npixels=npixels, mask=self.boolmask)
            #self.cat = source_properties(self.image, self.segmentation, mask=self.boolmask)
            self.cat = SourceCatalog(self.image, self.segmentation, mask=self.boolmask)
            if self.image2 is not None:
                # measure halpha properties using same segmentation image
                self.cat2 = SourceCatalog(self.image2, self.segmentation, mask=self.boolmask)
        else:
            if self.sky_noise is not np.nan:
                self.threshold = self.sky_noise
            else:
            
                self.threshold = detect_threshold(self.image, nsigma=snrcut)
            self.segmentation = detect_sources(self.image, self.threshold, npixels=npixels)
            #self.cat = source_properties(self.image, self.segmentation)
            self.cat = SourceCatalog(self.image, self.segmentation)
            if self.image2 is not None:
                # measure halpha properties using same segmentation image
                self.cat2 = SourceCatalog(self.image2, self.segmentation)
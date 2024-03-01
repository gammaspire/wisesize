#!/usr/bin/env python

'''
Much of this code is adapted from photwrapper.py from https://github.com/rfinn/halphagui
The aim is to perform photometry on W1 and W3 images from the unWISE catalog

'''

import os
homedir=os.getenv("HOME")

from photutils import detect_threshold, detect_sources

from photutils.segmentation import SourceCatalog

from photutils import EllipticalAperture, CircularAperture

from photutils.isophote import EllipseGeometry, Ellipse
from photutils import aperture_photometry
from photutils.detection import find_peaks

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

def get_fraction_masked_pixels(catalog,objectIndex):
    """ 
    get area in the segmentation image, 
    masked area, and fraction of pixels masked
    """

    objNumber = catalog.label[objectIndex]
    dat = catalog.data[objectIndex]    
    masked_dat = catalog.data_ma[objectIndex]

    # create flag for pixels associated with object in segmentation map
    goodflag = catalog.segment[objectIndex] == objNumber

    # get number of pixels in the original segmentation image
    number_total = np.sum(goodflag)

    number_masked = number_total - np.sum(goodflag & masked_dat.mask)

    return number_total, number_masked, number_masked/number_total

class wise_ellipse():
    
    def __init__(self, obj_catalog_path, w1_image_path, w3_image_path, w1_psf_path, w3_psf_path, mask_path=None, objra=None, objdec=None):
        
        self.obj_cat = Table.read(obj_catalog_path)
        
        self.w1_image_path = w1_image_path
        self.w3_image_path = w3_image_path
        
        self.w1_im, self.w1_head = fits.getdata(self.w1_image_path, header=True)
        self.w3_im, self.w3_head = fits.getdata(self.w3_image_path, header=True)
        
        self.objra = self.w1_head['CENTRA']
        self.objdec = self.w3_head['CENTDEC']
        
        #find galaxy name
        self.galaxy_name = self.obj_cat['VFID'][getnearpos(self.obj_cat['RA'], self.objra, self.obj_cat['DEC'], self.objdec)]

        # get image dimensions - will use this to determine the max sma to measure
        self.yimage_max, self.ximage_max = self.w1_im.shape

        # check to see if obj position is passed in - need to do this for off-center objects
        if (objra is not None): # unmask central elliptical region around object
            # get wcs from mask image
            wcs = WCS(self.w1_head)
            
            # get x and y coord of galaxy from (RA,DEC) using mask wcs
            #print(f"\nobject RA={self.objra:.4f}, DECx={self.objdec:.4f}\n")
            self.xcenter,self.ycenter = wcs.wcs_world2pix(self.objra,self.objdec,0)
            self.xcenter_ra = self.xcenter
            self.ycenter_dec = self.ycenter            
            # convert sma to pixels using pixel scale from mask wcs
            self.pixel_scale = wcs.pixel_scale_matrix[1][1]
            #self.objsma_pixels = self.objsma/(self.pixel_scale*3600)
            
        try:
            self.gain = self.w1_head['GAIN']
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

    def make_gauss2d_kernel(self):
        
        #initiate the fitting function
        fit_t = fitting.LevMarLSQFitter()

        amp_w1 = np.max(self.w1_psf)
        sigma = 5     #resulting model STD is too infinitesimal otherwise, so I chose a somewhat arbitrary starting value.
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
        self.sig_kernelx = np.sqrt(g_w3.x_stddev[0]**2 - g_w1.x_stddev[0]**2)
        self.sig_kernely = np.sqrt(g_w3.y_stddev[0]**2 - g_w1.y_stddev[0]**2)
        
        #fit new Gaussian kernel to the w1 image, ideally to smooth it in a comparable way to the blurred nature of W3
        self.smoothed_w1 = convolve(self.w1_im, Gaussian2DKernel(x_stddev=self.sig_kernelx, y_stddev=self.sig_kernely))
        #self.smoothed_w1_masked = convolve(self.w1_masked, Gaussian2DKernel(x_stddev=self.sig_kernelx, y_stddev=self.sig_kernely))
        
        #Define FWHM using new kernel's sigma values. I only need the FWHM for the smoothed w1, as FWHM is only used here in the fitting of elliptical apertures (which I fit to smoothed W1).  
        #Rose used 3.5 for FWHM, I think for the H-alpha images. While 3.5 worked well for my galaxies, I will try this more tailored version first.
        
        rms_sigma = np.sqrt(self.sig_kernelx**2 + self.sig_kernely**2)/2
        self.fwhm = gaussian_sigma_to_fwhm*rms_sigma
        
        print(f'STDDEV W3 --> x={g_w3.x_stddev[0]}, y={g_w3.y_stddev[0]}')
        print(f'STDDEV W1 --> x={g_w1.x_stddev[0]}, y={g_w1.y_stddev[0]}')
        print(f'FITTED PSF --> x={self.sig_kernelx}, y={self.sig_kernely}')
        
    #quick plot check of the fitted PSF found with self.make_gauss2d_kernel()
    def check_fitted_psf(self):
        
        plt.figure()
        plt.imshow(Gaussian2DKernel(x_stddev=self.sig_kernelx, y_stddev=self.sig_kernely))
        plt.title(r'$\sigma_x$='+str(round(self.sig_kernelx,5))+r', $\sigma_y$='+str(round(self.sig_kernely,5)),fontsize=14)
        plt.xlabel('x px',fontsize=14)
        plt.ylabel('y px',fontsize=14)
        plt.show()    
    
    def plot_conv_mosaics(self, savefig=False):
        
        norm_w1 = self.smoothed_w1/np.max(self.smoothed_w1)
        norm_w3 = self.w3_im/np.max(self.w3_im)
        
        im_rat_w1 = np.abs(self.smoothed_w1-self.w1_im)   #not normalized
        im_rat_w3 = np.abs(norm_w1-norm_w3)   #normalized
        
        percentile1 = 0.5
        percentile2 = 99.5
        
        images = [self.w1_im, self.smoothed_w1, im_rat_w1,
                 self.w3_im, self.smoothed_w1, im_rat_w3]
        titles = ['W1 Image',r'W1$_{conv}$ Image',r'W1 - W1$_{conv}$ (W1 stretch)',
                  'W3 Image',r'W1$_{conv}$ Image',r'W3 - W1$_{conv}$ (W3 stretch)']
        
        v1 = [scoreatpercentile(self.w1_masked,percentile1),
            scoreatpercentile(self.w3_masked,percentile1),
             scoreatpercentile(norm_w3,percentile1)]
        
        v2 = [scoreatpercentile(self.w1_masked,percentile2),
            scoreatpercentile(self.w3_masked,percentile2),
             scoreatpercentile(norm_w3,percentile2)]
        
        norms = [
         simple_norm(self.w1_masked,'asinh',max_percent=percentile2,min_cut=v1[0],max_cut=v2[0]), 
         simple_norm(self.w1_masked, 'asinh', max_percent=percentile2, min_cut=v1[0], max_cut=v2[0]), 
         simple_norm(self.w1_masked, 'asinh', max_percent=percentile2, min_cut=v1[0], max_cut=v2[0]),
         simple_norm(self.w3_masked,'asinh',max_percent=percentile2,min_cut=v1[1],max_cut=v2[1]),
         simple_norm(self.w1_masked, 'asinh', max_percent=percentile2, min_cut=v1[0], max_cut=v2[0]),
         simple_norm(norm_w3, 'asinh', max_percent=percentile2, min_cut=v1[2], max_cut=v2[2])
                ]
        
        plt.figure(figsize=(11,9))
        for i, im in enumerate(images):
            plt.subplot(2,3,i+1)
            plt.imshow(im,origin='lower',norm=norms[i])
            plt.title(titles[i],fontsize=15)        
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.005)
        
        if savefig:
            plt.savefig(f'{homedir}/Desktop/{self.galaxy_name}_kernel.png', dpi=100, bbox_inches='tight', pad_inches=0.2)
        
        plt.show()
    
    def run_two_image_phot(self,savefig=False):

        self.make_gauss2d_kernel()
        self.plot_conv_mosaics(savefig=savefig)
        
        self.detect_objects()
        self.find_central_object() 
        self.get_ellipse_guess()
        self.measure_phot()
        self.get_all_frac_masked_pixels()
        self.calc_sb()
    
    def check_ptsources(self,box_size=5):
        '''
        -perform elliptical photometry on point sources in an image 
         (i.e., prominent foreground stars)
        -should only run if there is more than one object in the detection 
         list with a peak value above the given threshold in W1 image --> 
         must be 5-sigma above the background (median)
        -this will be a decent indicator of how well the PSF applied to W1 replicates the
         blurring effects in the W3 image
        -output will also include figure of encircled detected sources
        '''
        
        im_std = np.std(self.smoothed_w1)
        im_median = np.median(self.smoothed_w1)
        im_threshold = im_median + (5*im_std)
        
        tbl = find_peaks(self.smoothed_w1, threshold=im_threshold, box_size=box_size)
        #assumes galaxy center lies ~center of the image
        xc, yc = len(self.smoothed_w1)/2, len(self.smoothed_w1)/2
        
        #flag to isolate objects that are NOT the galaxy. we want the star(s), after all
        galaxy_dist_flag = (np.sqrt((tbl['x_peak'] - xc)**2 + (tbl['y_peak'] - yc)**2))>2
        
        if len(tbl)>1:
            #apply flag if more than just the galaxy is in the table
            tbl_stars = tbl[galaxy_dist_flag]
            
            print('photutils detected the following objects:')
            print(tbl)
            
            #isolate star with brightest peak --> more likely to have more robust photometry
            tbl_brightest_star = tbl_stars[tbl_stars['peak_value']==np.max(tbl_stars['peak_value'])]
                        
            plt.figure(figsize=(3,3))
            positions=np.transpose((tbl['x_peak'],tbl['y_peak']))
            #silly photutils.aperture coordinate conventions require floats ONLY
            self.main_center = (tbl_brightest_star['x_peak'].data[0],
                                tbl_brightest_star['y_peak'].data[0])
            self.xcenter = self.main_center[0]
            self.ycenter = self.main_center[1]
            
            apertures=CircularAperture(positions,r=5)
            main_aperture = CircularAperture(self.main_center,r=5)
            plt.imshow(self.smoothed_w1, cmap='Greys_r', origin='lower')
            apertures.plot(color='r')
            main_aperture.plot(color='cyan')
            plt.show()
            
            self.get_ellipse_guess(star=True)
            self.measure_phot(star=True)
            self.calc_sb()
            self.plot_profiles(flux_yscale='linear',star=True)
            
        else:
            'No stars detected! Your box_size might be too high/low or there are no prominent foreground stars in your image.'
            
 
    def detect_objects(self, snrcut=1.5, npixels=10):
        ''' 
        run photutils detect_sources to find objects in fov.  
        you can specify the snrcut, and only pixels above this value will be counted.
        '''
        
        try:
            
            skystd = self.w1_head['SKYSTD']
            self.sky_noise = skystd
            self.sky = self.w1_head['SKYMED']
        except KeyError:
            print("WARNING: SKYSTD not found in ",self.galaxy_name)
            self.sky_noise = np.nan
            
        try:
            if self.w3_head is not None:
                self.sky_noise2 = self.w3_head['SKYSTD']
                self.sky2 = self.w3_head['SKYMED']
            else:
                print("WARNING: SKYSTD not found in ",self.galaxy_name)
                self.sky_noise2 = np.nan
                self.sky2 = np.nan
        except KeyError:
            print("WARNING: SKYSTD not found in ",self.galaxy_name)
            self.sky_noise2 = np.nan
            self.sky2 = np.nan
        
        if self.mask_flag:
            if self.sky_noise is not np.nan:
                self.threshold = self.sky_noise
            else:
                self.threshold = detect_threshold(self.smoothed_w1, nsigma=snrcut,mask=self.boolmask)
            self.segmentation = detect_sources(self.smoothed_w1, self.threshold, npixels=npixels, mask=self.boolmask)
            self.cat = SourceCatalog(self.smoothed_w1, self.segmentation, mask=self.boolmask)
            self.cat2 = SourceCatalog(self.w3_im, self.segmentation, mask=self.boolmask)
        else:
            if self.sky_noise is not np.nan:
                self.threshold = self.sky_noise
            else:
                self.threshold = detect_threshold(self.smoothed_w1, nsigma=snrcut)
                
            self.segmentation = detect_sources(self.smoothed_w1, self.threshold, npixels=npixels)
            self.cat = SourceCatalog(self.smoothed_w1, self.segmentation)
            self.cat2 = SourceCatalog(self.w3_im, self.segmentation)
            
    def find_central_object(self):
        ''' 
        find the central object in the image and get its objid in segmentation image.
        object is stored as self.objectIndex
        '''
        
        if self.objra is not None:
            #print("getting object position from RA and DEC")
            xc = self.xcenter_ra
            yc = self.ycenter_dec
        else:
            ydim,xdim = self.w3_im.shape
            xc = xdim/2
            yc = ydim/2            
        distance = np.sqrt((np.ma.array(self.cat.xcentroid) - xc)**2 + (np.ma.array(self.cat.ycentroid) - yc)**2)        
        # save object ID as the row in table with source that is closest to center
        # check to see if len(distance) is > 1

        if len(distance) > 1:
            try:
                self.objectIndex = np.arange(len(distance))[(distance == min(distance))][0]
            except IndexError:
                print("another $#@$# version change???",np.arange(len(distance))[(distance == min(distance))],len(distance))
                print('x vars: ',self.cat.xcentroid, xc)
                print('y vars: ', self.cat.ycentroid, yc)                
                print(self.cat)
                sys.exit()
        else:
            self.objectIndex = 0
            print("WARNING: only one object in the SourceCatalog!",distance)
        #print(self.objectIndex)
 
        distance = np.sqrt((np.ma.array(self.cat2.xcentroid) - xc)**2 + (np.ma.array(self.cat2.ycentroid) - yc)**2)        
        # save object ID as the row in table with source that is closest to center
        self.objectIndex2 = np.arange(len(distance))[(distance == min(distance))][0]
            
        if self.objra is not None:
            # check that distance of this object is not far from the original position
            xcat = self.cat.xcentroid[self.objectIndex]
            ycat = self.cat.ycentroid[self.objectIndex]

            offset = np.sqrt((xcat-self.xcenter_ra)**2 + (ycat-self.ycenter_dec)**2)
            if offset > 100:
                print()
                print("Hold the horses - something is not right!!!")
            
    def get_mask_from_segmentation(self):
        # create a copy of the segmentation image
        # replace the object index values with zeros        
        segmap = self.segmentation.data == self.cat.label[self.objectIndex]

        # subtract this from segmentation

        mask_data = self.segmentation.data - segmap*self.cat.label[self.objectIndex]
        # smooth 
        segmap_float = ndi.uniform_filter(np.float64(mask_data), size=10)
        mask = segmap_float > 0.5

        self.mask_image = mask
        self.boolmask = mask
        self.mask_flag = True
        
    def get_ellipse_guess(self, r=2.5, star=False):
        '''
        this gets the guess for the ellipse geometry from the detection catalog 
        '''
        
        #for performing elliptical photometry on point sources!
        if star:
            self.xcenter = self.main_center[0]
            self.ycenter = self.main_center[1]
            self.position = self.main_center
            self.sma = 3   #approximated for smoothed W1 point sources
            self.start_size = self.sma
            self.b = self.sma
            self.eps = 0   #no ellipticity for circles
            self.theta = 0   #no position angle...just a little ol' point source
        
        else:
            obj = self.cat[self.objectIndex]
            self.xcenter = obj.xcentroid
            self.ycenter = obj.ycentroid

            self.position = (obj.xcentroid, obj.ycentroid)

            #print(self.position,self.xcenter,obj.xcentroid,self.ycenter,obj.ycentroid)

            self.sma = obj.semimajor_sigma.value * r
            self.start_size = self.sma
            self.b = obj.semiminor_sigma.value * r
            self.eps = 1 - self.b/self.sma
            # orientation is angle in radians, CCW relative to +x axis
            t = obj.orientation.value
            #print('inside get_ellipse_guess, orientation = ',obj.orientation)
            if t < 0: # convert to positive angle wrt +x axis
                self.theta = np.pi+obj.orientation.to(u.rad).value
            else:
                self.theta = obj.orientation.to(u.rad).value # orientation in radians
            # EllipticalAperture gives rotation angle in radians from +x axis, CCW
        
        try:
            self.aperture = EllipticalAperture(self.position, self.sma, self.b, theta=self.theta)
        except ValueError:
            print("\nTrouble in paradise...")
            print(self.position,self.sma,self.b,self.theta)
            sys.exit()
        
        # EllipseGeometry using angle in radians, CCW from +x axis
        self.guess = EllipseGeometry(x0=self.xcenter,y0=self.ycenter,sma=self.sma,eps = self.eps, pa = self.theta)
    
    def measure_phot(self, star=False):
        '''
        # rmax is max radius to measure ellipse
        # could cut this off based on SNR
        # or could cut this off based on enclosed flux?
        # or could cut off based on image dimension, and do the cutting afterward
        
        #rmax = 2.5*self.sma
        '''
        # rmax is set according to the image dimensions
        # look for where the semi-major axis hits the edge of the image
        # could by on side (limited by x range) or on top/bottom (limited by y range)
        # 
        
        if star:
            self.fwhm = 0.2/0.5
            rmax = 2.5*self.sma
        
        else:
            rmax = np.min([(self.ximage_max - self.xcenter)/abs(np.cos(self.theta)),\
                           (self.yimage_max - self.ycenter)/abs(np.sin(self.theta))])
        
        index = np.arange(80)
        apertures = (index+1)*.5*self.fwhm*(1+(index+1)*.1)
        # cut off apertures at edge of image
        self.apertures_a = apertures[apertures < rmax]
        print('Number of apertures = ',len(self.apertures_a))
        self.apertures_b = (1.-self.eps)*self.apertures_a
        self.area = np.pi*self.apertures_a*self.apertures_b # area of each ellipse

        self.flux1 = np.zeros(len(self.apertures_a),'f')
        self.flux2 = np.zeros(len(self.apertures_a),'f')
        self.flux1_err = np.zeros(len(self.apertures_a),'f')
        self.flux2_err = np.zeros(len(self.apertures_a),'f')
        self.allellipses = []
        for i in range(len(self.apertures_a)):
            # EllipticalAperture takes rotation angle in radians, CCW from +x axis
            ap = EllipticalAperture((self.xcenter, self.ycenter),self.apertures_a[i],self.apertures_b[i],self.theta)#,ai,bi,theta) for ai,bi in zip(a,b)]
            self.allellipses.append(ap)

            if star:
                self.phot_table1 = aperture_photometry(self.smoothed_w1, ap, method = 'subpixel', 
                                                       subpixels=1)
                self.phot_table2 = aperture_photometry(self.w3_im, ap, method = 'subpixel', 
                                                       subpixels=1)
            else:
                if self.mask_flag:
                    # check for nans, and add them to the mask
                    nan_mask = self.w1_im == np.nan
                    combined_mask =  self.boolmask | nan_mask
                    self.phot_table1 = aperture_photometry(self.smoothed_w1, ap, mask=combined_mask)
                    self.phot_table2 = aperture_photometry(self.w3_im, ap, mask=combined_mask)
                else:
                    # subpixel is the method used by Source Extractor
                    self.phot_table1 = aperture_photometry(self.smoothed_w1, ap, method = 'subpixel', 
                                                           subpixels=5)
                    self.phot_table2 = aperture_photometry(self.w3_im, ap, method = 'subpixel', 
                                                           subpixels=5)
            
            self.flux1[i] = self.phot_table1['aperture_sum'][0]
            self.flux2[i] = self.phot_table2['aperture_sum'][0]
            
            # calculate noise
            self.flux1_err[i] = self.get_noise_in_aper(self.flux1[i], self.area[i])
            self.flux2_err[i] = self.get_noise_in_aper(self.flux2[i], self.area[i])
    
    def calc_sb(self):
        # calculate surface brightness in each aperture

        # first aperture is calculated differently
        self.sb1 = np.zeros(len(self.apertures_a),'f')
        self.sb1_err = np.zeros(len(self.apertures_a),'f')

        self.sb1[0] = self.flux1[0]/self.area[0]
        self.sb1_err[0] = self.get_noise_in_aper(self.flux1[0], self.area[0])/self.area[0]
        # outer apertures need flux from inner aperture subtracted
        for i in range(1,len(self.area)):
            self.sb1[i] = (self.flux1[i] - self.flux1[i-1])/(self.area[i]-self.area[i-1])
            self.sb1_err[i] = self.get_noise_in_aper((self.flux1[i] - self.flux1[i-1]),(self.area[i]-self.area[i-1]))/(self.area[i]-self.area[i-1])

        # calculate SNR to follow Becky's method of cutting off analysis where SNR = 2
        self.sb1_snr = np.abs(self.sb1/self.sb1_err)
        # repeat for image 2 if it is provided
        self.sb2 = np.zeros(len(self.apertures_a),'f')
        self.sb2_err = np.zeros(len(self.apertures_a),'f')
        self.sb2[0] = self.flux2[0]/self.area[0]
        self.sb2_err[0] = self.get_noise_in_aper(self.flux2[0], self.area[0])/self.area[0]
        for i in range(1,len(self.area)):
            self.sb2[i] = (self.flux2[i] - self.flux2[i-1])/(self.area[i]-self.area[i-1])
            self.sb2_err[i] = self.get_noise_in_aper((self.flux2[i] - self.flux2[i-1]),(self.area[i]-self.area[i-1]))/(self.area[i]-self.area[i-1])
        self.sb2_snr = np.abs(self.sb2/self.sb2_err)
    
    def get_noise_in_aper(self, flux, area):
        ''' calculate the noise in an area '''
        if self.sky_noise is not None:
            noise_e = np.sqrt(flux*self.gain + area*self.sky_noise*self.gain)
            noise_adu = noise_e/self.gain
        else:
            noise_adu = np.nan
        return noise_adu
    
    def get_all_frac_masked_pixels(self):
        # set all objects' masked fraction equal to this value
        # in the end, I will only keep the value for the central object...
        ntotal,nmasked,frac_masked = get_fraction_masked_pixels(self.cat,self.objectIndex)
        allfmasked = frac_masked*np.ones(len(self.cat))
        self.cat.add_extra_property('MASKEDFRAC',allfmasked)
        self.masked_fraction = allfmasked
        self.pixel_area = ntotal
        self.masked_pixel_area = ntotal - nmasked
    
    def plot_profiles(self,galaxyname=None,flux_yscale='log',star=False,savefig=False):
        ''' enclosed flux and surface brightness profiles, save figure '''
        
        #for some W3 images, especially if galaxies/point sources are relatively faint, the apertures will enclose pixels with negative values, which riddle the background. statistically, the larger amount of background enclosed the more negative pixels, and thus a potentially lower enclosed flux than the previous aperture. this is no bueno, so one option to try is to cut off the flux/SB values when the (flux[i+1]-flux[i])<0, or some variant of this condition. another option is to find where W1 curve plateaus; however, it may likely never plateau since increasingly larger apertures will enclose more pixels, and potentially non-central objects (e.g., foreground stars).
        
        cutoff_index=None            #define variable outside of loop
        diff = np.diff(self.flux2)   #outputs array with the difference values between consecutive elements
        for n in diff:               #for every element in diff, find the first which is negative
            if n < 0:
                index = np.min(np.where(diff==n))  #could be multiple indices where the difference is
                                                   #the same value as n; pick the first one.

                cutoff_index = index+1             #remove that and subsequent elements
                break                              #I will use this cutoff index for flux and sb of both galaxies

        #create boolean array to 'trim' the fluxes/sbs/apertures from cutoff_index onwards
        trim_flag = np.zeros(len(self.flux2),dtype='bool')   #len(flux2)=len(flux1)=len(apertures_a), etc.
        trim_flag[0:cutoff_index] = True     
        
        plt.close("all")        
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(wspace=.3)
        
        plt.subplot(2,3,1)
        plt.errorbar(self.apertures_a[trim_flag],self.flux1[trim_flag],self.flux1_err[trim_flag],fmt='r.')
        plt.title('Smoothed W1 (3.4-micron)')
        if star:
            plt.title('Smoothed W1 Star')
        plt.ylabel('Enclosed flux')
        if flux_yscale=='log':
            plt.gca().set_yscale('log')
        
        plt.subplot(2,3,2)
        plt.errorbar(self.apertures_a[trim_flag],self.flux2[trim_flag],
                     self.flux2_err[trim_flag],fmt='b.')
        plt.title('W3 (12-micron)')
        if star:
            plt.title('Unsmoothed W3 Star')
        if flux_yscale=='log':
            plt.gca().set_yscale('log')
        
        plt.subplot(2,3,3)
        label1='Smoothed W1'
        label2='W3'
        if star:
            label1='Smoothed W1 Star'
            label2='W3 Star'
        plt.errorbar(self.apertures_a[trim_flag], self.flux1[trim_flag]/np.max(self.flux1[trim_flag]), 
                     self.flux1_err[trim_flag]/np.max(self.flux1_err[trim_flag]), fmt='r.',label=label1)
        plt.errorbar(self.apertures_a[trim_flag], self.flux2[trim_flag]/np.max(self.flux2[trim_flag]), 
                     self.flux2_err[trim_flag]/np.max(self.flux2_err[trim_flag]), fmt='b.',label=label2)
        plt.title('Normalized Curves')
        if flux_yscale=='log':
            plt.gca().set_yscale('log')
        plt.legend()
        
        plt.subplot(2,3,4)
        plt.errorbar(self.apertures_a[trim_flag],self.sb1[trim_flag],self.sb1_err[trim_flag],fmt='r.')
        plt.ylabel('Surface Brightness')
        plt.xlabel('Semi-Major Axis [px]')
        plt.gca().set_yscale('log')
        
        plt.subplot(2,3,5)
        plt.errorbar(self.apertures_a[trim_flag],self.sb2[trim_flag],
                     self.sb2_err[trim_flag],fmt='b.')
        plt.xlabel('Semi-Major Axis [px]')
        plt.gca().set_yscale('log')
        
        plt.subplot(2,3,6)
        plt.errorbar(self.apertures_a[trim_flag], self.sb1[trim_flag]/np.max(self.sb1[trim_flag]), 
                     self.sb1_err[cutoff_index]/np.max(self.sb1_err[cutoff_index]), fmt='r.')
        plt.errorbar(self.apertures_a[trim_flag], self.sb2[trim_flag]/np.max(self.sb2[trim_flag]), 
                     self.sb2_err[trim_flag]/np.max(self.sb2_err[trim_flag]), fmt='b.')
        plt.gca().set_yscale('log')
        plt.xlabel('Semi-Major Axis [px]')
        
        if savefig:
            plt.savefig(homedir+'/Desktop/'+galaxyname+'-enclosed-flux.png')
        
        plt.show()
    
if __name__ == '__main__':
    print('Use: \n e_obj = ellipse(obj_catalog_path, w1_image_path, w3_image_path, w1_psf_path, w3_psf_path, mask_path=None, objra=None, objdec=None) \n e_obj.run_two_image_phot() \n e_obj.check_fitted_psf() \n e_obj.plot_profiles(ax_scale=log,savefig=False) \n e_obj.check_ptsources(box_size=5)')
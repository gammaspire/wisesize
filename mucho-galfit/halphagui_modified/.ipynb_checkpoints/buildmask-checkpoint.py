#!/usr/bin/env python

'''
PURPOSE:

The goal of the program is to create a mask for a galaxy image to mask
out other objects within the cutout area.

USAGE:


you just need to run this on R-band images.


PROCEDURE:


REQUIRED MODULES:
   os
   astropy
   numpy
   argsparse
   matplotlib
   scipy

USAGE:

* if running on wise images, try:

python ~/github/halphagui/maskwrapper.py --image AGC006015-unwise-1640p454-w1-img-m.fits --ngrow 1 --sesnr 2 --minarea 5 --auto


NOTES:
- rewrote using a class

TESTING

objparams = [self.defcat.cat['RA'][self.igal],self.defcat.cat['DEC'][self.igal],mask_scalefactor*self.radius_arcsec[self.igal],self.BA[self.igal],self.PA[self.igal]+90]


python ~/github/halphagui/maskwrapper.py --image VFID0610-NGC5985-INT-20190530-p040-R.fits --haimage VFID0610-NGC5985-INT-20190530-p040-CS.fits --sepath ~/github/halphagui/astromatic/ --gaiapath /home/rfinn/research/legacy/gaia-mask-dr9.virgo.fits --objra 234.90448 --objdec 59.33198 --objsma 139.25 --objBA .496 --objPA 104.646


'''

import os
import sys
import numpy as np
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Tophat2DKernel, convolve
from astropy.convolution.kernels import CustomKernel
from astropy.table import Table
from astropy.coordinates import SkyCoord

from scipy.stats import scoreatpercentile

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import patches

# import gaia function to get stars within region
from get_gaia_stars import gaia_stars_in_rectangle

import imutils

try:
    from photutils.segmentation import detect_threshold, detect_sources
    #from photutils import source_properties
    from photutils.segmentation import SourceCatalog    
    from photutils.segmentation import deblend_sources
except ModuleNotFoundError:
    warnings.warn("Warning - photutils not found")
except ImportError:
    print("got an import error with photutils - check your version number")

import timeit

defaultcat='default.sex.HDI.mask'

#####################################
###  FUNCTIONS
#####################################

def remove_central_objects(mask, sma=20, BA=1, PA=0, xc=None,yc=None):
    """ 
    find any pixels within central ellipse and set their values to zero 

    PARAMS:
    mask = 2D array containing masked pixels, like from SE segmentation image
    sma = semi-major axis in pixels
    BA = ratio of semi-minor to semi-major axes
    PA = position angle, measured in degree counter clockwise from +x axis

    OPTIONAL ARGS:
    xc = center of ellipse in pixels; assumed to be center of image if xc is not specified
    yc = center of ellipse in pixels; assumed to be center of image if yc is not specified
    
    RETURNS:
    newmask = copy of input mask, with pixels within ellipse set equal to zero

    """
    # changing the xmax and ymax - if the ellipse looks wrong, then swap back
    ymax,xmax = mask.shape
    # set center of ellipse as the center of the image
    if (xc is None) and (yc is None):
        xc,yc = xmax//2,ymax//2
    
    a = sma
    b = BA*sma
    phirad = np.radians(PA)

    X,Y = np.meshgrid(np.arange(xmax),np.arange(ymax))
    
    p1 = ((X-xc)*np.cos(phirad)+(Y-yc)*np.sin(phirad))**2/a**2
    p2 = ((X-xc)*np.sin(phirad)-(Y-yc)*np.cos(phirad))**2/b**2
    flag2 = p1+p2 < 1
    newmask = np.copy(mask)
    newmask[flag2] = 0
    # we could also get all the unique values associated with flag2, and then remove them
    ellipse_params = [xc,yc,sma,BA,phirad]
    return newmask,ellipse_params

def mask_radius_for_mag(mag):
    """ 
    function is from legacy pipeline  
    https://github.com/legacysurvey/legacypipe/blob/6d1a92f8462f4db9360fb1a68ef7d6c252781027/py/legacypipe/reference.py#L314-L319
    """
    # Returns a masking radius in degrees for a star of the given magnitude.
    # Used for Tycho-2 and Gaia stars.

    # This is in degrees, and is from Rongpu in the thread [decam-chatter 12099].
    return 1630./3600. * 1.396**(-mag)


class buildmask():
    def link_files(self):
        # TODO: replace sextractor with photutils
        # these are the sextractor files that we need
        # set up symbolic links from sextractor directory to the current working directory
        sextractor_files=['default.sex.HDI.mask','default.param','default.conv','default.nnw']
        for file in sextractor_files:
            if os.path.exists(file):
                os.remove(file)
            os.system('ln -sf '+self.sepath+'/'+file+' .')
            #os.copy(self.sepath+'/'+file, file)
    def clean_links(self):
        # clean up symbolic links to sextractor files
        # sextractor_files=['default.sex.sdss','default.param','default.conv','default.nnw']
        sextractor_files=['default.sex.HDI.mask','default.param','default.conv','default.nnw']
        for file in sextractor_files:
            os.system('unlink '+file)

    def read_se_cat(self):
        sexout=fits.getdata(self.catname)
        self.xsex=sexout['XWIN_IMAGE']
        self.ysex=sexout['YWIN_IMAGE']
        self.fwhm = sexout['FWHM_IMAGE']
        dist=np.sqrt((self.yc-self.ysex)**2+(self.xc-self.xsex)**2)
        #   find object ID

        # some objects are rturning an empty sequence - how to handle this?
        # I guess this means that the object itself wasn't detected?
        # or nothing was detected?
        if len(dist) < 1:
            # set objnumb to nan
            objnumb = np.nan
        else:
            objIndex=np.where(dist == min(dist))
            objNumber=sexout['NUMBER'][objIndex]
            objnumb = objNumber[0] # not sure why above line returns a list
        return objnumb

    def runse(self,galaxy_id = None,weightim=None,weight_threshold=1):
        # TODO update this to implement running SE with two diff thresholds
        # TODO make an alternate function that creates segmentation image from photutils
        # this is already done in ell
        print('using a deblending threshold = ',self.threshold)
        print("image = ",self.image_name)
        self.catname = self.image_name.replace('.fits','.cat')
        self.segmentation = self.image_name.replace('.fits','-segmentation.fits')

        print("segmentation image = ",self.segmentation)
        sestring = f"sex {self.image_name} -c {self.config} -CATALOG_NAME {self.catname} -CATALOG_TYPE FITS_1.0 -DEBLEND_MINCONT {self.threshold} -DETECT_THRESH {self.snr} -ANALYSIS_THRESH {self.snr_analysis} -CHECKIMAGE_NAME {self.segmentation} -DETECT_MINAREA {self.minarea}"
        if weightim is not None:
            sestring += r" -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {weightim} -WEIGHT_THRESH {weight_threshold}"
        print(sestring)
        os.system(sestring)
        self.maskdat = fits.getdata(self.segmentation)
        # grow masked areas
        bool_array = np.array(self.maskdat.shape,'bool')
        #for i in range(len(self.xsex)):
        # check to see if the object is not centered in the cutout

        
    def get_photutils_mask(self,galaxy_id = None):
        # TODO make an alternate function that creates segmentation image from photutils
        from astropy.stats import sigma_clipped_stats
        from photutils import make_source_mask

        # create mask to cut low SNR pixels based on SNR in SFR image
        mask = make_source_mask(imdat,nsigma=self.snr,npixels=self.minarea,dilate_size=5)
        masked_data = np.ma.array(imdat,mask=mask)
        
        
        self.catname = self.image_name.replace('.fits','.cat')
        self.segmentation = self.image_name.replace('.fits','-segmentation.fits')

        
        self.maskdat = fits.getdata(self.segmentation)
        # grow masked areas
        bool_array = np.array(self.maskdat.shape,'bool')
        #for i in range(len(self.xsex)):
        # check to see if the object is not centered in the cutout
    def remove_center_object(self):
        """ this removes the object in the center of the mask, which presumably is the galaxy """
        # need to replace this with a function that will remove any objects within the specificed central ellipse
        if self.remove_center_object_flag:
            if self.off_center_flag:
                print('setting center object to objid ',self.galaxy_id)
                self.center_object = self.galaxy_id
            else:
                self.center_object = self.read_se_cat()
            if self.center_object is not np.nan:
                self.maskdat[self.maskdat == self.center_object] = 0


        if self.objsma is not None:
            if hasattr(self.objsma, "__len__"):
                # loop over objects in fov
                self.ellipseparams = []                
                for i in range(len(self.objsma)):

                    #print(f"sma={self.objsma_pixels[i]},BA={self.objBA[i]}, PA={self.objPA[i]},xc={self.xpixel[i]},yc={self.ypixel[i]}")
                    self.maskdat,eparams = remove_central_objects(self.maskdat, sma=self.objsma_pixels[i],\
                                                                             BA=self.objBA[i], PA=self.objPA[i], \
                                                                             xc=self.xpixel[i],yc=self.ypixel[i])
                    self.ellipseparams.append(eparams)
                
                pass
            else:
                # remove central objects within elliptical aperture
                print("ellipse params in remove_central_object :",self.xpixel,self.ypixel,self.objsma_pixels,self.objBA,self.objPA)
                self.maskdat,self.ellipseparams = remove_central_objects(self.maskdat, sma=self.objsma_pixels, \
                                                                         BA=self.objBA, PA=self.objPA, \
                                                                         xc=self.xpixel,yc=self.ypixel)
        else:
            
            print("no ellipse params")
            self.ellipseparams = None
        self.update_mask()
        
    def update_mask(self):
        self.add_user_masks()
        self.add_gaia_masks()
        self.write_mask()
    def add_user_masks(self):
        """ this adds back in the objects that the user has masked out """
        # add back the square masked areas that the user added
        self.maskdat = self.maskdat + self.usr_mask
        # remove objects that have already been deleted by user
        if len(self.deleted_objects) > 0:
            for objID in self.deleted_objects:
                self.maskdat[self.maskdat == objID] = 0.
    def write_mask(self):
        """ write out mask image """

        # add ellipse params to imheader
        if self.ellipseparams is not None:
            #print("HEY!!!")
            #print()
            #print("Writing central ellipse parameters to header")
            #print(self.ellipseparams)
            #print()
            if hasattr(self.objsma,"__len__"):
                xc,yc,r,BA,PA = self.ellipseparams[0]
            else:
                xc,yc,r,BA,PA = self.ellipseparams
            self.imheader.set('ELLIP_XC',float(xc),comment='XC of mask ellipse')
            self.imheader.set('ELLIP_YC',float(yc),comment='YC of mask ellipse')
            self.imheader.set('ELLIP_A',r,comment='SMA of mask ellipse')
            self.imheader.set('ELLIP_BA',BA,comment='BA of mask ellipse')
            self.imheader.set('ELLIP_PA',np.degrees(PA),comment='PA (deg) of mask ellipse')
        else:
            print("HEY!!! writing mask, but no parameters for central ellipse!")

            
        fits.writeto(self.mask_image,self.maskdat,header = self.imheader,overwrite=True)
        invmask = self.maskdat > 0.
        invmask = np.array(~invmask,'i')
        fits.writeto(self.mask_inv_image,invmask,header = self.imheader,overwrite=True)
        if not self.auto:
            self.mask_saved.emit(self.mask_image)
            self.display_mask()
    def add_gaia_masks(self):
        # check to see if gaia stars were already masked
        if self.add_gaia_stars:
            if self.gaia_mask is None :
                self.get_gaia_stars()
                self.make_gaia_mask()
            else:
                self.maskdat += self.gaia_mask
    def get_gaia_stars(self, useastroquery=True):
        """ 
        get gaia stars within FOV

        """

        # check to see if gaia table already exists
        outfile = self.image_name.replace('.fits','_gaia_stars.csv')
        if os.path.exists(outfile):
 
            self.brightstar = Table.read(outfile)
            #print(self.brightstar.colnames)
            self.xgaia = self.brightstar['xpixel']
            self.ygaia = self.brightstar['ypixel']

        else:
            brightstar = gaia_stars_in_rectangle(self.racenter,self.deccenter,self.dydeg+.01,self.dxdeg+.01)
            try:
                # get gaia stars within FOV
                # adding buffer to the search dimensions for bright stars that might be just off FOV
                brightstar = gaia_stars_in_rectangle(self.racenter,self.deccenter,self.dydeg+.01,self.dxdeg+.01)

                # Check to see if any stars are returned
                if len(brightstar) > 0:
                    print("found gaia stars in FOV!")
                    # get radius from mag-radius relation
                    mask_radius = mask_radius_for_mag(brightstar['phot_g_mean_mag'])
                    brightstar.add_column(mask_radius,name='radius')

                    starcoord = SkyCoord(brightstar['ra'],brightstar['dec'],frame='icrs',unit='deg')        
                    self.xgaia,self.ygaia = self.image_wcs.world_to_pixel(starcoord)
                    brightstar.add_column(self.xgaia,name='xpixel')
                    brightstar.add_column(self.ygaia,name='ypixel')                

                    self.brightstar = brightstar
                else:
                    self.brightstar = None
                    self.xgaia = None
                    self.ygaia = None

            except:
                print()
                print('WARNING: error using astroquery to get gaia stars')
                print()
                      
                # read in gaia catalog
                try:
                    brightstar = Table.read(self.gaiapath)
                    # Convert ra,dec to x,y        
                    starcoord = SkyCoord(brightstar['ra'],brightstar['dec'],frame='icrs',unit='deg')        
                    x,y = self.image_wcs.world_to_pixel(starcoord)

                    # add buffer to catch bright stars off FOV
                    buffer = 0.1*self.xmax
                    flag = (x > -buffer) & (x < self.xmax+buffer) & (y>-buffer) & (y < self.ymax+buffer)

                    # add criteria for proper motion cut
                    # Hopefully this fix should resolve cases where center of galaxy is masked out as a star...
                    # changing to make this a SNR > 5 detection, rather than 5 mas min proper motion
                    pmflag = np.sqrt(brightstar['pmra']**2*brightstar['pmra_ivar'] + brightstar['pmdec']**2*brightstar['pmdec_ivar']) > 5

                    flag = flag & pmflag
                    if np.sum(flag) > 0:
                        self.brightstar = brightstar[flag]
                        self.xgaia = x
                        self.ygaia = y
                        brightstar.add_column(self.xgaia,name='xpixel')
                        brightstar.add_column(self.ygaia,name='ypixel')                

                    else:
                        self.brightstar = None
                        self.xgaia = None
                        self.ygaia = None


                except FileNotFoundError:
                    warnings.warn(f"Can't find the catalog for gaia stars({self.gaiapath}) - running without bright star masks!")
                    self.add_gaia_stars = False
                    return

            # Write out resulting file for future use
            if self.brightstar is not None:
                outfile = self.image_name.replace('.fits','_gaia_stars.csv')
                self.brightstar.write(outfile,format='csv')
            


    def make_gaia_mask(self):
        """
        mask out bright gaia stars using the legacy dr9 catalog and magnitude-radius relation:  
        https://github.com/legacysurvey/legacypipe/blob/6d1a92f8462f4db9360fb1a68ef7d6c252781027/py/legacypipe/reference.py#L314-L319
        """

        self.get_gaia_stars()

        # set up blank
        self.gaia_mask = np.zeros_like(self.maskdat)
        
        if self.brightstar is not None:
            # add stars to mask according to the magnitude-radius relation
            mag = self.brightstar['phot_g_mean_mag']
            xstar = self.xgaia
            ystar = self.ygaia
            rad = self.brightstar['radius'] # in degrees
            
            # Convert radius to pixels            
            radpixels = rad/self.pscalex.value


            # use the same value for all gaia stars. set this above max value in mask
            mask_value = np.max(self.maskdat) + 100 
            print('mask value = ',mask_value)
            for i in range(len(mag)):
                # mask stars
                print(f"star {i}: {xstar[i]:.1f},{ystar[i]:.1f},{radpixels[i]:.1f}")
                pixel_mask = circle_pixels(float(xstar[i]),float(ystar[i]),float(radpixels[i]),self.xmax,self.ymax)
                #print(f"number of pixels masked for star {i} = {np.sum(pixel_mask)}")
                #print('xcursor, ycursor = ',self.xcursor, self.ycursor)
                #print("\nshape of pixel_mask = ",pixel_mask.shape)
                #print("\nshape of gaia_mask = ",self.gaia_mask.shape)                
                self.gaia_mask[pixel_mask] = mask_value*np.ones_like(self.gaia_mask)[pixel_mask]

            # add gaia stars to main mask                
            self.maskdat = self.maskdat + self.gaia_mask
        else:
            print("No bright stars on image - woo hoo!")

    def run_photutil(self, snrcut=1.5,npixels=10):
        ''' 
        run photutils detect_sources to find objects in fov.  
        you can specify the snrcut, and only pixels above this value will be counted.
        
        this also measures the sky noise as the mean of the threshold image
        '''
        self.threshold = detect_threshold(self.image, nsigma=snrcut)
        segment_map = detect_sources(self.image, self.threshold, npixels=npixels)
        # deblind sources a la source extractor
        # tried this, and the deblending is REALLY slow
        # going back to source extractor
        self.segmentation = deblend_sources(self.image, segment_map,
                               npixels=10, nlevels=32, contrast=0.001)        
        self.maskdat = self.segmentation.data
        #self.cat = source_properties(self.image, self.segmentation)
        self.cat = SourceCatalog(self.image, self.segmentation)        
        # get average sky noise per pixel
        # threshold is the sky noise at the snrcut level, so need to divide by this
        self.sky_noise = np.mean(self.threshold)/snrcut
        #self.tbl = self.cat.to_table()

        if self.off_center_flag:
            print('setting center object to objid ',self.galaxy_id)
            self.center_object = self.galaxy_id
        else:
            distance = np.sqrt((self.cat.xcentroid - self.xc)**2 + (self.cat.ycentroid - self.yc)**2)
            # save object ID as the row in table with source that is closest to center
            objIndex = np.arange(len(distance))[(distance == min(distance))][0]
            # the value in shown in the segmentation image is called 'label'
            self.center_object = self.cat.label[objIndex]

        self.maskdat[self.maskdat == self.center_object] = 0
        self.update_mask()

    
    def grow_mask(self, size=7):

        """
        Convolution: one way to grow the mask is to convolve the image with a kernel

        however, this does not preserve the pixels value of the original
        object, which come from the sextractor segmentation image.

        if the user wants to remove an object, it's much easier to do this
        by segmentation number rather than by pixels (as in the reverse of how we add objects
        to mask).

        Alternative: is to loop over just the masked pixels, and replace all pixels
        within a square region with the masked value at the central pixel.
        This will preserve the numbering from the segmentation image.

        Disadvantage: everything assumes a square shape after repeated calls.

        Alternative is currently implemented.
        
        """
        # convolve mask with top hat kernel
        # kernel = Tophat2DKernel(5)
        #mykernel = np.ones([5,5])
        #kernel = CustomKernel(mykernel)
        #self.maskdat = convolve(self.maskdat, kernel)
        #self.maskdat = np.ceil(self.maskdat)

        # we don't want to grow the size of the gaia stars, do we???
        self.maskdat -= self.gaia_mask
        nx,ny = self.maskdat.shape
        masked_pixels = np.where(self.maskdat > 0.)
        for i,j in zip(masked_pixels[0], masked_pixels[1]):
            rowmin = max(0,i-int(size/2))
            rowmax = min(nx,i+int(size/2))
            colmin = max(0,j-int(size/2))
            colmax = min(ny,j+int(size/2))
            if rowmax <= rowmin:
                # something is wrong, return without editing mask
                continue
            if colmax <= colmin:
                # something is wrong, return without editing mask
                continue
            #print(i,j,rowmin, rowmax, colmin, colmax)
            self.maskdat[rowmin:rowmax,colmin:colmax] = self.maskdat[i,j]*np.ones([rowmax-rowmin,colmax-colmin])
        # add back in the gaia star masks
        self.maskdat += self.gaia_mask
        if not self.auto:
            self.display_mask()
        # save convolved mask as new mask
        self.write_mask()

    def show_mask_mpl(self):
        # plot mpl figure
        # this was for debugging purposes
        print("plotting mask and central ellipse")
        self.fig = plt.figure(1,figsize=self.figure_size)
        plt.clf()
        plt.subplots_adjust(hspace=0,wspace=0)
        plt.subplot(1,2,1)
        plt.imshow(self.image,cmap='gray_r',vmin=self.v1,vmax=self.v2,origin='lower')
        plt.title('image')
        plt.subplot(1,2,2)
        #plt.imshow(maskdat,cmap='gray_r',origin='lower')
        plt.imshow(self.maskdat,cmap=self.cmap,origin='lower',vmin=np.min(self.maskdat),vmax=np.max(self.maskdat))
        plt.title('mask')
        plt.gca().set_yticks(())
        #plt.draw()
        #plt.show(block=False)
        #print("in show_mask_mpl: objsma = ",self.objsma)        
        try:
            
            if hasattr(self.objsma, "__len__"):
                #print("working with multiple galaxies")
                # add ellipse for each galaxy if there is more than one
                for e in self.ellipseparams:
                    xc,yc,r,BA,PA = e
                    PAdeg = np.degrees(PA)
                    #print(f"BA={BA},PA={PAdeg} deg")        
                    #print("just checking - adding ellipse drawing ",self.ellipseparams)
                    ellip = patches.Ellipse((xc,yc),2*r,2*r*BA,angle=PAdeg,alpha=.2)
                    plt.gca().add_patch(ellip)
            else:
                xc,yc,r,BA,PA = self.ellipseparams
                PAdeg = np.degrees(PA)
                #print(f"BA={BA},PA={PAdeg} deg")        
                #print("just checking - adding ellipse drawing ",self.ellipseparams)
                ellip = patches.Ellipse((xc,yc),r,r*BA,angle=PAdeg,alpha=.2)
                plt.gca().add_patch(ellip)

        except:
            print("problem plotting ellipse with mask")
        # outfile
        outfile = self.mask_image.replace('.fits','.png')
        plt.savefig(outfile)
        
        #plt.show()
        

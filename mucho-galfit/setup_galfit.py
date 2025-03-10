#!/usr/bin/env python

'''
GOAL:
* set up directories for running galfit on WISESize sample

PROCEDURE:
* read .fits tables with PRIMARY flags

* for galaxies with PRIMARY flag, create a directory

USAGE:
* on draco, move to /mnt/astrophysics/mg-output-wisesize

* call as:
    python /mnt/astrophysics/kconger_wisesize/github/wisesize/mucho-galfit/setup_galfit.py

    * will create directories of primary group galaxies. each directory will contain the image, noise, and psf.

'''

import os
import sys

##########################################################################     
### FUNCTIONS
##########################################################################     

#functions to change .fits.fz to .fits
def funpack_image_cfitsio(input,output):
    command = 'funpack -O {} {}'.format(output,input)
    print(command)
    os.system(command)
def funpack_image(input,output,nhdu=1):
    from astropy.io import fits
    hdu = fits.open(input)
    print('input file = ',input)
    fits.writeto(output,data=hdu[nhdu].data, header=hdu[nhdu].header, overwrite=True)
    hdu.close()
    #print('finished unpacking image')

#unpack composite images into their constituent wavelength bands
#save results in the same spot as where JM images are stored (path_to_im=data_root_dir)
def extract_bands(path_to_im,im_name=None,grz=False,WISE=False):
    if grz:
        ims,header=fits.getdata(path_to_im,im_name,header=True)
        g_im, r_im, z_im = ims[0], ims[1], ims[2]
        
        g_im_name = im_name_grz.replace('.fits','-im-g.fits')
        r_im_name = im_name_grz.replace('.fits','-im-r.fits')
        z_im_name = im_name_grz.replace('.fits','-im-z.fits')
        
        fits.writeto(path_to_im+g_im_name,ims[0],header=header,overwrite=True)
        fits.writeto(path_to_im+r_im_name,ims[1],header=header,overwrite=True)
        fits.writeto(path_to_im+z_im_name,ims[2],header=header,overwrite=True)

    if WISE:
        ims,header=fits.getdata(path_to_im,im_name,header=True)
        w1_im, w2_im, w3_im, w4_im = ims[0], ims[1], ims[2], ims[3]
        
        w1_im_name = im_name.replace('.fits','-im-W1.fits')
        w2_im_name = im_name.replace('.fits','-im-W2.fits')
        w3_im_name = im_name.replace('.fits','-im-W3.fits')
        w4_im_name = im_name.replace('.fits','-im-W4.fits')
        
        fits.writeto(path_to_im+w1_im_name,ims[0],header=header,overwrite=True)
        fits.writeto(path_to_im+w2_im_name,ims[1],header=header,overwrite=True)
        fits.writeto(path_to_im+w3_im_name,ims[2],header=header,overwrite=True)
        fits.writeto(path_to_im+w4_im_name,ims[3],header=header,overwrite=True)
    
#convert invvar image to noise
def convert_invvar_noise(invvar_image, noise_image):
    from astropy.io import fits
    import numpy as np
    # read in invvar image
    print('invvar image = ',invvar_image, os.path.basename(invvar_image))
    hdu = fits.open(invvar_image)
    data = hdu[0].data
    header = hdu[0].header
    hdu.close()
    # operate on pixels to take sqrt(1/x)
    noise_data = np.sqrt(1/data)
    
    # check for bad values, nan, inf
    # set bad pixels to very high value, like 1e6
    noise_data = np.nan_to_num(noise_data,nan=1.e6)

    # write out as noise image
    fits.writeto(noise_image,noise_data,header=header,overwrite=True)
    
#path_to_repos e.g., /mnt/astrophysics/kconger_wisesize/
def get_images(objid,ra,dec,objname,output_loc,data_root_dir):
    ###############################################################################
    ### GET IMAGES
    ###############################################################################

    #output_loc is the directory holding the individual galaxy output directories (which GALFIT will be pulling from!)
    #e.g., /mnt/astrophysics/kconger_wisesize/mg_output_wisesize/OBJ1000/
    output_dir = os.path.join(output_loc,objid+'/')
    if not os.path.exists(output_dir):
        print("making the output directory ",output_dir)
        os.mkdir(output_dir)

    #data_root_dir is where JM's input images are initially stored
    if not os.path.exists(data_root_dir):
        print(f"could not find data_root_dir - exiting")
        sys.exit()

    ra_val = str(int(ra)) if len(str(int(ra)))==3 else '0'+str(int(ra))
    dec_val = str(int(dec)) if len(str(int(dec)))==2 else '0'+str(int(dec))
    
    if dec>32.:   #if DEC>32 degrees, then galaxy is in "north" catalog
        data_dir = f'{data_root_dir}/dr9-north/native/{ra_val}/'
    if dec<32.:
        data_dir = f'{data_root_dir}/dr9-south/native/{ra_val}/'
    
    
    if not os.path.exists(data_dir):
        print(f"could not find data_dir - exiting")
        sys.exit()

    print("source directory for JM images = ",data_dir)
    
    ra_string = ra_val + str(np.modf(ra)[0])[1:6]    
    dec_string = dec_val + str(np.modf(dec)[0])[1:6]
    im_name = f'SGA2025_J{ra_string}+{dec_string}.fits'

        
        
    extract_bands(data_dir,im_name=im_name,grz=False,WISE=False)
    
    for bandpass in ['r','g','z','W1','W2','W3','W4']:
        image = f'{im_name}*{bandpass}*'
        #invvar_image = f'{group_name}-custom-invvar-{bandpass}.fits.fz'    
        #psf_image = f'{group_name}-custom-psf-{bandpass}.fits.fz'

        os.system(f'cp {image} {output_dir}')
        
        # created images
        #sigma_image = f'{group_name}-custom-std-{bandpass}.fits'

        # check if noise image exists, if not make it from invvar    
        #if not os.path.exists(sigma_image):
        #    convert_invvar_noise(os.path.join(output_dir,invvar_image),os.path.join(output_dir,sigma_image))
        

    ###############################################################################
    ### END GET IMAGES
    ###############################################################################


##########################################################################     
### END FUNCTIONS
##########################################################################     

if __name__ == '__main__':

    from astropy.table import Table

    # define environment variable so funpack can find the correct variables
    #os.environ["LD_LIBRARY_PATH"]="/opt/ohpc/pub/compiler/gcc/9.4.0/lib64:/home/siena.edu/rfinn/software/cfitsio-4.2.0/"
    
    param_file = '/mnt/astrophysics/kconger_wisesize/github/wisesize/mucho-galfit/paramfile.txt'
        
    #create dictionary with keyword and values from param textfile
    param_dict={}
    with open(param_file) as f:
        for line in f:
            try:
                key = line.split()[0]
                val = line.split()[1]
                param_dict[key] = val
            except:
                continue
    
    main_dir = param_dict['main_dir']
    path_to_pyscripts = main_dir+param_dict['path_to_scripts']
    outdir = main_dir+param_dict['path_to_images']
    data_root_dir = param_dict['data_root_dir']
    
    main_catalog = param_dict['main_catalog']
    phot_catalog = param_dict['phot_catalog']
    
    objid_col = param_dict['objid_col']
    primary_group_col = param_dict['primary_group_col']
    group_mult_col = param_dict['group_mult_col']
    group_name_col = param_dict['group_name_col']
    objname_col = param_dict['objname_col']

    try:
        etab = Table.read(phot_catalog)
        maintab = Table.read(main_catalog)        
    
    except FileNotFoundError:
        print("ERROR: problem locating catalogs - exiting")
        sys.exit()

    os.chdir(outdir)
    
    # for each galaxy, create a directory and write sourcelist
    for i in range(len(maintab)):
        if etab[primary_group_col][i] & (etab[group_mult_col][i] > 0): # make directory for primary targets
            galpath = outdir+etab[objid_col][i]

            # make directory if it doesn't already exist
            if not os.path.exists(galpath):
                os.mkdir(galpath)
            os.chdir(galpath)
            
            #sourcefile = galpath+'/{}sourcelist'.format(maintab['VFID'][i])
            #sourcelist = open(sourcefile,'w')
            ## write out one line with VFID, objname, RA, DEC, wavelength
            #output_string = maintab['VFID'][i] + ' ' + maintab['objname'][i] + ' ' + str(maintab['RA'][i]) + ' ' + str(maintab['DEC'][i]) + ' ' + str(wavelength) + ' \n'.format()
            #sourcelist.write(output_string)
            #sourcelist.close()

            # copy images
            obj_id = maintab[objid_col][i]
            ra = maintab['RA'][i]
            dec = maintab['DEC'][i]
            objname = maintab[objname_col][i]
            group_name = etab[group_name_col][i] # this is either the objname, or objname_GROUP for groups

            get_images(obj_id,ra,objname,outdir,data_root_dir)

            os.chdir(outdir)

            # for testing
            #if i > 1:
            #    sys.exit()
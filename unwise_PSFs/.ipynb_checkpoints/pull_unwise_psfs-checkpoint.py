'''
AIM: cp unWISE PSFs to github directories for every galaxy in the WISESize sample.

*load WISESize table
*match galaxy RA+DEC with tile centers, add table column with tile COADD IDs
*use COADD ID to pull unWISE PSF
*write PSFs to relevant directories

'''

import os

mnt_path = '/mnt/astrophysics/kconger_wisesize/'
code_path = mnt_path+'github/'
tile_path = code_path+'/wisesize/unwise_PSFs/tiles.fits'  #contains COADD IDs and the RA+DEC of tile centers

os.chdir(code_path+'/unwise_psf/py')
import unwise_psf.unwise_psf as unwise_psf


def read_table(path_to_table):
    return Table.read(path_to_table)

def read_tiles(path_to_tiles):
    return Table.read(path_to_tiles)

def get_coadd_ids(table,tiles):
    
    #isolate RA and DEC columns
    wise_RA = table['RA']
    wise_DEC = table['DEC']

    tile_RA = tiles['ra']
    tile_DEC = tiles['dec']
    
    #create empty index array
    idx_arr=np.zeros(len(RA),dtype=int)

    #for every catalog RA and DEC, find tilefile index where the central RA&DEC most closely matches to catalog RA&DEC
    for n in range(len(wise_RA)):
        idx = (np.abs(tile_RA - wise_RA[n]) + np.abs(tile_RA - wise_DEC[n])).argmin()
        idx_arr[n]=idx
    
    #set new column (coadd_id) which contains the appropriate row-matched tile name for each galaxy
    table['coadd_id'] = tiles[idx_arr]['coadd_id']
    
    return table

def pull_unwise_psf(ra, dec, coadd_id, band, path_to_image_dir):
    
    ra_val = str(int(ra)) if len(str(int(ra)))==3 else '0'+str(int(ra))
    dec_val = str(int(dec)) if len(str(int(dec)))==2 else '0'+str(int(dec))
    
    #the [1, removes the 0 preceding the decimal, and the :6] goes to the fourth decimal place
    ra_string = ra_val + str(np.modf(ra)[0])[1:6]    
    dec_string = dec_val + str(np.modf(dec)[0])[1:6]
    
    #generate filename following the string convention
    im_name = f'SGA2025_J{ra_string}+{dec_string}-PSF-W{band}.fits'

    #pull unwise psf
    band_psf = unwise_psf.get_unwise_psf(band,coadd_id)
    
    save_unwise_psf(im_name, band_psf, path_to_image_dir)
    
    
def save_unwise_psf(im_name, band_psf, path_to_image_dir):
    band_psf.write(path_to_image_dir+im_name, overwrite=True)
    
    
    
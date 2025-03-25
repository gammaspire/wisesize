'''
AIM: generate W1-4 unWISE PSFs for image with given RA and DEC

PROCEDURE: 
Single Galaxy:
* read input tile table; contains RA, DEC and coadd ids of tiles
* for given input galaxy (or rather, its image), pull RA and DEC then find closest tile center. match galaxy with tile's coadd id
* for W1-4, use coadd id to generate correct unWISE PSF
* move PSFs to correct objid directory
Multiple Galaxies:
* read input table and tile table, use primary group flag (if it exists) to "trim" maintab; only want primary galaxy PSFs
* create row-matched coadd id column, append to input table
* for every PRIMARY galaxy in this table, generate unWISE PSFs in every band and move to correct directory
'''

import os
from astropy.table import Table

mnt_path = '/mnt/astrophysics/kconger_wisesize/'
code_path = mnt_path+'github/'

os.chdir(code_path+'/unwise_psf/py')
import unwise_psf.unwise_psf as unwise_psf


#read main table
def read_table(path_to_table):
    return Table.read(path_to_table)

#create coadd_id table
def get_coadd_table(table,tiles):
    
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


def read_tiles(path_to_tiles):
    return Table.read(path_to_tiles)


def get_header_radec(path_to_image):
    
    from astropy.io import fits
    
    header=fits.getheader(path_to_image)
    RA = header['CRVAL1']
    DEC = header['CRVAL2']
    return RA, DEC


#pull single coadd_id for a single RA-DEC set
def get_coadd_id(path_to_tiles, path_to_image_dir, tile_table=None):

    if tile_table == None:
        tile_table = read_tiles(path_to_tiles)
    
    #pull name of ANY im; its header will contain RA and DEC
    #will also use to define PSF name
    #includes full image path AND image filename
    path_to_image = glob.glob(path_to_image_dir+'*im*.fits')[0]
    
    tile_RA = tile_table['ra']
    tile_DEC = tile_table['dec']
    
    galaxy_RA, galaxy_DEC = get_header_radec(path_to_image)
    
    #for galaxy RA and DEC, find tilefile index where the central RA&DEC most closely matches galaxy RA&DEC
    coadd_idx = (np.abs(tile_RA - galaxy_RA) + np.abs(tile_RA - galaxy_DEC)).argmin()
    
    coadd_id = tile_table['coadd_id'][coadd_idx]
    return coadd_id


#need imname to determine correct formatting of PSF filename  
#I suspect providing full pathname AND image name as part of imname is acceptable.
def pull_unwise_psf(path_to_image_dir, coadd_id, band):

    #pull name of ANY im; use to define PSF name
    #includes full image path AND image filename
    imname = glob.glob(path_to_image_dir+'*im*.fits')[0]
    
    #find the index where 'im' starts
    index = imname.find('im')
    #replace that section of the string with the desired indicators for PSF file
    psfname = imname.replace(imname[index:],f'PSF-W{band}.fits')
    
    #pull unwise psf
    band_psf = unwise_psf.get_unwise_psf(band,coadd_id)
    
    save_unwise_psf(psfname, band_psf)


def save_unwise_psf(im_name, band_psf):
    band_psf.write(im_name, overwrite=True)
    
if __name__ == '__main__':

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
    outdir = main_dir+param_dict['path_to_images']
    tile_path = main_dir+param_dict['tile_path']   #contains COADD IDs and the RA+DEC of tile centers
    
    
    if '-objid' in sys.argv:
        p = sys.argv.index('-objid')
        objid = str(sys.argv[p+1])
        
        path_to_image_dir = outdir+objid+'/'  #where to save each PSF image...also where to find reference image for formatting
        
        #same for every wavelength band
        coadd_id = get_coadd_id(tile_path, path_to_image_dir)
    
        #for W1-W4, generate and save unwise PSF for the given galaxy
        for band in range(1,5):
            pull_unwise_psf(path_to_image_dir, coadd_id, band)
        
        
    else:
        print('no objid arg found. running for all galaxies. add -objid [objid] in terminal for only one galaxy.')
        response = input('continue? y for "yes" and n for "no"')
        if response=='n':
            sys.exit()
        elif response=='y':
            objid_col = param_dict['objid_col']   #go ahead and define objid_col variable

            #try to read in main catalog; if not found, code will terminate
            try:
                maintab = read_table(param_dict['main_catalog'])
                
                primary_group_col = param_dict['primary_group_col']
                
                #trim to only include primary galaxies; if no such flag exists, assume all galaxies are primary.
                try:
                    primary_flag = maintab[primary_group_col]
                except:
                    primary_flag = np.ones(len(maintab),dtype=bool)   #all true
                
                maintab = maintab[primary_flag]
                
            except:
                print('cannot find main table. exiting.')
                sys.exit()
            
            #read in tile table, then generate coadd table
            tiletab = read_tiles(tile_path)
            coadd_table = get_coadd_table(maintab,tiletab)
            
            #for every primary objid, generate W1-4 unWISE PSFs and move to correct primary directory
            for i, objid in enumerate(maintab[objid_col]):

                path_to_image_dir = outdir+objid+'/'
                coadd_id = coadd_table['coadd_id'][i]

                for band in range(1,5):
                    pull_unwise_psf(path_to_image_dir, coadd_id, band)
        else:
            print('did not recognize response. terminating.')
            sys.exit()
    

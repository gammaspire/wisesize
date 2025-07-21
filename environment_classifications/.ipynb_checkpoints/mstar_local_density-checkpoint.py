import numpy as np
from astropy.table import Table
from matplotlib import figure 
from scipy.spatial import KDTree
import time
import sys

import os
homedir=os.getenv("HOME")

###CONVERSION FUNCTIONS###
sys.path.append(homedir+'/github/wisesize/')
from universal_functions import *


#FOR RA-DEC!
def plot_Sigma_Mstar(all_RA, all_DEC, all_Sigma_Mstar):

    #define "good" flag, which filters out all instances where galaxies did not have a Sigma_Mstar
    #given our isolation constraints. this, however, should not ever be the case.
    good_flag = (all_Sigma_Mstar!=-999)
    
    fig = figure.Figure(figsize=(14,6))
    ax = fig.add_subplot()
        
    im = ax.scatter(all_RA[good_flag], all_DEC[good_flag], c=np.log10(all_Sigma_Mstar[good_flag]), cmap='viridis', alpha=0.5, s=5, vmin=-1, vmax=1.5)
    ax.invert_xaxis()
    ax.set_xlabel('RA [deg]',fontsize=14)
    ax.set_ylabel('DEC [deg]',fontsize=14)
    
    ax.tick_params(labelsize=14)
    
    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(fr'$\log(\Sigma_{{M_*}}$ / [M$_{{*}}$ Mpc$^{{-2}}$])', fontsize=14)
    
    fig.savefig(homedir+f'/Desktop/Sigma_Mstar_plot.png', dpi=100, bbox_inches='tight', pad_inches=0.2)


def save_to_table(cat, all_Sigma_Mstar, version=1):
    
    #apply flags if they exist; else, 
    try:
        mstarflag = cat['Mstar_all_flag']
    except:
        if len(cat) != len(all_Sigma_Mstar):
            print('Input table length not same as Sigma_Mstar array length. Check that flag column names match the function!')
            print('Exiting.')
            sys.exit()
        mstarflag = np.ones(len(cat),dtype=bool)
    
    #also add WISESize flags
    raflag = (cat['RA']>87) & (cat['RA']<300)
    decflag = (cat['DEC']>-10) & (cat['DEC']<85)
    zflag = (cat['Z']>0.002) & (cat['Z']<0.025)

    #these are ALL flags applied to the 5NN input table
    flags = (mstarflag) & (zflag) & (raflag) & (decflag)

    all_Sigma_Mstar_parent = np.full(len(cat), -999)
    all_Sigma_Mstar_parent[flags] = all_Sigma_Mstar
    
    cat[f'2D_Sigma_Mstar'] = all_Sigma_Mstar_parent
    
    save_path = homedir+f'/Desktop/wisesize/nedlvs_parent_v{version}.fits'
    cat.write(save_path,overwrite=True)
    
    print(f'2D_Sigma_Mstar column added to (or updated in) {save_path}')
    
    
class central_galaxy():
    def __init__(self, ra, dec, redshift, cat):
        
        self.ra = ra
        self.dec = dec
        self.redshift = redshift
        
        self.cat = cat
        
    def isolate_galaxy_region(self, vr_limit, radius_limit):
        
        #use redshifts to calculate width of "slice"
        z_lower, z_upper = get_redshift_bounds(self.redshift, vr_limit)

        #from that list, cut galaxies which are beyond the redshift slice!
        z_flag = (self.cat['Z']>z_lower) & (self.cat['Z']<z_upper)

        #use radius limit to further isolate the search region
        radius_flag = get_radius_flag(self.redshift, self.ra, self.dec, self.cat['RA'], self.cat['DEC'], radius_limit)

        cat = self.cat[radius_flag&z_flag]
        
        #this catalog contains all enclosed galaxies (including the central galaxy)
        self.trimmed_cat = cat

    
    #sums stellar masses of all enclosed galaxies within the region defined with isolate_galaxy_region()
    def sum_enclosed_mstar(self):
        masses = self.trimmed_cat['Mstar']
        sum_masses = np.sum(masses)   #in units of Msol
        return sum_masses
    
    
    #divide sum by enclosed circle area
    def calc_Sigma_Mstar(self, radius_limit):
        sum_masses = self.sum_enclosed_mstar()
        self.density_Mstar = sum_masses / (np.pi * radius_limit**2)

    
if __name__ == "__main__":
    
    cat_full = Table.read(homedir+'/Desktop/wisesize/nedlvs_parent_v1.fits')
    Mstar_full = Table.read(homedir+'/Desktop/wisesize/archive_tables/NEDLVS_20210922_v2.fits')['Mstar']
    
    if '-h' or '-help' in sys.argv:
        print('-vr_limit [int in km/s; default is 500] -radius_limit [number in Mpc OR "r200" to use the r200 of the group when available; default is 1] -write [will write array output to nedlvs_parent table]')
    
    if '-vr_limit' in sys.argv:
        p = sys.argv.index('-vr_limit')
        vr_limit = int(sys.argv[p+1])
        print(f'Using vr_limit = {vr_limit} km/s')
    else:
        vr_limit = 500  #km/s
        print('Using vr_limit = 500 km/s')
    
    if '-radius_limit' in sys.argv:
        p = sys.argv.index('-radius_limit')
        if (sys.argv[p+1] != 'r200'):
            radius_limit = float(sys.argv[p+1])
            print(f'Using radius_limit = {radius_limit} Mpc')
            
            #create row-matched array for every galaxy...ensures no errors now that array parameter is an option
            radius_limit = np.zeros(len(cat_full))+radius_limit
            
        else:
            try:
                radius_limit = cat_full['group_R200']
                print(f'Using variable radius limit')
            except:
                print('R200 column not found in nedlvs_parent catalog! Exiting.')
                sys.exit()
    else:
        radius_limit = 1.  #Mpc
        print('Using radius_limit = 1 Mpc')
        
        #create row-matched array for every galaxy...ensures no errors now that array parameter is an option
        radius_limit = np.zeros(len(cat_full))+radius_limit

    print('Applying mass completeness limit flag to catalog...')
    try:
        mstarflag = cat_full['Mstar_all_flag']
    except:
        print('No mass completeness limit flag found! Ignoring.')
        mstarflag = np.ones(len(cat_full),dtype=bool)

    print('Removing objects beyond the WISESize redshift and RA-DEC range...')
    zflag = (cat_full['Z']>0.002) & (cat_full['Z']<0.025)
    raflag = (cat_full['RA']>87) & (cat_full['RA']<300)
    decflag = (cat_full['DEC']>-10) & (cat_full['DEC']<85)

    #applying all flags at once
    cat = cat_full[(mstarflag) & (zflag) & (raflag) & (decflag)]
    Mstar = Mstar_full[(mstarflag) & (zflag) & (raflag) & (decflag)]
        
    radius_limit = radius_limit[(mstarflag) & (zflag) & (raflag) & (decflag)]
    radius_limit[radius_limit==-99.] = 0.3 #Mpc, corresponding to 300 kpc. default for galaxies not in Tempel group!
                                            
    #append Mstar column to trimmed catalog
    cat['Mstar'] = Mstar

    print(f'Number of starting galaxies: {len(cat)}')
        
    all_Sigma_Mstar = np.zeros(len(cat))
    ra = cat['RA']
    dec = cat['DEC']
    redshift = cat['Z']

    start_time = time.perf_counter()
    
    for n in range(len(cat)):
        #define galaxy class object
        galaxy = central_galaxy(ra[n], dec[n], redshift[n], cat)
        
        #set up trimmed catalog (which restricts galaxies to be within redshift (and possibly radius) slice
        galaxy.isolate_galaxy_region(vr_limit, radius_limit[n])

        #self-explanatory -- calculates sum of stellar masses enclosed in circle about the central galaxy
        galaxy.sum_enclosed_mstar()

        galaxy.calc_Sigma_Mstar(radius_limit[n])
        all_Sigma_Mstar[n] = galaxy.density_Mstar

    
    print('# galaxies in input table without :',len(all_Sigma_Mstar[all_Sigma_Mstar==-999]))
    
    if '-write' in sys.argv:
        save_to_table(cat_full, all_Sigma_Mstar, version=1)
    
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    print(f'Execution Time: {(execution_time/60.):.2} minute(s)')
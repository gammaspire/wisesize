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


def save_to_table(cat, all_Sigma_Mstar, all_ngal, version=1):
    
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
    
    all_ngal_parent = np.full(len(cat), -999)
    all_ngal_parent[flags] = all_ngal
    
    cat[f'2D_Sigma_Mstar'] = all_Sigma_Mstar_parent
    cat[f'2D_Sigma_ngal'] = all_Sigma_Mstar_parent
    
    save_path = homedir+f'/Desktop/wisesize/nedlvs_parent_v{version}.fits'
    cat.write(save_path,overwrite=True)
    
    print(f'2D_Sigma_Mstar and 2D_Sigma_ngal columns added to (or updated in) {save_path}')
    
    
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
    
    #sums number of galaxies within the enclosed region
    def sum_enclosed_ngal(self):
        self.ngal = len(self.trimmed_cat)
    
    #divide sum by enclosed circle area
    def calc_Sigma_Mstar(self, radius_limit):
        sum_masses = self.sum_enclosed_mstar()
        self.density_Mstar = sum_masses / (np.pi * radius_limit**2)


##########################################   
#NEED for when importing code as a module!
##########################################
def Sigma_Mstar_Ngal(vr_limit=1000, radius_limit=1.0):
    """
    Compute local stellar mass density (Sigma_Mstar) and number of galaxies (ngal)
    around each galaxy in the catalog.

    Parameters:
        vr_limit (int): Velocity dispersion limit in km/s.
        radius_limit_value (float or str): Fixed radius in Mpc (e.g., 1.0) or "r200" for group radius.

    Returns:
        all_Sigma_Mstar (np.ndarray): Array of stellar mass surface densities.
        all_ngal (np.ndarray): Array of galaxy counts within projected radius.
    """
    
    import time
    from astropy.table import Table
    from scipy.spatial import KDTree
    import numpy as np
    import os
    
    homedir = os.getenv("HOME")
      
    cat_full = Table.read(homedir+'/Desktop/wisesize/nedlvs_parent_v1.fits')
    Mstar_full = Table.read(homedir+'/Desktop/wisesize/archive_tables/NEDLVS_20210922_v2.fits')['Mstar']

    #apply isolation flags
    try:
        mstarflag = cat_full['Mstar_all_flag']
    except:
        print('Mass completeness flag not found.')
        mstarflag = np.ones(len(cat_full), dtype=bool)

    zflag = (cat_full['Z'] > 0.002) & (cat_full['Z'] < 0.025)
    raflag = (cat_full['RA'] > 87.) & (cat_full['RA'] < 300.)
    decflag = (cat_full['DEC'] > -10.) & (cat_full['DEC'] < 85.)
    
    print(f'Calculating Sigma_M and Ngal for vr_limit = {vr_limit} and radius_limit = {radius_limit}')
    cat = cat_full[mstarflag & zflag & raflag & decflag]
    Mstar = Mstar_full[mstarflag & zflag & raflag & decflag]

    if radius_limit == 'r200':
        try:
            radius_limit = cat['group_R200']
        except:
            raise ValueError("R200 column not found in catalog.")
    else:
        radius_limit = np.full(len(cat), float(radius_limit))

    radius_limit[radius_limit == -99.] = 0.3  # fallback default

    cat['Mstar'] = Mstar

    all_Sigma_Mstar = np.zeros(len(cat))
    all_ngal = np.zeros(len(cat))

    ra = cat['RA']
    dec = cat['DEC']
    redshift = cat['Z']

    start_time = time.perf_counter()
    
    for n in range(len(cat)):
        galaxy = central_galaxy(ra[n], dec[n], redshift[n], cat)
        galaxy.isolate_galaxy_region(vr_limit, radius_limit[n])
        galaxy.sum_enclosed_mstar()
        galaxy.calc_Sigma_Mstar(radius_limit[n])
        all_Sigma_Mstar[n] = galaxy.density_Mstar
        galaxy.sum_enclosed_ngal()
        all_ngal[n] = galaxy.ngal
        
        print(f"Number of galaxies with no Sigma_M: {(all_Sigma_Mstar == -999).sum()}")
        
    print(f"Finished in {(time.perf_counter() - start_time)/60:.2f} minutes")

    return all_Sigma_Mstar, all_ngal        
        
        
#######################################################################################################
#this part ONLY RUNS if the code is run from a command line as opposed to if it is imported as a module
#######################################################################################################
if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="Compute local stellar mass density (Sigma_Mstar) and galaxy counts")
    parser.add_argument("-vr_limit", type=int, default=500, help="Velocity dispersion limit in km/s (default 500)")
    parser.add_argument("-radius_limit", type=str, default="1", 
                        help='Radius limit in Mpc (e.g., 1.0) or "r200" for group radius (default "1")')
    parser.add_argument("-write", action="store_true", help="Write output arrays to nedlvs_parent FITS table")

    args = parser.parse_args()

    print("Running from command line with:")
    print(f"  vr_limit     = {args.vr_limit}")
    print(f"  radius_limit = {args.radius_limit}")
    print(f"  write        = {args.write}")

    # Run the core function to compute Sigma_Mstar and ngal arrays
    all_Sigma_Mstar, all_ngal = Sigma_Mstar_Ngal(vr_limit=args.vr_limit, radius_limit=args.radius_limit)

    if args.write:
        from astropy.table import Table
        import os

        homedir = os.getenv("HOME")
        cat_full = Table.read(homedir + '/Desktop/wisesize/nedlvs_parent_v1.fits')

        # Save the results to the table with your existing save_to_table function
        save_to_table(cat_full, all_Sigma_Mstar, all_ngal, version=1)

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


def get_sgy_bounds(SGY):
    # FOR COMPARISON WITH CASTIGNANI+2022 5NN 2D DENSITITIES FOR VFS!
    # "The 2D density is evaluated by including galaxies within a ΔSGY = 5.6 h^{−1} Mpc width, which 
    # corresponds to the 2σ statistical uncertainty along the line of sight at the distance of Virgo."
    
    upper_sgy_bound = SGY + 5.6/2
    lower_sgy_bound = SGY - 5.6/2
    
    return lower_sgy_bound, upper_sgy_bound


#FOR RA-DEC!
def plot_kNN(k, all_RA, all_DEC, all_kNN):

    #define "good" flag, which filters out all instances where galaxies did not have a kNN
    #given our isolation constraints
    good_flag = (all_kNN!=-999)
    
    fig = figure.Figure(figsize=(14,6))
    ax = fig.add_subplot()
        
    im = ax.scatter(all_RA[good_flag], all_DEC[good_flag], c=np.log10(all_kNN[good_flag]), cmap='viridis', alpha=0.5, s=5, vmin=-1, vmax=1.5)
    ax.invert_xaxis()
    ax.set_xlabel('RA [deg]',fontsize=14)
    ax.set_ylabel('DEC [deg]',fontsize=14)
    
    ax.tick_params(labelsize=14)
    
    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(fr'$\log$($\Sigma_{k}$/'+r'Mpc$^{-2}$)',fontsize=14)
    
    fig.savefig(homedir+f'/Desktop/{k}NN_plot.png', dpi=100, bbox_inches='tight', pad_inches=0.2)


def save_to_table(cat, all_kNN, k=5, version=1):
    
    #apply flags if they exist; else, 
    try:
        mstarflag = cat['Mstar_all_flag']
    except:
        if len(cat) != len(all_kNN):
            print('Input table length not same as kNN array length. Check that flag column names match the function!')
            print('Exiting.')
            sys.exit()
        mstarflag = np.ones(len(cat),dtype=bool)
    
    #also add WISESize flags
    raflag = (cat['RA']>87) & (cat['RA']<300)
    decflag = (cat['DEC']>-10) & (cat['DEC']<85)
    zflag = (cat['Z']>0.002) & (cat['Z']<0.025)

    #these are ALL flags applied to the 5NN input table
    flags = (mstarflag) & (zflag) & (raflag) & (decflag)

    all_kNN_parent = np.full(len(cat), -999)
    all_kNN_parent[flags] = all_kNN
    
    cat[f'2D_{k}NN'] = all_kNN_parent
    
    save_path = homedir+f'/Desktop/wisesize/nedlvs_parent_v{version}.fits'
    cat.write(save_path,overwrite=True)
    
    print(f'2D_{k}NN column added to (or updated in) {save_path}')
    
    
class central_galaxy():
    def __init__(self, ra, dec, redshift, k, cat, sgy=None, sgx=None, sgz=None):
        
        self.ra = ra
        self.dec = dec
        self.redshift = redshift
        self.k = k
        self.sgy = sgy
        self.sgx = sgx
        self.sgz = sgz
        
        self.cat = cat
        
        #for VFS, 5NN is actually calculated as 4NN. the difference is ultimately rather trivial, but for
        #consistency purposes I adjust the index accordingly
        #I remove the central galaxy THEN find the 5th smallest distance. conversely, VFS does not remove the
        #central galaxy
        #0, 1, 2, 3, 4, 5  (5 is FIFTH neighbor...exclude CENTRAL GALAXY!)
        #Castignani+2022 includes central galaxy, meaning they really calculate k=4
        
    def isolate_galaxy_region(self, vr_limit, radius_limit, virgo_env=None):
        
        #use redshifts to calculate width of "slice"
        if virgo_env is None:
            z_lower, z_upper = get_redshift_bounds(self.redshift, vr_limit)
            
            #from that list, cut galaxies which are beyond the redshift slice!
            z_flag = (self.cat['Z']>z_lower) & (self.cat['Z']<z_upper)
            
            cat = self.cat[z_flag]
            
            #this will very likely not ever be the case
            if radius_limit < 100.:
                ra_lower, ra_upper, dec_lower, dec_upper = get_radius_bounds(self.redshift,self.ra,self.dec,
                                                                         radius_limit)
                #generate a list of all galaxies whose RA and DEC values are within these bounds
                ra_flag = (cat['RA']>ra_lower) & (cat['RA']<ra_upper)
                dec_flag = (cat['DEC']>dec_lower) & (cat['DEC']<dec_upper)
                ra_dec_flag = ra_flag & dec_flag

                cat = cat[ra_dec_flag]
            
            self.trimmed_virgo_env=None
         
        #needed for VFS comparison, otherwise can ignore.
        else:
            sgy_lower, sgy_upper = get_sgy_bounds(self.sgy)            
            
            #cut galaxies which are beyonw the sgy slice!
            sgy_flag = (virgo_env['SGY']>sgy_lower) & (virgo_env['SGY']<sgy_upper)
            
            #sgy_flag = (np.abs(virgo_env['SGY']-self.sgy) < 5.6)    
            
            cat = cat[sgy_flag]
            
            self.trimmed_virgo_env = virgo_env[sgy_flag]
        
        self.trimmed_cat = cat
        
    
    #for galaxies within this "trimmed" catalog, calculate the projected distance
    #between these and the central galaxy
    def calc_projected_distances(self):
        
        #NUMPY VECTORIZATION! I previously used a python for loop, which was wildy sub-optimal (very slow)
        #checks first if user put in SG coordinates; if not, defaults to RA-DEC great circle distances
        if self.trimmed_virgo_env is not None:
            dx = self.sgx - self.trimmed_virgo_env['SGX']
            dz = self.sgz - self.trimmed_virgo_env['SGZ']
            self.projected_distances = np.sqrt(dx**2 + dz**2)

        elif self.sgy is not None:
            dx = self.sgx - self.trimmed_cat['sgx_arr']
            dz = self.sgz - self.trimmed_cat['sgz_arr']
            self.projected_distances = np.sqrt(dx**2 + dz**2)
        
        else:
            dist_rad = RADEC_to_dist_all(self.ra, self.dec, self.trimmed_cat['RA'], self.trimmed_cat['DEC'])
            self.projected_distances = rad_to_Mpc(dist_rad, self.redshift)
            
    
    #determine redshift bounds --> creates mask to isolate galaxies within the redshift limit
    #apply this AFTER performing the tree query
    def calc_bound_mask(self, vr_limit, virgo_env, indices):
        
        if virgo_env is not None:
            #filter neighbors by SGY
            #print(indices)
            #print(len(virgo_env))
            #print(np.max(indices))
            sgy = virgo_env['SGY'][indices]
            sgy_lower, sgy_upper = get_sgy_bounds(self.sgy)
            mask = (sgy >= sgy_lower) & (sgy <= sgy_upper)
            return mask
            
        z = cat['Z'][indices]
        z_lower, z_upper = get_redshift_bounds(self.redshift, vr_limit)
        mask = (z >= z_lower) & (z <= z_upper)
        return mask
    
    
    def calc_kSigma(self, virgo_env=None):
        
        #0th index is the central galaxy's distance to itself. 
        #the [1:] removes this 0th index
        sorted_distances = np.sort(self.projected_distances)[1:]
            
        if len(sorted_distances) >= self.k-1:
            #use the k valid neighbors, calculate projected distance. pull kth distance from r_k
            index = self.k-1 if virgo_env is None else self.k-2
            
            r_k = sorted_distances[index]
            self.density_kSigma = self.k / (np.pi * r_k**2)
            
        else:
            self.density_kSigma = -999
      
    
    def calc_kSigma_with_tree(self, tree, coord_array, vr_limit, virgo_env=None):
        
        #central galaxy position
        point = coord_array

        #get distances for 500 objects relative to central galaxy...and including the central galaxy. 
        #if self.k = 5, this means I am hoping that among the nearest 500 neighbors, at least 5 survive the z cut 
        
        dists, indices = tree.query(point, k=500)
        mask = self.calc_bound_mask(vr_limit, virgo_env, indices)
        
        zero_mask = (dists!=0)
        
        filtered_dists=dists[mask&zero_mask]
        
        #if k=600 is not sufficient, complete a second iteration with k=10000
        if len(filtered_dists) < self.k:
            dists, indices = tree.query(point, k=10000)
            mask = self.calc_bound_mask(vr_limit, virgo_env, indices)
            
            zero_mask = (dists!=0)
            
            filtered_dists=dists[mask&zero_mask]
            
        if len(filtered_dists) >= self.k-1:
            #use the k valid neighbors, calculate projected distance. pull kth distance from r_k
            index = self.k-1 if virgo_env is None else self.k-2
            
            r_k = filtered_dists[index]
            self.density_kSigma = self.k / (np.pi * r_k**2)
            
        else:
            self.density_kSigma = -999
    
    
    
if __name__ == "__main__":
    
    if '-h' or '-help' in sys.argv:
        print('-vr_limit [int in km/s; default is 500] -radius_limit [int in Mpc; default is 100 (no radius bounds)] -k [int; default is 5 (for fifth nearest neighbor)] -vfs [if included, will use VFS catalog and SGY bounds (from Castignani+22) in place of the vr_limit slice; otherwise, will default to NED-LVS catalog] -radec [will calculate great circle distances using RA-DEC coordinates; note this will be considerably slower than the default SG coordinates due to KDTrees only supporting linear distances] -write [will write array output to nedlvs_parent table]')
    
    if '-vr_limit' in sys.argv:
        p = sys.argv.index('-vr_limit')
        vr_limit = int(sys.argv[p+1])
        print(f'Using vr_limit = {vr_limit} km/s')
    else:
        vr_limit = 500  #km/s
        print('Using vr_limit = 500 km/s')
    
    if '-radius_limit' in sys.argv:
        p = sys.argv.index('-radius_limit')
        radius_limit = int(sys.argv[p+1])
        print(f'Using radius_limit = {radius_limit} Mpc')
    else:
        radius_limit = 100.  #Mpc
        print('Using radius_limit = 100 Mpc --> no limit!')
    
    if '-k' in sys.argv:
        p = sys.argv.index('-k')
        k = int(sys.argv[p+1])
        print(f'Using k = {k}')
    else:
        k = 5
        print(f'Using k = 5')
    
    if '-vfs' in sys.argv:
        phot_r = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_r_photometry.fits')
        M_r_flag = phot_r['M_r']<=-15.7
        print('Applying absolute r-band magnitude completeness flag (M_r<=-15.7) to VFS...')
        virgo_env = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_environment.fits')[M_r_flag]
        cat = Table.read(homedir+'/Desktop/virgowise_files/VF_WISESIZE_photSNR.fits')[M_r_flag]
        print('Using SG Coordinates from VFS catalogs...')

    else:
        virgo_env = None
        cat_full = Table.read(homedir+'/Desktop/wisesize/nedlvs_parent_v1.fits')
        
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
        
        print(f'Number of starting galaxies: {len(cat)}')
        
    all_kNN = np.zeros(len(cat))
    ra = cat['RA']
    dec = cat['DEC']
    redshift = cat['Z']
    
    if '-vfs' in sys.argv:
        sgx_arr = virgo_env['SGX']
        sgy_arr = virgo_env['SGY']
        sgz_arr = virgo_env['SGZ']
        
    elif '-radec' not in sys.argv:
        #need to use SG coordinates to avoid problems with celestial sphere configuration and its effect on RA distances
        #output arrays will have same length as ra, dec, redshift arrays
        sgx_arr, sgy_arr, sgz_arr = RADEC_to_SG(ra, dec, redshift)
        
        #add columns to main table
        cat['sgx_arr'] = sgx_arr
        cat['sgy_arr'] = sgy_arr
        cat['sgz_arr'] = sgz_arr
        
    else:
        #assuming user wants RADEC distances, set SG coords to be None arrays
        sgx_arr = sgy_arr = sgz_arr = [None] * len(cat)
    
    
    coords = np.vstack((sgx_arr,sgz_arr)).T
    
    start_time = time.perf_counter()
    
    #ignore tree if user wants to use RA-DEC distances
    tree = KDTree(coords) if '-radec' not in sys.argv else None

    for n in range(len(cat)):
        #define galaxy class object
        galaxy = central_galaxy(ra[n], dec[n], redshift[n], k, cat, sgy_arr[n], sgx_arr[n], sgz_arr[n])
        
        #compute kSigma using the tree if using supergalactic coordinates
        if '-radec' not in sys.argv:
            galaxy.calc_kSigma_with_tree(tree, coords[n], vr_limit, virgo_env)
            all_kNN[n] = galaxy.density_kSigma
        
        else:
            #set up trimmed catalog (which restricts galaxies to be within redshift (and possibly radius) slice
            galaxy.isolate_galaxy_region(vr_limit, radius_limit)
            
            #self-explanatory -- runs the function to calculated all projected distances relative to central galaxy
            galaxy.calc_projected_distances()
            
            galaxy.calc_kSigma()
            all_kNN[n] = galaxy.density_kSigma

    
    #plot_kNN(k, cat['RA'], cat['DEC'], all_kNN)
    print('# galaxies in input table without kSigma:',len(all_kNN[all_kNN==-999]))
    
    if '-write' in sys.argv:
        save_to_table(cat_full, all_kNN, k, version=1)
    
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    print(f'Execution Time: {(execution_time/60.):.2} minute(s)')
    
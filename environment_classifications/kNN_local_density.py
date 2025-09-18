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
    #also plot galaxies omitted from Sigma5 calculations due to edge effects
    ax.scatter(all_RA[~good_flag], all_DEC[~good_flag], alpha=0.5, color='crimson',s=3,label='Removed (Edge Effects)')
    ax.invert_xaxis()
    ax.set_xlabel('RA [deg]',fontsize=14)
    ax.set_ylabel('DEC [deg]',fontsize=14)
    
    ax.axhline(-10,alpha=0.2,color='black')
    ax.axhline(85,alpha=0.2,color='black')
    
    ax.axvline(87,alpha=0.2,color='black')
    ax.axvline(300,alpha=0.2,color='black')
    
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
    
    
    #eliminate edge effects - flag galaxies which may have biased Sigma_M and Ngal due to their radii extending beyond the limits of the survey area
    def check_galaxy_buffer(self, radius_limit=3):
        
        #find inner ra and dec buffer zones
        #I need to convert the radius limit to degrees, to start. 
        #note this will differ depending on redshift (AND declination...yay great circle distances).
        radius_limit_degrees = Mpc_to_deg(radius_limit, self.redshift)

        #define the 'buffer zones' and the flags corresponding to whether the galaxy is within them
        
        #left vertical
        inner_ra_one = 87. + radius_limit_degrees
        buffer_zone_one = (self.ra<=inner_ra_one) & (self.ra>=87.)
        
        #right vertical
        inner_ra_two = 300. - radius_limit_degrees
        buffer_zone_two = (self.ra>=inner_ra_two) & (self.ra<=300.)
        
        #top horizontal
        inner_dec_one = 85. - radius_limit_degrees
        buffer_zone_three = (self.dec>=inner_dec_one) & (self.dec<=85.)
        
        #bottom horizontal
        inner_dec_two = -10. + radius_limit_degrees
        buffer_zone_four = (self.dec<=inner_dec_two) & (self.dec>=-10.)
        
        #return True or False depending on whether galaxy is within any of the buffer zones
        return (buffer_zone_one | buffer_zone_two | buffer_zone_three | buffer_zone_four)
    
    
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

            

##########################################   
#NEED for when importing code as a module!
##########################################
def compute_kNN_densities(vr_limit=500, radius_limit=100, k=5, use_vfs=False, use_radec=False, write=False):
    """
    Computes kNN densities for a galaxy catalog, either VFS or NED-LVS.
    
    Parameters
    ----------
    vr_limit : int
        Velocity range limit in km/s (only used if use_vfs is False).
    radius_limit : float
        Radius limit in Mpc for spatial trimming.
    k : int
        k in the k-nearest neighbor calculation.
    use_vfs : bool
        Whether to use the VFS catalog instead of NED-LVS.
    use_radec : bool
        If True, computes distances from RA/DEC (slower); if False, uses SG coordinates with KDTree.
    write : bool
        If True, will write the output back into nedlvs_parent FITS table.

    Returns
    -------
    all_kNN : np.ndarray
        Array of kNN surface densities.
    """
    print(f"Running kNN density calculation with k={k}, vr_limit={vr_limit}, radius_limit={radius_limit}")
    import time
    from astropy.table import Table
    from scipy.spatial import KDTree
    import numpy as np
    import os
    
    homedir = os.getenv("HOME")
    
    if use_vfs:
        phot_r = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_r_photometry.fits')
        M_r_flag = phot_r['M_r'] <= -15.7
        virgo_env = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_environment.fits')[M_r_flag]
        cat = Table.read(homedir+'/Desktop/virgowise_files/VF_WISESIZE_photSNR.fits')[M_r_flag]
        print('Using VFS catalog with SG coordinates.')
    else:
        virgo_env = None
        cat_full = Table.read(homedir+'/Desktop/wisesize/nedlvs_parent_v1.fits')
        
        try:
            mstarflag = cat_full['Mstar_all_flag']
        except:
            mstarflag = np.ones(len(cat_full), dtype=bool)
            print('Mass completeness flag not found.')
        
        zflag = (cat_full['Z'] > 0.002) & (cat_full['Z'] < 0.025)
        raflag = (cat_full['RA'] > 87) & (cat_full['RA'] < 300)
        decflag = (cat_full['DEC'] > -10) & (cat_full['DEC'] < 85)
        cat = cat_full[mstarflag & zflag & raflag & decflag]
        print(f'NED-LVS catalog selected with {len(cat)} galaxies after applying flags.')
    
    all_kNN = np.zeros(len(cat))
    ra = cat['RA']
    dec = cat['DEC']
    redshift = cat['Z']

    if use_vfs:
        sgx_arr = virgo_env['SGX']
        sgy_arr = virgo_env['SGY']
        sgz_arr = virgo_env['SGZ']
    elif not use_radec:
        sgx_arr, sgy_arr, sgz_arr = RADEC_to_SG(ra, dec, redshift)
        cat['sgx_arr'] = sgx_arr
        cat['sgy_arr'] = sgy_arr
        cat['sgz_arr'] = sgz_arr
    else:
        sgx_arr = sgy_arr = sgz_arr = [None] * len(cat)
    
    coords = np.vstack((sgx_arr, sgz_arr)).T
    tree = KDTree(coords) if not use_radec else None
    
    start_time = time.perf_counter()
    
    for n in range(len(cat)):
        
        galaxy = central_galaxy(ra[n], dec[n], redshift[n], k, cat, sgy_arr[n], sgx_arr[n], sgz_arr[n])
        
        #check whether galaxy is within the buffer zones. default is 3 Mpc if user has not defined a limit
        flag=galaxy.check_galaxy_buffer(radius_limit) if (radius_limit<100) else galaxy.check_galaxy_buffer()
        if flag:
            all_kNN[n] = -999
            continue   #if galaxy is within this buffer zone, assign -999

        if not use_radec:
            galaxy.calc_kSigma_with_tree(tree, coords[n], vr_limit, virgo_env)
        else:
            galaxy.isolate_galaxy_region(vr_limit, radius_limit)
            galaxy.calc_projected_distances()
            galaxy.calc_kSigma()
        
        all_kNN[n] = galaxy.density_kSigma

    print(f"# galaxies with no valid kNN: {(all_kNN == -999).sum()}")

    if write and not use_vfs:
        save_to_table(cat_full, all_kNN, k, version=1)

    print(f"Finished in {(time.perf_counter() - start_time)/60:.2f} minutes")
    
    return all_kNN
            
   
    
#######################################################################################################
#this part ONLY RUNS if the code is run from a command line as opposed to if it is imported as a module
#######################################################################################################    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute kNN galaxy densities")
    parser.add_argument("-vr_limit", type=int, default=500, help="Velocity range limit in km/s")
    parser.add_argument("-radius_limit", type=float, default=100.0, help="Radius limit in Mpc")
    parser.add_argument("-k", type=int, default=5, help="k-th nearest neighbor")
    parser.add_argument("-vfs", action="store_true", help="Use the VFS catalog")
    parser.add_argument("-radec", action="store_true", help="Use RA/DEC for distance instead of SG coordinates")
    parser.add_argument("-write", action="store_true", help="Write results back into table")

    args = parser.parse_args()

    print("Running from command line with:")
    print(f"  vr_limit      = {args.vr_limit}")
    print(f"  radius_limit  = {args.radius_limit}")
    print(f"  k             = {args.k}")
    print(f"  use_vfs       = {args.vfs}")
    print(f"  use_radec     = {args.radec}")
    print(f"  write_results = {args.write}")

    #bzzt. call function.
    all_kNN = compute_kNN_densities(
        vr_limit=args.vr_limit,
        radius_limit=args.radius_limit,
        k=args.k,
        use_vfs=args.vfs,
        use_radec=args.radec,
        write=args.write)
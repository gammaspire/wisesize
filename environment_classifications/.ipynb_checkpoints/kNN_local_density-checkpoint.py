import numpy as np
from astropy.table import Table
from matplotlib import figure 
from scipy.spatial import KDTree
import time
import sys

import os
homedir=os.getenv("HOME")

def convert_to_Mpc(dist_degrees, redshift):
    #r_k is initially in degrees.
    #degrees --> radians --> Mpc
    
    dist_radians = dist_degrees * np.pi / 180.   #radians
    
    z_distance = (3.e5 * redshift)/74.   #Mpc
    
    dist_mpc = z_distance * np.tan(dist_radians)   #Mpc
    
    return dist_mpc

def get_sgy_bounds(SGY):
    # FOR COMPARISON WITH CASTIGNANI+2022 5NN 2D DENSITITIES FOR VFS!
    # "The 2D density is evaluated by including galaxies within a ΔSGY = 5.6 h^{−1} Mpc width, which 
    # corresponds to the 2σ statistical uncertainty along the line of sight at the distance of Virgo."
    
    upper_sgy_bound = SGY + 5.6/2
    lower_sgy_bound = SGY - 5.6/2
    
    return lower_sgy_bound, upper_sgy_bound

def get_redshift_bounds(redshift, vr_limit):
    #because these are nearby galaxies, v=cz approximately holds
    #for the upper and lower vcosmic bounds, default is 500 km/s
    #convert that to redshift...then find the bounds.
    redshift_sigma = vr_limit/3.e5
    
    upper_z_bound = redshift + redshift_sigma
    lower_z_bound = redshift - redshift_sigma
    
    return lower_z_bound, upper_z_bound

def get_radius_bounds(redshift, ra, dec, radius_limit):
    
    #we also want a "radius" of radius_sigma --> want galaxies within radius_sigma 'square' around central galaxy
    #convert radius_sigma to arcsec for easy comparison between RA and DEC values!
    #default radius_sigma is 2. Mpc
    #v=H0d=cz also approximately holds, where H0 = 74 km/s/Mpc
    distance = (3.e5 * redshift)/74.   #Mpc
    
    #to find the radius corresponding to radius_sigma, tan(theta) = radius_sigma/d
    #theta = arctan(radius_sigma/d)   #RADIANS
    radius_sigma = np.arctan(radius_limit/distance)   #radians
    radius_sigma = radius_sigma*180./np.pi   #degrees   
    
    #define upper and lower bounds --> the coordinate must be within 4 Mpc of central galaxy
    #can loosely define this as the RA and DEC values never exceeding +/- 4 Mpc
    upper_RA_bound = ra + radius_sigma   #value of max RA if DEC doesn't change
    upper_DEC_bound = dec + radius_sigma   #value of max DEC if RA doesn't change
    lower_RA_bound = ra - radius_sigma   #value of min RA if DEC doesn't change
    lower_DEC_bound = dec - radius_sigma   #value of min DEC if RA doesn't change
    
    return lower_RA_bound, upper_RA_bound, lower_DEC_bound, upper_DEC_bound

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
    
    
class central_galaxy():
    def __init__(self, ra, dec, redshift, k, sgy=None, sgx=None, sgz=None):
        
        self.ra = ra
        self.dec = dec
        self.redshift = redshift
        self.k = k
        self.sgy = sgy
        self.sgx = sgx
        self.sgz = sgz
        
        #for VFS, 5NN is actually calculated as 4NN. the difference is ultimately rather trivial, but for
        #consistency purposes I adjust the index accordingly
        #I remove the central galaxy THEN find the 5th smallest distance. conversely, VFS does not remove the
        #central galaxy
        self.neighbor_index = self.k+1  #0, 1, 2, 3, 4, 5  (5 is FIFTH neighbor...exclude CENTRAL GALAXY!)
        if self.sgy != None:
            self.neighbor_index = self.k  #Castignani+2022 includes central galaxy, meaning they really calculate k=4
        
    def isolate_galaxy_region(self, cat, vr_limit, radius_limit, virgo_env=None):
        
        #if no sgy value, then use redshifts to calculate width of "slice"
        if self.sgy == None:
            z_lower, z_upper = get_redshift_bounds(self.redshift, vr_limit)
            
            #from that list, cut galaxies which are beyond the redshift slice!
            z_flag = (cat['Z']>z_lower) & (cat['Z']<z_upper)
            
            cat = cat[z_flag]
        
            if radius_limit < 100.:
                ra_lower, ra_upper, dec_lower, dec_upper = get_radius_bounds(self.redshift,self.ra,self.dec,
                                                                         radius_limit)
                #generate a list of all galaxies whose RA and DEC values are within these bounds
                ra_flag = (cat['RA']>ra_lower) & (cat['RA']<ra_upper)
                dec_flag = (cat['DEC']>dec_lower) & (cat['DEC']<dec_upper)
                ra_dec_flag = ra_flag & dec_flag

                cat = cat[ra_dec_flag]
         
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
        if self.sgy is not None:
            dx = self.sgx - self.trimmed_virgo_env['SGX']
            dz = self.sgz - self.trimmed_virgo_env['SGZ']
            self.projected_distances = np.sqrt(dx**2 + dz**2)
        else:
            d_ra = self.ra - self.trimmed_cat['RA']
            d_dec = self.dec - self.trimmed_cat['DEC']
            self.projected_distances = np.sqrt(d_ra**2 + d_dec**2)

    
    def calc_bound_mask(self, vr_limit, virgo_env, indices):
        
        if self.sgy is not None:
            #filter neighbors by SGY
            sgy = virgo_env['SGY'][indices]
            sgy_lower, sgy_upper = get_sgy_bounds(self.sgy)
            mask = (sgy >= sgy_lower) & (sgy <= sgy_upper)
            return mask
            
        z = cat['Z'][indices]
        z_lower, z_upper = get_redshift_bounds(self.redshift, vr_limit)
        mask = (z >= z_lower) & (z <= z_upper)
        return mask
    
    def calc_kSigma_with_tree(self, tree, coord_array, vr_limit, virgo_env=None):
        
        #central galaxy position
        point = coord_array

        #get distances for 100 objects relative to central galaxy...and including the central galaxy. 
        #if self.k = 5, this means I am hoping that among the nearest 100 neighbors, at least 5 survive the z cut 
        
        dists, indices = tree.query(point, k=600)
        mask = self.calc_bound_mask(vr_limit, virgo_env, indices)
        
        zero_mask = (dists!=0)
        
        filtered_dists=dists[mask&zero_mask]
        
        #if k=200 is not sufficient, complete a second iteration with k=10000
        if len(filtered_dists) < self.k:
            dists, indices = tree.query(point, k=10000)
            mask = self.calc_bound_mask(vr_limit, virgo_env, indices)
            
            zero_mask = (dists!=0)
            
            filtered_dists=dists[mask&zero_mask]
            
        if len(filtered_dists) >= self.k:
            #use the k valid neighbors, calculate projected distance. pull kth distance from r_k
            index = self.k-1 if self.sgy is None else self.k   
            
            r_k = filtered_dists[index]
            self.density_kSigma = self.k / (np.pi * r_k**2)
            
        else:
            self.density_kSigma = -999

            
            
            
    #FOLLOWING IS NOW ARCHIVED. KEEPING FOR MUSEUM PURPOSES.
    
    #from list of projected distance from nearby galaxies in the RA-DEC-z slice,
    #calculate the kSigma density for the central galaxy
    def calc_kSigma(self):
    
        #arrange from closeset to farthest
        self.projected_distances = np.sort(self.projected_distances)

        #remove the '0' element, since that is the distance of the central galaxy to itself!
        self.projected_distances = np.delete(self.projected_distances,0)

        #now...calculate
        try:
            #projected distance to 5th nearest neighbor
            r_k = self.projected_distances[self.neighbor_index]

            #convert back to Mpc (ONLY if not using supergalactic coordinates!)
            if self.sgy==None:
                r_k = convert_to_Mpc(r_k, self.redshift)

            self.density_kSigma = self.k/(np.pi * r_k**2)
    
        #if there are not >4 galaxies in the array, then return a NaN.
        except:
            self.density_kSigma = -999

            
            
            
if __name__ == "__main__":
    
    if '-h' or '-help' in sys.argv:
        print('-vr_limit [int in km/s; default is 500] -radius_limit [int in Mpc; default is 100 (no radius bounds)] -k [int; default is 5 (for fifth nearest neighbor)] -vfs [if included, will use VFS catalog and SGY bounds (from Castignani+22) in place of the vr_limit slice; otherwise, will default to WISESize catalog]')
    
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
        print('Applying absolute r-band magnitude completeness flag (M_r<=-15.7)...')
        virgo_env = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_environment.fits')[M_r_flag]
        cat = Table.read(homedir+'/Desktop/virgowise_files/VF_WISESIZE_photSNR.fits')[M_r_flag]
        print('Using SGY from VFS catalogs...')
    else:
        virgo_env = None
        #cat = Table.read(homedir+'/Desktop/wisesize/wisesize_v2.fits')
        cat = Table.read(homedir+'/Desktop/wisesize/nedlvs_parent_v1.fits')
        
    all_kNN = np.zeros(len(cat))
    ra = cat['RA']
    dec = cat['DEC']
    redshift = cat['Z']
    
    if virgo_env is not None:
        sgx_arr = virgo_env['SGX']
        sgy_arr = virgo_env['SGY']
        sgz_arr = virgo_env['SGZ']
        coords = np.vstack((sgx_arr,sgz_arr)).T
    else:
        sgx_arr = sgy_arr = sgz_arr = [None] * len(cat)
        coords = np.vstack((ra,dec)).T
    
    
    start_time = time.perf_counter()
    
    tree = KDTree(coords)

    for n in range(len(cat)):
        galaxy = central_galaxy(ra[n], dec[n], redshift[n], k, sgy_arr[n], sgx_arr[n], sgz_arr[n])
        
        #compute kSigma using the tree
        galaxy.calc_kSigma_with_tree(tree, coords[n], vr_limit, virgo_env)
        all_kNN[n] = galaxy.density_kSigma

    
    plot_kNN(k, cat['RA'], cat['DEC'], all_kNN)
    print('Number of Galaxies without kSigma:',len(all_kNN[all_kNN==-999]))
    
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    print(f'Execution Time: {(execution_time/60.):.2} minute(s)')
    
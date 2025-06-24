####################################################################################################
#A collection of "universal functions" that can be applied to any number of scripts for this project
####################################################################################################

###CONVERT TO SUPERGALACTIC COORDINATES###
#input: RA, DEC, Z arrays
#wil return SGX, SGY, SGZ coordinate arrays
#if any redshifts are for whatever reason negative, the function will print the number of negative entries 
#and replace the corresponding index of the SG arrays with -999
#USES H0 = 74 km/s/Mpc
#output: sgx, sgy, sgz arrays row-matched to input arrays
def RADEC_to_SG(RA, DEC, Z):
    
    import astropy.units as u
    from astropy.coordinates import SkyCoord, ICRS, Galactic, Supergalactic
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    homedir=os.getenv("HOME")

    zflag = (Z<0)
    print(f'Number of objects fed into RADEC_to_SG() with negative redshift: {len(Z[zflag])}')
    
    ra = RA[~zflag]*u.deg
    dec = DEC[~zflag]*u.deg
    z = Z[~zflag]
    c = 3e5 * u.km/u.s
    H0 = 74 *u.km/u.s/u.Mpc
    distance = z*c/H0
    
    #create SkyCoord object
    c_icrs = SkyCoord(ra=ra, dec=dec, distance=distance, frame='icrs')
    
    #transform to supergalactic
    c_sgc = c_icrs.transform_to(Supergalactic())   #outputs SGL, SGB, distance
    
    #SGL, SGB, distance --> SGX, SGY, SGZ
    sgx = c_sgc.represent_as('cartesian').x.value
    sgy = c_sgc.represent_as('cartesian').y.value
    sgz = c_sgc.represent_as('cartesian').z.value

    #if True, keep sgx value
    #if False, np.nan
    sgx_arr = np.full(len(zflag), np.nan)
    sgx_arr[~zflag] = sgx
    
    sgy_arr = np.full(len(zflag), np.nan)
    sgy_arr[~zflag] = sgy
    
    sgz_arr = np.full(len(zflag), np.nan)
    sgz_arr[~zflag] = sgz

    #plot for diagnostic purposes. which SG coordinate best correlates with redshift?
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sc1 = axs[0].scatter(z, sgx_arr[~zflag], s=2)
    axs[0].set_xlabel('Redshift')
    axs[0].set_ylabel('|SGX [Mpc]|')

    sc2 = axs[1].scatter(z, sgy_arr[~zflag], s=2)
    axs[1].set_xlabel('Redshift')
    axs[1].set_ylabel('|SGY [Mpc]|')

    sc3 = axs[2].scatter(z, sgz_arr[~zflag], s=2)
    axs[2].set_xlabel('Redshift')
    axs[2].set_ylabel('|SGZ [Mpc]|')

    plt.tight_layout()
    
    save_loc = homedir+'/Desktop/redshift_SG.png'
    plt.savefig(save_loc,dpi=100, bbox_inches='tight', pad_inches=0.2)    
    print(f'Diagnostic plot saved to: {save_loc}')
    
    return sgx_arr, sgy_arr, sgz_arr


###CALCULATE RA-DEC 'GREAT CIRCLE' DISTANCES ON CELESTIAL SPHERE###
#input: (RA1, DEC1) of object one and (RA2, DEC2) of object two, redshift of CENTRAL galaxy
#uses astropy.coordinates.separation to "compute great circle distances"
#output: distance in radians
def RADEC_to_dist(RA1, RA2, DEC1, DEC2):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    obj_1 = SkyCoord(RA1*u.deg, DEC1*u.deg, frame='icrs')
    obj_2 = SkyCoord(RA2*u.deg, DEC2*u.deg, frame='icrs')
    
    dist_deg = obj_1.separation(obj_2)
    dist_rad = dist_deg.radian

    return dist_rad


###CALCULATE RA-DEC 'GREAT CIRCLE' DISTANCES ON CELESTIAL SPHERE###
#input: (RA1, DEC1) of object one and (RA_array, DEC_array) arrays for rest of sample
#uses astropy.coordinates.separation to "compute great circle distances"
#output: distance array row-matched to input arrays
def RADEC_to_dist_all(RA1, DEC1, RA_array, DEC_array):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    obj_1 = SkyCoord(RA1*u.deg, DEC1*u.deg, frame='icrs')
    obj_all = SkyCoord(RA_array*u.deg, DEC_array*u.deg, frame='icrs')
    
    dist_deg = obj_1.separation(obj_all)
    dist_rad = dist_deg.radian

    return dist_rad


###CONVERT CELESTIAL SPHERE 'GREAT CIRCLE' DISTANCES TO MPC###
#input: radial distance, redshift of CENTRAL galaxy (we flatten onto a spherical surface with a radius of the
#central galaxy's redshift)
#assume H0 = 74 km/s/Mpc
#returns dist_mpc, which is the arc-length of the radial distance
def convert_to_Mpc(rad_dist, redshift):
    #--> radians --> Mpc
    
    r = (3.e5 * redshift)/74.   #Mpc
    
    # S = r * theta
    dist_mpc = r * rad_dist   #Mpc
    
    return dist_mpc



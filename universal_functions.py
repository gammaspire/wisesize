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


###CALCULATE RA-DEC-Z SPHERICAL DISTANCES###
#input: (RA1, DEC1, Z1) of object one and (RA_array, DEC_array, Z_array) arrays for rest of sample
#uses astropy.coordinates.separation to "compute great circle distances"
#output: 3D-distance array (in MPC) row-matched to input arrays
def RADECZ_to_dist_all(RA1, DEC1, Z1, RA_array, DEC_array, Z_array):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    #failsafe to not bother 
    if any(Z_array<0) or (Z1<0):
        print('negative redshifts detected!')
        return np.array([])
    
    obj_1 = SkyCoord(RA1*u.deg, DEC1*u.deg, distance=Z1*3.e5*u.Mpc/74., frame='icrs')
    obj_all = SkyCoord(RA_array*u.deg, DEC_array*u.deg, distance=Z_array*3.e5*u.Mpc/74., frame='icrs')
    
    dist_mpc = (obj_1.separation_3d(obj_all))
    
    return dist_mpc.value




###CONVERT CELESTIAL SPHERE 'GREAT CIRCLE' DISTANCES TO MPC###
#input: radial distance, redshift of CENTRAL galaxy (we flatten onto a spherical surface with a radius of the
#central galaxy's redshift)
#assume H0 = 74 km/s/Mpc
#returns dist_mpc, which is the arc-length of the radial distance
def rad_to_Mpc(rad_dist, redshift):
    #radians --> Mpc
    r = (3.e5 * redshift)/74.   #Mpc; distance to object
    
    # S = r * theta
    dist_mpc = r * rad_dist   #Mpc
    
    return dist_mpc


###CONVERT MPC DISTANCE TO DEGREES###
#input: dist in Mpc, redshift of central object
#assume H0 = 74 km/s/Mpc
#returns distance in Mpc
def Mpc_to_deg(dist_mpc, redshift):
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    
    r = (3.e5 * redshift)/74.   #Mpc; distance to object
    
    
    
    #theta = S/r
    dist_rad = (dist_mpc/r) * u.radian   #radians
    
    #convert radians to degrees
    dist_deg = dist_rad.to(u.deg)
    
    return dist_deg.value


###CALCULATE UPPER AND LOWER REDSHIFT BOUNDS GIVEN RECESSION VELOCITY LIMIT###
#input: redshift of central object, recession velocity limit (redshift +/- vr_limit)
#this linear scaling only applies to low-z objects!
#returns upper and lower z bounds
def get_redshift_bounds(redshift, vr_limit):
    #because these are nearby galaxies, v=cz approximately holds
    #convert vr_limit to redshift...then find the bounds.
    redshift_sigma = vr_limit/3.e5
    
    upper_z_bound = redshift + redshift_sigma
    lower_z_bound = redshift - redshift_sigma
    
    #prevents any negative values from entering the scene.
    if upper_z_bound<0.:
        upper_z_bound=0
    if lower_z_bound<0:
        lower_z_bound=0
    
    return lower_z_bound, upper_z_bound


###CALCULATE UPPER AND LOWER RA-DEC BOUNDS GIVEN RADIUS LIMIT###
#input: redshift, ra, dec of central object; radius limit in Mpc
#returns upper and lower bounds for ra and dec in degrees
def get_radius_bounds(redshift, ra, dec, radius_limit):
    import numpy as np
    
    #we also want a "radius" of radius_sigma --> want galaxies within radius_sigma 'square' around central galaxy
    #convert radius_sigma to arcsec for easy comparison between RA and DEC values
    #v=H0d=cz also approximately holds, where H0 = 74 km/s/Mpc
    distance = (3.e5 * redshift)/74.   #Mpc
    
    #to find the radius corresponding to radius_sigma, tan(theta) = radius_sigma/d
    #theta = arctan(radius_sigma/d)   #RADIANS
    radius_sigma = np.arctan(radius_limit/distance)   #radians
    radius_sigma = radius_sigma*180./np.pi   #degrees   
    
    #define upper and lower bounds --> the coordinate must be within 4 Mpc of central galaxy
    #can loosely define this as the RA and DEC values never exceeding +/- 4 Mpc
    upper_RA_bound = (ra + radius_sigma) % 360   #value of max RA if DEC doesn't change
    upper_DEC_bound = dec + radius_sigma         #value of max DEC if RA doesn't change
    lower_RA_bound = (ra - radius_sigma) % 360   #value of min RA if DEC doesn't change
    lower_DEC_bound = dec - radius_sigma         #value of min DEC if RA doesn't change
    
    return lower_RA_bound, upper_RA_bound, lower_DEC_bound, upper_DEC_bound


###CALCULATE RADIAL BOUND FLAG GIVEN RADIUS LIMIT###
#input: redshift, ra, dec of central object; ra, dec arrays of all objects; radius limit in Mpc
#returns boolean flag row-matched to ra_all and dec_all identifying objects which are within the radius limit
def get_radius_flag(redshift, ra, dec, ra_all, dec_all, radius_limit):
    import numpy as np
    
    dist_deg = Mpc_to_deg(radius_limit, redshift)
    
    flag = (np.sqrt((ra-ra_all)**2 + (dec-dec_all)**2) <= dist_deg)
    
    return flag
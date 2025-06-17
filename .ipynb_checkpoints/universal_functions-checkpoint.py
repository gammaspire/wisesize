####################################################################################################
#A collection of "universal functions" that can be applied to any number of scripts for this project
####################################################################################################

###CONVERT TO SUPERGALACTIC COORDIANTES###
#input: RA, DEC, Z arrays
#wil return SGX, SGY, SGZ coordinate arrays
#if any redshifts are for whatever reason negative, the function will print the number of negative entries 
#and replace the corresponding index of the SG arrays with -999
#USES H0 = 74 km/s/Mpc
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

    sc1 = axs[0].scatter(z, np.abs(sgx_arr[~zflag]), s=2)
    axs[0].set_xlabel('Redshift')
    axs[0].set_ylabel('|SGX [Mpc]|')

    sc2 = axs[1].scatter(z, np.abs(sgy_arr[~zflag]), s=2)
    axs[1].set_xlabel('Redshift')
    axs[1].set_ylabel('|SGY [Mpc]|')

    sc3 = axs[2].scatter(z, np.abs(sgz_arr[~zflag]), s=2)
    axs[2].set_xlabel('Redshift')
    axs[2].set_ylabel('|SGZ [Mpc]|')

    plt.tight_layout()
    
    save_loc = homedir+'/Desktop/redshift_SG.png'
    plt.savefig(save_loc,dpi=100, bbox_inches='tight', pad_inches=0.2)    
    print(f'Diagnostic plot saved to: {save_loc}')
    
    return sgx_arr, sgy_arr, sgz_arr
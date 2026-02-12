'''
GOAL: determine SFR completeness limit for NED-LVS parent sample 

CUT BY SNR FIRST!

Steps:
    * Create a plot of SFR vs. redshift for all galaxies and for those above our WISE SNR cut. Use the lowest 5% SFR at the upper 20% of our velocity range to determine our SFR limit.
    * Identify the (SNR cut) subsample's redshift value at and beyond which the elements in the array represent the 20% highest redshifts.
    * Of these 20% highest values, we now must isolate the galaxies with the lowest 5% SFR values.
'''    

import numpy as np


#create hexbin SFR vs. Mstar plot of sample
def plot_sfrmstar(logSFR, logMstar, SFRlimit, hexbin=True, nbins=200):

    from matplotlib import pyplot as plt
    
    #hexbin version of log(SFR) vs. log(Mstar) plot

    plt.figure(figsize=(5,4))
    
    if hexbin:
        plt.hexbin(logMstar,logSFR,cmap='Greens',label='All Galaxies',gridsize=nbins,bins='log')
    else:
        plt.scatter(logMstar,logSFR,color='green',alpha=0.2,s=10,label='All Galaxies')
    
    plt.xlabel(r'log(Mstar / [M$_\odot$])',fontsize=12)
    plt.ylabel(r'log(SFR / [M$_\odot$ yr$^{-1}$])',fontsize=12)

    plt.axhline(SFRlimit,color='red',alpha=0.4,ls='--',label=f'SFR Completeness = {SFRlimit:.3f}')

    plt.xlim(8,12)
    plt.ylim(-4,4)

    plt.legend(fontsize=8)
    #plt.title(r'NED-LVS Galaxies [using m$_J$ = 16.6 limit]',fontsize=12)

    plt.show()


def get_zflag(z, percentile):
    #sort redshift array
    z_sorted = np.sort(z)
    
    #define length of z_sorted
    z_len = len(z_sorted)
    
    #determine the "percentile" index
    index = int(z_len*percentile)
    
    #find the value at this index
    z_lim = z_sorted[index]
    
    #lastly...define the flag
    z_flag = (z > z_lim)
    
    return z_flag
    
    
def get_sfrflag(sfr_cut):
    
    #sort sfr values from least to greatest
    sfr_sorted = np.sort(sfr_cut)
    
    #define length of sfr_sorted
    sfr_len = len(sfr_sorted)
    
    #multiply length by 0.05 to find the "5%th" index
    index = int(sfr_len*0.05)
    
    #the value at this index
    sfr_limit = sfr_sorted[index]
    
    #the flag removes the lowest 5% SFR values within the redshift limit. 
    sfr_flag = (sfr_cut > sfr_limit)
    
    return sfr_limit, sfr_flag


def sfr_completeness(z, logSFR, z_quantile=0.8, sfr_quantile=0.05, plot=False, logMstar=None):
    """
    Heuristic SFR completeness limit:
    - take galaxies above the z_quantile of z (e.g. top 20% in redshift)
    - define completeness as the sfr_quantile of logSFR in that high-z subset (e.g. 5th percentile)

    Assumes you've already applied your WISE SNR cut before calling.
    """
    z = np.asarray(z)
    logSFR = np.asarray(logSFR)

    #some fun filters -- do not want infinite values.
    ok = np.isfinite(z) & np.isfinite(logSFR)
    z = z[ok]
    logSFR = logSFR[ok]

    z_lim = np.quantile(z, z_quantile)
    highz = z >= z_lim

    sfr_limit = np.quantile(logSFR[highz], sfr_quantile)

    if plot:
        plot_sfrmstar(logSFR, logMstar, sfr_limit, hexbin=True, nbins=200)
    
    print(f"SFR Completeness (heuristic): {sfr_limit:.3f}")
    return sfr_limit
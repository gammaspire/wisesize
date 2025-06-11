'''
SEE COMPANION NOTEBOOK FOR DETAILED NOTES CONCERNING THE MASS COMPLETENESS STEPS

NOTE: be sure to apply any other needed cuts to your sample before calculating the mass completeness limit

GOAL: calculate mass completeness limit for a given sample of galaxies (given the input luminosity, magnitude, stellar mass, and redshift arrays)

To run all in, say, a Jupyter Notebook, type
%run mass_completeness.py
mass_completeness(mag, mag_lim, z, z_max, Mstar, percentile, sfr=None, plot=False, hexbin=False, nbins=200)
    * percentile can range from, preferably, 0.90 to 0.95
    * mag, z, and Mstar should be row-matched and taken from galaxy sample 
        * galaxy sample should have all other cuts already applied (e.g., S/N, morphology)
    * mag_lim is m_J = 16.6, m_r = 17.7
    * z_max is the upper redshift limit of survey or galaxy sample
    * switch to plot=True for output plot of SFR vs. Mstar with mass completeness limit overlay. 
        * for hexbin plot, can change nbins to dictate the number of bins shown. > larger number means finer bins
        * default nbin is 200
'''

import numpy as np
from matplotlib import pyplot as plt
import os
homedir=os.getenv('HOME')
from astropy.table import Table


#isolate galaxies that are near the upper limit of our redshift range (z_max).
#returns the flag which applies this condition
def get_zflag(z, z_max):
    return (np.abs(z_max-z)<0.05*z_max) #& (z<=z_max)


#isolate galaxies which are sufficiently brighter than our survey's magnitude limit
#for m_J (NED-LVS), mag_lim = 16.6
#for m_r (SDSS), mag_lim = 17.7
#returns the flag which applies this condition
def get_magflag(mag, mag_lim):
    delta_mag = mag - mag_lim
    return (delta_mag<-0.5) & (delta_mag>-1.0)


#calculate Lscale
#be sure to only run this function AFTER applying the flags above!
#that is, delta_mag should be freshly generated with the mag array AFTER applying the above flags
def get_Lscale(Lum, delta_mag):
    return Lum * 10**(0.4*delta_mag)


#calculate Mscale
#be sure to only run this function AFTER applying the flags above!
#that is, delta_mag should be freshly generated with the mag array AFTER applying the above flags
def get_Mscale(Mstar, delta_mag):
    return Mstar * 10**(0.4*delta_mag)


#determine the mass completeness limit!
#percentile can range from 0.90 to 0.95
#if there is a masked array involved, then apply to remove any cells without entries
def get_mass_limit(Mscale, percentile):
    try:
        #the ~M_scale.mask will remove any entries with no values
        Mscale = np.sort(Mscale[~Mscale.mask])
    except:
        #if not masked array, proceed as usual
        Mscale = np.sort(Mscale)
    
    #find percentile index
    index = int(len(Mscale)*percentile)
    
    #isolate Mscale corresponding to the index above
    Mlimit = np.log10(Mscale[index])
    
    print(f'Number of scaled galaxies from which to select mass limit: {len(Mscale)}')
    print(f'Calculated a log(Mscale) limit of {Mlimit:.2f}')
    return Mlimit


#create hexbin SFR vs. Mstar plot of sample
def plot_sfrmstar(logSFR, logMstar, Mlimit, hexbin=True, nbins=200):

    #hexbin version of log(SFR) vs. log(Mstar) plot

    plt.figure(figsize=(5,4))
    
    if hexbin:
        plt.hexbin(logMstar,logSFR,cmap='Greens',label='All Galaxies',gridsize=nbins,bins='log')
    else:
        plt.scatter(logMstar,logSFR,color='green',alpha=0.2,s=10,label='All Galaxies')
    
    plt.xlabel(r'log(Mstar / [M$_\odot$])',fontsize=12)
    plt.ylabel(r'log(SFR / [M$_\odot$ yr$^{-1}$])',fontsize=12)

    plt.axvline(Mlimit,color='red',alpha=0.4,ls='--',label=f'Mass Completeness = {Mlimit:.2f}')

    plt.xlim(8,12)
    plt.ylim(-4,4)

    plt.legend(fontsize=8)
    #plt.title(r'NED-LVS Galaxies [using m$_J$ = 16.6 limit]',fontsize=12)

    plt.show()
    
    
#all together now...
def mass_completeness(mag, mag_lim, z, z_max, Mstar, percentile, sfr=None, plot=False, hexbin=False, nbins=200):
    
    #get_zflag, get_magflag
    z_flag = get_zflag(z, z_max)
    mag_flag = get_magflag(mag, mag_lim)
    
    flags = z_flag & mag_flag
        
    #define cut versions of apparent magnitude stellar mass arrays
    mag_cut = mag[flags]
    Mstar_cut = Mstar[flags]
    
    #redefine delta_mag with the apparent magnitude cut
    delta_mag = mag_cut - mag_lim
    
    #define the Mscale array
    Mscale = get_Mscale(Mstar_cut, delta_mag)
    
    #compute the mass completeness limit
    Mlimit = get_mass_limit(Mscale, percentile)
    
    #if user defined plot=True, then generate hexbin plot!
    if plot:
        logSFR = np.log10(sfr)
        logMstar = np.log10(Mstar)
        if hexbin:
            plot_sfrmstar(logSFR, logMstar, Mlimit, nbins=nbins)
            return
        plot_sfrmstar(logSFR, logMstar, Mlimit, hexbin=False)
    
##################################################################
#CREATE NED-LVS PARENT SAMPLE
#ADDS RA, DEC, Z, OBJID, AND VARIOUS FLAGS
    #SFR_flag, WISESize_flag, SNR_flag, Mstar_flag, sSFR_flag
##################################################################


import numpy as np
from astropy.table import Table

import os
homedir=os.getenv("HOME")

import sys
sys.path.append(homedir+'/github/wisesize/parent_sample/')
from mass_completeness import *
from sfr_completeness import *


def create_parent(wisesize_table, nedlvs_table, luminosity_table, version=1):
    
    save_path=homedir+f'/Desktop/wisesize/nedlvs_parent_v{version}.fits'

    #convert objnames from tempel catalog to a set
    wisesize_names = set(wisesize_table['OBJNAME'])

    #create a boolean mask for whether each name in the parent table is in the nedlvs-tempel2017 table
    wisesize_flag = [name in wisesize_names for name in nedlvs_table['OBJNAME']]

    #create a set of {objname: objid}
    name_to_id = {
        objname: objid
        for objname, objid in zip(wisesize_names, wisesize_table['OBJID'])
    }

    #apply mapping to nedlvs_table, defaulting to '--'
    objids = [
        name_to_id.get(str(name), '--')
        for name in nedlvs_table['OBJNAME']
    ]


    #create SNR flag!
    snr_nuv_flag = (luminosity_table['Lum_NUV'] / luminosity_table['Lum_NUV_unc']) > 20.
    snr_w3_flag = (luminosity_table['Lum_W3'] / luminosity_table['Lum_W3_unc']) > 20.
    snr_combined_flag = (snr_nuv_flag.data) | (snr_w3_flag.data)
    
    
    #isolating all objects with a "galaxy" objtype
    t_gal = luminosity_table[luminosity_table['objtype']=='G']
    
    #create Mstar (all) flag!
    #this will correspond to the Mstar limit for ALL NED-LVS galaxies
    mag_lim = 16.6
    mag = t_gal['m_J']

    z=t_gal['z']
    z_max=0.025

    mstar = t_gal['Mstar']
    sfr = t_gal['SFR_hybrid']

    percentile = 0.95
    
    #returns a number
    Mstar_limit = mass_completeness(mag, mag_lim, z, z_max, mstar, percentile, plot=False)
    Mstar_flag = np.log10(luminosity_table['Mstar'])>Mstar_limit
    
    
    #create Mstar (WISESize) flag!
    #this will correspond to the Mstar limit for all WISESize galaxies -- need for GALFIT, etc.!
    
    #isolating all objects with a "galaxy" objtype *and* have WISESize flag
    t_gal_size = luminosity_table[(luminosity_table['objtype']=='G') & (wisesize_flag)]
    
    mag_lim = 16.6
    mag = t_gal_size['m_J']

    z=t_gal_size['z']
    z_max=0.025

    mstar = t_gal_size['Mstar']
    sfr = t_gal_size['SFR_hybrid']

    percentile = 0.95
    
    #returns a number
    Mstar_size_limit = mass_completeness(mag, mag_lim, z, z_max, mstar, percentile, plot=False)
    Mstar_size_flag = np.log10(luminosity_table['Mstar'])>Mstar_limit
    
    
    #create SFR flag
    #returns a number
    SFR_limit = sfr_completeness(nedlvs_table['Z'][snr_combined_flag],
                                 luminosity_table['SFR_hybrid'][snr_combined_flag], percentile, plot=False)
    
    SFR_flag = np.log10(luminosity_table['SFR_hybrid'])>SFR_limit
    
    
    #create sSFR flag
    sSFR_limit = -11.5
    sSFR_flag = np.log10(luminosity_table['SFR_hybrid']/luminosity_table['Mstar'])>sSFR_limit
    
    
    parent_table = Table([nedlvs_table['OBJNAME'],objids,nedlvs_table['RA'],
                              nedlvs_table['DEC'],luminosity_table['z'],nedlvs_table['OBJTYPE'],
                              wisesize_flag,snr_nuv_flag,snr_w3_flag,snr_combined_flag,
                              Mstar_flag,SFR_flag,sSFR_flag,Mstar_size_flag],
                          names=['OBJNAME','OBJID','RA','DEC','Z','OBJTYPE','WISESize_flag','SNR_NUV_flag',
                              'SNR_W3_flag','SNR_flag','Mstar_all_flag','SFR_flag','sSFR_flag','Mstar_size_flag'])

    #parent_table.sort('OBJID')
    

    parent_table.write(save_path,overwrite=True)
    print(f'Table saved to {save_path}')
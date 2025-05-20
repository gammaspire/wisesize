'''
if galaxy in tempel catalogs, extract environment info
if not (ALL environment flags are False), find nearest galaxy in RA-DEC-Z space and assign environment
'''

import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
%matplotlib inline

import os

homedir=os.getenv("HOME")


#create a flag for the nedlvs-parent catalog that indicates whether the galaxy is in the Tempel+2017 group/cluster catalog.
#nedlvs_parent --> 1.8 million galaxy catalog
#nedlvs_tempel2017 --> cross-match of NED-LVS and Tempel+2017 catalog (using TOPCAT)
def create_tempel2017_flag(nedlvs_parent, nedlvs_tempel2017):

    #isolate groupID and ngal columns from the cross-matched table
    groupid_all = nedlvs_tempel2017['GroupID']
    ngal_all = nedlvs_tempel2017['Ngal']

    #convert objnames from tempel catalog to a set
    tempel_names = set(name.strip().lower() for name in nedlvs_tempel2017['OBJNAME'])

    #create a boolean mask for whether each name in the parent table is in the nedlvs-tempel2017 table
    tempel2017_flag = [name.strip().lower() in tempel_names for name in nedlvs_parent['OBJNAME']]

    #add the column which indicates whether a galaxy appears in the Tempel+2017 catalog
    nedlvs_parent['tempel2017_flag'] = tempel2017_flag
    
    return nedlvs_parent
    

#create a flag for the nedlvs-parent catalog that indicates whether the galaxy is in the Tempel+2014 filament catalog.
#nedlvs_parent --> 1.8 million galaxy catalog
#nedlvs_tempel2047 --> cross-match of NED-LVS and Tempel+2014 catalog (using TOPCAT)
def create_tempel2014_flag(nedlvs_parent, nedlvs_tempel2014):
    
    #convert objnames from tempel catalog to a set
    tempel_names = set(name.strip().lower() for name in nedlvs_tempel2014['OBJNAME'])

    #create a boolean mask for whether each name in the parent table is in the nedlvs-tempel2014 table
    tempel2014_flag = [name.strip().lower() in tempel_names for name in nedlvs_parent['OBJNAME']]

    #add the column which indicates whether a galaxy appears in the Tempel+2014 catalog
    nedlvs_parent['tempel2014_flag'] = tempel2014_flag
    
    return nedlvs_parent
    

def tempel2017_pg_flag(nedlvs_parent, nedlvs_tempel2017):
    
    groupid_all = nedlvs_tempel2017['GroupID']
    ngal_all = nedlvs_tempel2017['Ngal']
    
    #isolates groups with 2-4 members, inclusive
    tempel2017_pg_flag = (groupid_all!=0) & (ngal_all<5)
    
    #build a "lookup" dictionary: {OBJNAME: pg_flag}
    pg_dict = {
        name.strip().lower(): pg_flag
        for name, pg_flag in zip(nedlvs_tempel2017['OBJNAME'], tempel2017_pg_flag)
    }

    #create the pg flag column! any galaxies with no objname in this dict will receive a default False entry
    pg_flag_column = [
        pg_dict.get(name.strip().lower(), False)
        for name in nedlvs_parent['OBJNAME']
    ]
    
    nedlvs_parent['tempel2017_poorgroup_flag'] = pg_flag_column
    
    return nedlvs_parent


#create flags for Tempel+2017 groups -- rich group or cluster!
#that is, flags are row-matched to Tempel+2017 groups/clusters to indicate whether group is cluster or rich group
def tempel2017_group_flags(tempel2017_groups):
    rg_flag = np.zeros(len(tempel2017_groups),dtype=bool)
    cluster_flag = np.zeros(len(tempel2017_groups),dtype=bool)

    #note: n begins at 1, but index should begin at 0 (so I use [n-1] for indexing)
    for n in tempel2017_groups['GroupID']:

        #calculate M200
        M200 = tempel2017_groups[n-1]['M200']*1e12   #Msol

        #find number of members in the group
        ngal = tempel2017_groups[n-1]['Ngal']

        rg_flag[n-1] = (M200<1e14) & (ngal>=5)
        cluster_flag[n-1] = (M200>=1e14) & (ngal>=5)
        
    return rg_flag, cluster_flag


#now create rich group ($\geq 5$ galaxies) and cluster ($M_{200}>10^{14} M_{\odot}$) flags.
def tempel2017_rg_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups):
    
    groupid_all = nedlvs_tempel2017['GroupID']
    ngal_all = nedlvs_tempel2017['Ngal']
    
    rg_flag, _ = tempel2017_group_flags(tempel2017_groups)
    
    #I now have flags for whether the groups are rich groups or clusters, row-matched to the GroupID column
    #now I match galaxies to these group flags...I suppose.

    #build a "lookup" dictionary: {GroupID: rg_flag}
    rg_dict = {
        groupid: rg_flag
        for groupid, rg_flag in zip(tempel2017_groups['GroupID'], rg_flag)
    }
    
    #create the rg flag column for tempel2017 galaxies!
    tempel2017_rg_flag = [
        rg_dict.get(group_id, False)
        for group_id in nedlvs_tempel2017['GroupID']
    ]
    
    #now...CREATE NED-LVS COLUMN as before -- match NED-LVS parent galaxies with these Tempel+2017 rich group flags
    #build a "lookup" dictionary: {objname: rg_flag}
    rg_dict = {
        name.strip().lower(): rg_flag
        for name, rg_flag in zip(nedlvs_tempel2017['OBJNAME'], tempel2017_rg_flag)
    }
    
    rg_flag_column = [
        rg_dict.get(name.strip().lower(), False)
        for name in nedlvs_parent['OBJNAME']
    ]

    nedlvs_parent['tempel2017_richgroup_flag'] = rg_flag_column
    
    return nedlvs_parent
    
    
def tempel2017_cluster_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups):

    groupid_all = nedlvs_tempel2017['GroupID']
    ngal_all = nedlvs_tempel2017['Ngal']
    
    _, cluster_flag = tempel2017_group_flags(tempel2017_groups)
    
    cluster_dict = {
        groupid: cluster_flag
        for groupid, cluster_flag in zip(tempel2017_groups['GroupID'], cluster_flag)
    }

    #create the cluster flag column for tempel2017 galaxies!
    tempel2017_cluster_flag = [
        cluster_dict.get(group_id, False)
        for group_id in nedlvs_tempel2017['GroupID']
    ]

    #and now...create the nedlvs columns
    cluster_dict = {
        name.strip().lower(): cluster_flag
        for name, cluster_flag in zip(nedlvs_tempel2017['OBJNAME'], tempel2017_cluster_flag)
    }

    cluster_flag_column = [
        cluster_dict.get(name.strip().lower(), False)
        for name in nedlvs_parent['OBJNAME']
    ]

    #add the column
    nedlvs_parent['tempel2017_cluster_flag'] = cluster_flag_column

    return nedlvs_parent
    

#create filament flags from Tempel+2014
# * near filament: $\leq 0.5$ h$^{-1}$ Mpc
# * far filament: $0.5 <$ distance $\leq1.0$ h$^{-1}$ Mpc
def tempel2014_filament_flags(nedlvs_parent, nedlvs_tempel2014):    

    tempel2014_nearfil_flag = (nedlvs_tempel2014['fil_dist']<=0.5)
    tempel2014_farfil_flag = (nedlvs_tempel2014['fil_dist']>0.5) & (nedlvs_tempel2014['fil_dist']<=1.0)

    #build a "lookup" dictionary (objname: fil_flag)
    nearfil_dict = {
        name.strip().lower(): filnear_flag
        for name, filnear_flag in zip(nedlvs_tempel2014['OBJNAME'], tempel2014_nearfil_flag)
    }
    farfil_dict = {
        name.strip().lower(): filfar_flag
        for name, filfar_flag in zip(nedlvs_tempel2014['OBJNAME'], tempel2014_farfil_flag)
    }

    #create the flag columns!
    nearfil_flag_column = [
        nearfil_dict.get(name.strip().lower(), False)
        for name in nedlvs_parent['OBJNAME']
    ]

    farfil_flag_column = [
        farfil_dict.get(name.strip().lower(), False)
        for name in nedlvs_parent['OBJNAME']
    ]

    #add the columns
    nedlvs_parent['tempel2014_nearfilament_flag'] = nearfil_flag_column
    nedlvs_parent['tempel2014_farfilament_flag'] = farfil_flag_column
    
    
#lastly, field galaxies...those in Tempel+2014 or Tempel+2017 but not part of ANY of the above environments.
#REQUIRES NED-LVS PARENT CATALOG WITH TEMPEL NON-FIELD ENVIRONMENT COLUMNS ALREADY POPULATED
def tempel_field_flag(nedlvs_parent):
    
    #isolate galaxies which have BOTH a True tempel2017_flag and tempel2014_flag
    tempel_flag = (nedlvs_parent['tempel2014_flag']) | (nedlvs_parent['tempel2017_flag'])
    
    #isolate galaxies which are not members of ANY of the non-field environments
    env_flag = (~nedlvs_parent['tempel2017_poorgroup_flag']) & \
           (~nedlvs_parent['tempel2017_richgroup_flag']) & \
           (~nedlvs_parent['tempel2017_cluster_flag']) & \
           (~nedlvs_parent['tempel2014_nearfilament_flag']) & \
           (~nedlvs_parent['tempel2014_farfilament_flag'])
    
    #define the field flag
    field_flag = (tempel_flag) & (env_flag)
    
    #add the field flag
    nedlvs_parent['tempel_field_flag'] = field_flag
    
    return nedlvs_parent

def write_nedlvs_parent(nedlvs_parent, path, version_integer=1):
    
    name = f'nedlvs_parent_v{version_integer}.fits'
    nedlvs_parent.write(path+f'wisesize/{name}',overwrite=True)
    
    
if __name__ == "__main__":

    #LOAD TABLES
    path = homedir+'/Desktop/'
    
    nedlvs_parent = Table.read(path+'wisesize/nedlvs_parent_v1.fits')

    nedlvs_tempel2014 = Table.read(path+'wisesize/nedlvs_tempel2014.fits')
    nedlvs_tempel2017 = Table.read(path+'wisesize/nedlvs_tempel2017.fits')

    #need this table for M200 (halo mass of group/cluster galaxies) and Ngal (# galaxies in group)
    tempel2017_groups = Table.read(path+'tempel2017b.fits')

    print("""USAGE:
    ---create_tempel2017_flag(nedlvs_parent, nedlvs_tempel2017)
    ---create_tempel2014_flag(nedlvs_parent, nedlvs_tempel2014)
    ---tempel2017_pg_flag(nedlvs_parent, nedlvs_tempel2017)
    ---tempel2017_rg_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups)
    ---tempel2017_cluster_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups)
    ---tempel2014_filament_flags(nedlvs_parent, nedlvs_tempel2014)
    ---tempel_field_flag(nedlvs_parent)
    ---write_nedlvs_parent(nedlvs_parent, path, version_integer=1)
    
    
    """)
    
    print('-----------------------------------------------------')


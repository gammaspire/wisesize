'''
to-do: if ALL environment flags are False, find nearest galaxy in RA-DEC-Z space and assign environment
might have to do that KDTree thing again. see kNN-local_density.py code
'''

import numpy as np
from astropy.table import Table
import os

from scipy.spatial import KDTree

homedir=os.getenv("HOME")


#create a flag for the nedlvs-parent catalog that indicates whether the galaxy is in the Tempel+2017 group/cluster catalog.
#nedlvs_parent --> 1.8 million galaxy catalog
#nedlvs_tempel2017 --> cross-match of NED-LVS and Tempel+2017 catalog (using TOPCAT)
def create_tempel2017_flag(nedlvs_parent, nedlvs_tempel2017):

    #convert objnames from tempel catalog to a set
    tempel_names = set(str(name).strip().lower() for name in nedlvs_tempel2017['OBJNAME'])

    tempel_groupID = set(int(groupid) for groupid in nedlvs_tempel2017['GroupID'])
    
    #create a boolean mask for whether each name in the parent table is in the nedlvs-tempel2017 table
    tempel2017_flag = [
        str(name).strip().lower() in tempel_names
        for name in nedlvs_parent['OBJNAME']
    ]
    
    #create a set of {objname: groupID}
    name_to_groupid = {
        str(name).strip().lower(): int(groupid)
        for name, groupid in zip(nedlvs_tempel2017['OBJNAME'], nedlvs_tempel2017['GroupID'])
    }
    
    #do the same for Ngal
    #create a set of {objname: groupID}
    name_to_ngal = {
        str(name).strip().lower(): int(ngal)
        for name, ngal in zip(nedlvs_tempel2017['OBJNAME'], nedlvs_tempel2017['Ngal'])
    }

    #apply mapping to nedlvs-parent, defaulting to -99
    group_ids = [
        name_to_groupid.get(str(name).strip().lower(), -99)
        for name in nedlvs_parent['OBJNAME']
    ]
    ngal = [
        name_to_ngal.get(str(name).strip().lower(), -99)
        for name in nedlvs_parent['OBJNAME']
    ]

    return tempel2017_flag, group_ids, ngal
    

#create a flag for the nedlvs-parent catalog that indicates whether the galaxy is in the Tempel+2014 filament catalog.
#nedlvs_parent --> 1.8 million galaxy catalog
#nedlvs_tempel2017 --> cross-match of NED-LVS and Tempel+2014 catalog (using TOPCAT)
def create_tempel2014_flag(nedlvs_parent, nedlvs_tempel2014):
    
    #convert objnames from tempel catalog to a set
    tempel_names = set(name.strip().lower() for name in nedlvs_tempel2014['OBJNAME'])

    #create a boolean mask for whether each name in the parent table is in the nedlvs-tempel2014 table
    tempel2014_flag = [name.strip().lower() in tempel_names for name in nedlvs_parent['OBJNAME']]
    
    return tempel2014_flag


def create_kt2017_flag(nedlvs_parent, nedlvs_kt2017):
    
    #convert objnames from KT catalog to a set
    kt_names = set(name.strip().lower() for name in nedlvs_kt2017['OBJNAME'])

    #create a boolean mask for whether each name in the parent table is in the nedlvs-tempel2014 table
    kt2017_flag = [name.strip().lower() in kt_names for name in nedlvs_parent['OBJNAME']]
    
    return kt2017_flag


#create flags for Tempel+2017 groups -- group or cluster!
#that is, flags are row-matched to Tempel+2017 groups/clusters to indicate whether group is cluster or group
def tempel2017_group_flags(tempel2017_groups):
    
    group_flag = np.zeros(len(tempel2017_groups),dtype=bool)
    cluster_flag = np.zeros(len(tempel2017_groups),dtype=bool)

    #note: n begins at 1, but index should begin at 0 (so I use [n-1] for indexing)
    for n in tempel2017_groups['GroupID']:

        #calculate M200
        M200 = tempel2017_groups[n-1]['M200']*1e12   #Msol

        #find number of members in the group
        ngal = tempel2017_groups[n-1]['Ngal']
        
        group_flag[n-1] = (M200<1e14) & (ngal>=5)
        cluster_flag[n-1] = (M200>=1e14)
        
    return group_flag, cluster_flag


#use to output either group or cluster flag array, row-matched to NED-LVS parent catalog
def tempel2017_gc_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups, 
                         cluster=False, group=False):
    
    groupid_all = nedlvs_tempel2017['GroupID']
    
    if group:
        group_flag, _ = tempel2017_group_flags(tempel2017_groups)
    if cluster:
        _, group_flag = tempel2017_group_flags(tempel2017_groups)
    
    #I now have flags for whether the "groups" are groups or clusters, row-matched to the GroupID column
    #now I match galaxies to these group flags...

    #build a "lookup" dictionary: {GroupID: group_flag}
    group_dict = {
        groupid: group_flag
        for groupid, group_flag in zip(tempel2017_groups['GroupID'], group_flag)
    }
    
    #create the group flag column for tempel2017 galaxies!
    tempel2017_flag = [
        group_dict.get(group_id, False)
        for group_id in nedlvs_tempel2017['GroupID']
    ]
    
    #now...CREATE NED-LVS COLUMN as before -- match NED-LVS parent galaxies with these Tempel+2017 group flags
    #build a "lookup" dictionary: {objname: group_flag}
    group_dict_nedlvs = {
        name.strip().lower(): group_flag
        for name, group_flag in zip(nedlvs_tempel2017['OBJNAME'], tempel2017_flag)
    }
    
    #row-match flags to NED-LVS parent catalog
    group_flag_column = [
        group_dict_nedlvs.get(name.strip().lower(), False)
        for name in nedlvs_parent['OBJNAME']
    ]
    
    return group_flag_column
    

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
    
    return nearfil_flag_column, farfil_flag_column
    
    
#lastly, field galaxies...those in Tempel+2014 or Tempel+2017 but not part of ANY of the above environments.
#REQUIRES NED-LVS PARENT CATALOG WITH TEMPEL NON-FIELD ENVIRONMENT COLUMNS ALREADY POPULATED
def tempel_field_flag(nedlvs_parent):
    
    #isolate galaxies which have BOTH a True tempel2017_flag and tempel2014_flag
    tempel_flag = (nedlvs_parent['tempel2014_flag']) | (nedlvs_parent['tempel2017_flag'])
    
    #isolate galaxies which are not members of ANY of the non-field environments
    env_flag = (~nedlvs_parent['tempel2017_group_flag']) & \
           (~nedlvs_parent['tempel2017_cluster_flag']) & \
           (~nedlvs_parent['tempel2014_nearfilament_flag']) & \
           (~nedlvs_parent['tempel2014_farfilament_flag'])
    
    #define the field flag
    field_flag = (tempel_flag) & (env_flag)
    
    return field_flag


def KT2017_group_flags(kt2017_groups):
    
    pg_flag = np.zeros(len(kt2017_groups),dtype=bool)
    rg_flag = np.zeros(len(kt2017_groups),dtype=bool)
    
    for n, ngal in enumerate(kt2017_groups['Nm']):
        pg_flag[n] = (ngal>=2) & (ngal<5)
        rg_flag[n] = (ngal>=5)
        
    return pg_flag, rg_flag


#for whatever reason, PGC1 is the name of the groupid column
def KT2017_rpg_flag(nedlvs_parent, nedlvs_kt2017, kt2017_groups, rich=False, poor=False):

    groupid_all = nedlvs_kt2017['PGC1']
    
    if rich:
        _, group_flag = KT2017_group_flags(kt2017_groups)
    if poor:
        group_flag, _ = KT2017_group_flags(kt2017_groups)
    
    group_dict = {
        groupid: group_flag
        for groupid, group_flag in zip(kt2017_groups['PGC1'], group_flag)
    }

    #create the flag column for kourkchi+tully 2017 galaxies! pairs groupid of galaxy with T/F for group membership
    kt2017_flag = [
        group_dict.get(group_id, False)
        for group_id in nedlvs_kt2017['PGC1']
    ]

    #and now...create the nedlvs columns. this pairs the NED-LVS objname with the group flag
    group_dict_nedlvs = {
        name.strip().lower(): group_flag
        for name, group_flag in zip(nedlvs_kt2017['OBJNAME'], kt2017_flag)
    }

    #creates row-matched rg flag for full NED-LVS (not just the galaxies cross-matched with KT+2017)
    group_flag_column = [
        group_dict_nedlvs.get(name.strip().lower(), False)
        for name in nedlvs_parent['OBJNAME']
    ]

    return group_flag_column


#for every relevant environment column in nedlvs_parent, assign the galaxy in question with the
#cell values of the nearest Tempel galaxy in RA-DEC-redshift space
def match_nontempel_galaxies(nedlvs_parent):
    
    #create flag for galaxies in either Tempel+2017 OR Tempel+2014 catalogs
    tempel_flags = (nedlvs_parent['tempel2014_flag']) | (nedlvs_parent['tempel2017_flag'])

    #determine the tempel indices...that is, where in nedlvs_parent the Tempel galaxies lie
    #will be the same length as nedlvs_parent[tempel_flags] but with nedlvs_parent indices :-)
    tempel_indices = np.where(tempel_flags)[0]

    #do the same for nontempel_indices. crucial!
    nontempel_indices = np.where(~tempel_flags)[0]

    ra = nedlvs_parent['RA']
    dec = nedlvs_parent['DEC']
    z = nedlvs_parent['Z']

    #create 3D KDTree (just as with 5NN code)
    coords = np.vstack((ra,dec,z)).T

    tempel_tree = KDTree(coords[tempel_flags])

    #for non-Tempel galaxies galaxy:
        # find nearest galaxy that IS in the Tempel catalogs (using tempel_tree)
        # assign to that galaxy that nearest galaxy's environment flags

    #isolate parent columns
    nedlvs_envcols = [nedlvs_parent['tempel2017_groupIDs'], nedlvs_parent['tempel2017_group_flag'],
                      nedlvs_parent['tempel2017_cluster_flag'], nedlvs_parent['tempel2014_nearfilament_flag'],
                      nedlvs_parent['tempel2014_farfilament_flag'], nedlvs_parent['tempel_field_flag']]

    #for every set of non-Tempel coordinates, find nearest TEMPEL GALAXY INDEX. 
    #I then have to map that index onto the nedlvs_parent indices!
    for i, point in enumerate(coords[~tempel_flags]):

        central_index = nontempel_indices[i]

        #outputs nearest index, which I aptly name "index"
        #HOWEVER, this index corresponds to the index of the TEMPEL GALAXIES
        _, index = tempel_tree.query(point)

        index = tempel_indices[index]   #grab nedlvs_parent index of Tempel galaxy

        #for every relevant environment column in nedlvs_parent, assign the galaxy in question with the
        #cell values of the nearest Tempel galaxy in RA-DEC-redshift space
        for col in nedlvs_envcols:
            col[central_index] = col[index]
    
    #return the updated nedlvs_parent table!
    return nedlvs_parent


#COMBINE THEM ALL. ALL TOGETHER NOW!
def add_tempel_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups, nedlvs_kt2017, kt2017_groups):
    
    tempel2017_flag, groupIDs, ngal = create_tempel2017_flag(nedlvs_parent, nedlvs_tempel2017)
    tempel2014_flag = create_tempel2014_flag(nedlvs_parent, nedlvs_tempel2014)
    
    tempel_group_flag = tempel2017_gc_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups, group=True)
    tempel_cluster_flag = tempel2017_gc_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups, cluster=True)
    
    nearfil_flag, farfil_flag = tempel2014_filament_flags(nedlvs_parent, nedlvs_tempel2014)
    
    kt2017_flag = create_kt2017_flag(nedlvs_parent, nedlvs_kt2017)
    
    KT_pg_flag = KT2017_rpg_flag(nedlvs_parent, nedlvs_kt2017, kt2017_groups, poor=True)
    KT_rg_flag = KT2017_rpg_flag(nedlvs_parent, nedlvs_kt2017, kt2017_groups, rich=True)

    flags = [tempel2014_flag, tempel2017_flag, groupIDs, ngal, tempel_group_flag, tempel_cluster_flag,
            nearfil_flag, farfil_flag, kt2017_flag, KT_pg_flag, KT_rg_flag]
    names = ['tempel2014_flag', 'tempel2017_flag', 'tempel2017_groupIDs', 'tempel2017_Ngal', 
             'tempel2017_group_flag','tempel2017_cluster_flag', 'tempel2014_nearfilament_flag', 
             'tempel2014_farfilament_flag', 'KT2017_flag', 'KT2017_pg_flag', 'KT2017_rg_flag']
    
    for n in range(len(flags)):
        nedlvs_parent[names[n]] = flags[n]
    
    field_flag = tempel_field_flag(nedlvs_parent)
    nedlvs_parent['tempel_field_flag'] = field_flag
    
    return nedlvs_parent


#ALL TOGETHER NOW, BUT FOR REAL.
def add_all_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups, nedlvs_kt2017, kt2017_groups):
    
    nedlvs_parent_tempel = add_tempel_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups, nedlvs_kt2017, kt2017_groups)
    
    nedlvs_parent_all = match_nontempel_galaxies(nedlvs_parent_tempel)
    
    return nedlvs_parent_all


def write_nedlvs_parent(nedlvs_parent, path, version_integer=1):
    
    name = f'nedlvs_parent_v{version_integer}.fits'
    nedlvs_parent.write(path+f'wisesize/{name}',overwrite=True)
    
    
if __name__ == "__main__":

    #LOAD TABLES
    path = homedir+'/Desktop/'
    
    nedlvs_parent = Table.read(path+'wisesize/nedlvs_parent_v1.fits')

    import warnings
    warnings.filterwarnings('ignore')
    
    nedlvs_tempel2014 = Table.read(path+'wisesize/nedlvs_tempel2014.fits')
    nedlvs_tempel2017 = Table.read(path+'wisesize/nedlvs_tempel2017.fits')

    #need this table for M200 (halo mass of group/cluster galaxies) and Ngal (# galaxies in group)
    tempel2017_groups = Table.read(path+'tempel2017b.fits')
    
    #Kourkchi+Tully (2017) group galaxy catalog cross-matched with NED-LVS
    nedlvs_kt2017 = Table.read(path+'wisesize/nedlvs_KT2017.fits')
    
    #need THIS table for groupIDs and Ngal in each group
    kt2017_groups = Table.read(path+'KT2017_tabl1.fits')

    print("""USAGE:
    * create_tempel2017_flag(nedlvs_parent, nedlvs_tempel2017) 
         -- outputs flag and two arrays (NED-LVS galaxies in Tempel+2017, groupIDs, Ngal)
    * create_tempel2014_flag(nedlvs_parent, nedlvs_tempel2014) 
         -- outputs flag (NED-LVS galaxies in Tempel+2014)
    * create_kt2017_flag(nedlvs_parent, nedlvs_kt2017)
         -- outputs flag (NED-LVS galaxies in Kourkchi+Tully 2017)
    * tempel2017_gc_flag(nedlvs_parent, nedlvs_tempel2017, tempel2017_groups, cluster=False, group=False)
         -- outputs flag for NED-LVS galaxies in Tempel+2017 clusters, or groups
    * tempel2014_filament_flags(nedlvs_parent, nedlvs_tempel2014)
         -- outputs two flags for NED-LVS galaxies near (<0.5 h^-1 Mpc) or far from a filament (0.5 < dist < 1.0 h^-1 Mpc)
    * tempel_field_flag(nedlvs_parent)
         -- outputs flag --> galaxies in Tempel catalogs but NOT in any of the environments are True; else, False.
    * KT2017_rpg_flag(nedlvs_parent, nedlvs_kt2017, kt2017_groups, rich=False, poor=False)
         -- outputs flag for NED-LVS galaxies in Kourkchi+Tully (2017) rich groups (rich=True) or poor groups (poor=True)
    * add_tempel_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups, nedlvs_kt2017, kt2017_groups)         
         -- outputs updated nedlvs_parent table. TEMPEL GALAXY FLAGS ONLY! All other galaxies will default to "False"
    * match_nontempel_galaxies(nedlvs_parent)
         -- outputs updated nedlvs_parent with non-Tempel galaxies adopting the env flags of their nearest Tempel neighbor
    * add_all_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups, nedlvs_kt2017, kt2017_groups)
         -- outputs same as add_tempel_flags() BUT with the addition of match_nontempel_galaxies()
    * write_nedlvs_parent(nedlvs_parent, path_to_folder, version_integer=1)
         -- saves table path_to_folder/nedlvs_parent_v{version_integer}.fits
    """)
    
    print('-----------------------------------------------------')


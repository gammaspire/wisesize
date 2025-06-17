import numpy as np
from astropy.table import Table

import os
homedir=os.getenv("HOME")

def create_parent(wisesize_table, nedlvs_table, version=1):
    
    save_path=homedir+f'/Desktop/wisesize/nedlvs_parent_v{version}.fits'

    #convert objnames from tempel catalog to a set
    wisesize_names = set(wisesize_table['OBJNAME'])

    #create a boolean mask for whether each name in the parent table is in the nedlvs-tempel2017 table
    flag = [name in wisesize_names for name in nedlvs_table['OBJNAME']]

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


    parent_table = Table([nedlvs_table['OBJNAME'],objids,nedlvs_table['RA'],
                              nedlvs_table['DEC'],nedlvs_table['Z'],nedlvs_table['OBJTYPE'],
                              flag],
                               names=['OBJNAME','OBJID','RA','DEC','Z','OBJTYPE','WISESize_flag'])

    parent_table.sort('OBJID')


    parent_table.write(save_path,overwrite=True)
    print(f'Table saved to {save_path}')
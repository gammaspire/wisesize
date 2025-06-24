# WISESize Project
Various scripts for the analysis of nearby galaxies in the local Universe, primarily using GALEX UV, Legacy Survey grz, and WISE infrared bands.

## How to Create nedlvs_parent_v{#}.fits Catalog

### Generate the Scaffold
    * Download full NED-LVS catalog
    * Pull WISESize sample catalog from draco (after merging the north and south catalogs using that .py script I wrote for mucho-galfit (wisesize branch))
    * With these two tables, 
        * In either a python shell or Jupyter Notebook, 
        
        ```
        import sys
        sys.path.append(homedir+'/github/wisesize/parent_sample/')
        from create_parent_sample import create_parent
        ```
       
        * Define the two tables,
        
        ```
        wisesize_table = Table.read(homedir+'/Desktop/wisesize/wisesize_v4.fits')
        nedlvs_table = Table.read(homedir+'/Desktop/wisesize/NEDbyname-NEDLVS_20210922_v2.fits')
        ```
        
        * Then...run
        
        ```
        create_parent(wisesize_table, nedlvs_table, version=1)
        ```
       
        * The output will be saved as `/Users/k215c316/Desktop/wisesize/nedlvs_parent_v1.fits`
        * Row-matched to full NED-LVS, contains WISESize_flag, SNR_flag, OBJID, RA, DEC, Z, Mstar_flag, SFR_flag, sSFR_flag
            * NOTE: WISESize_flag does NOT include the SNR_flag

### Add Mass Completeness, SFR Completeness, and sSFR Flags

Companion notebooks located in `$wisesize/parent_sample/` but are also run as part of the Scaffold table in the above step

### Add Tempel Environment Flags 
    * Full instructions with commentary given in `~/github/wisesize/tempel_catalogs.ipynb`
    * Alternatively, in python script or Jupyter Notebook,
    
    ```
    import sys
    sys.path.append(homedir+'/github/wisesize/environment_classifications/')
    from environment_flags import *
    ```
    
    OR, navigate to the correct directory (e.g., with sys.path.append or os.chdir) and type:
    
    ```
    %run environment_flags.py
    ```
    
    * Define the desired tables for the arguments below OR use the defaults in the .py script (NOTE: code defaults to version=1 for nedlvs_parent! Depending on your preference,
        * append all flag columns to nedlvs_parent, ONLY affects galaxies that are in Tempel+2017 and/or Tempel+2014 (all other galaxies defaulted to False flags in every environment-related column)
       
        ```
        nedlvs_parent_updated = add_tempel_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups, nedlvs_kt2017, kt2017_groups)
        ```
    
        * Does the same as above but with non-Tempel galaxies adopting the env flags of their nearest Tempel neighbor
        
        ```
        nedlvs_parent_updated = add_all_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups,nedlvs_kt2017, kt2017_groups)
        ```
    
    * Save the output table
    
    ```
    write_nedlvs_parent(updated_nedlvs_parent, path_to_folder, version_integer=1)
    ```

### Add Fifth-Nearest-Neighbor Column
#### Note: Only run AFTER mass completeness, SFR completeness, sSFR completeness flags are added to parent catalog
#### - THEN run on parent catalog; flags will automatically be applied

INCOMPLETE:
* Run through kNN_local_density.ipynb to get 2D_5NN (and possibly 2D_3NN)
* Alternatively, in a notebook can run %run kNN_local_density.py -vr_limit 600 -k 5 -write
    * This will output all_kNN array, which will be row-matched to NED-LVS. Add column to Table and save (done automatically if -write is an argument, as above).
        * If -write not part of the argument, can either a block of code akin to the following:
    
        ```
        #read parent catalog
        from astroy.table import Table
        import os
        homedir=os.getenv("HOME")
        cat=Table.read(homedir+'/Desktop/wisesize/nedlvs_parent_v1.fits')

        mstarflag = cat['Mstar_flag']
        sfrflag = cat['SFR_flag']
        ssfrflag = cat['sSFR_flag']
        zflag = cat['Z']>0

        #these are ALL flags applied to the 5NN input table
        flags = (mstarflag) & (sfrflag) & (ssfrflag) & (zflag)

        all_5NN_parent = np.full(len(cat), -999)
        all_5NN_parent[flags] = all_5NN_500

        cat['2D_5NN'] = all_5NN_parent
        cat.write(homedir+'/Desktop/wisesize/nedlvs_parent_v1.fits',overwrite=True)
        ```
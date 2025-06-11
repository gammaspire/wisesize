* How to create nedlvs_parent_v{#}.fits catalog:
    * Download full NED-LVS, pull WISESize sample catalog from draco (after merging the north and south catalogs using that .py script I wrote for mucho-galfit (wisesize branch))
        * This will be row-matched to full NED-LVS catalog
    * Run through ~/github/wisesize/parent_sample.ipynb to get RA, DEC, REDSHIFT, etc., and WISESize sample flag
    * Run through ~/github/wisesize/tempel_catalogs.ipynb, OR just run environment_flags.py and nedlvs_parent = add_all_flags(nedlvs_parent, nedlvs_tempel2014, nedlvs_tempel2017, tempel2017_groups,
                             nedlvs_kt2017, kt2017_groups)
    * Run through kNN_local_density.ipynb to get 2D_5NN (and possibly 2D_3NN)
    * Alternatively, in a notebook can run %run kNN_local_density.py -vr_limit 500 -k 5
        * This will output all_kNN array, which will be row-matched to NED-LVS. Add column to Table and save.
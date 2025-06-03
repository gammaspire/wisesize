### How to create nedlvs_parent_v{#}.fits catalog
#### - download full NED-LVS, pull WISESize sample catalog from draco (after merging the north and south catalogs using that .py script I wrote for mucho-galfit (wisesize branch))
#### - run through ~/github/wisesize/parent_sample.ipynb to get RA, DEC, REDSHIFT, etc., and WISESize sample flag
#### - run through ~/github/wisesize/tempel_catalogs.ipynb, OR just run environment_flags.py to add tempel environment flags
#### - run through kNN_local_density.ipynb to get 2D_5NN (and possibly 2D_3NN)
####   - add columns, write new table
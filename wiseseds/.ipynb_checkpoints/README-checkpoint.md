To run (after navigating to this directory locally):
```
conda activate cigale
python run_cigale_NS.py -params params.txt'
```
Doing so will run CIGALE on both north and south input txt files, with specific parameters according to params.txt. Output (and input) files will default to specific $/Desktop directories, as indicated in params.txt, unless I specify they go elsewhere.

If I desire to edit the parameter ranges written in the pcigale.ini file (which CIGALE interfaces with directly -- the params.txt file simply streamlines the automatic creation of pcigale.ini...for me), then please refer to run_cigale_one.py. Note that the loop is set up such that parameters which none of the modules use will NOT be included in the mature pcigale.ini file...so try not to accidentally edit irrelevant parameter ranges.

The current setup of this code is intended to facilitate the running of CIGALE for a parent sample of galaxies belonging to either the Northern or Southern hemisphere per their (declination) coordinates in RA-DEC space. I have tried to generalize the script as much as possible, but I have tested it on but one set of galaxies (VFS) and some of the labeling may still remain too unique to that set's conventions (e.g., column names). In these cases where naming errors occur, try poking and prodding at gen_input_files.py. This script is also, incidentally, where I can edit the error floors for certain wavelength bands. 

**SED PLOT GENERATION IS A DEFAULT. IF UNDESIRED, EDIT THE run_cigale_ns.py SCRIPT.**
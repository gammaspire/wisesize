#this script assumes that generate_input_files.py is successfully run, and that the user has activated their cigale (conda) environment

import os
homedir = os.getenv("HOME")

import numpy as np
import sys
import re

sys.path.insert(0, 'utils')
from param_utils import Params
from init_utils import add_params

def run_genconf(dir_path):
    os.chdir(dir_path)
    #will generate configuration files pcigale.ini and pcigale.ini.spec
    #can check/edit the various parameters in the file before the next step
    os.system('pcigale genconf')
        
def run_cigale(dir_path):

    os.chdir(dir_path)   #just in case...
    
    #once completed, this line will generate an 'out' directory containing 'results.fits', among other files.
    os.system('pcigale run')
    
def run_sed_plots(dir_path):
    
    os.chdir(dir_path)  #AGAIN, just in case...
    os.system('pcigale-plots sed')
    print(f'Find .pdf plots in {dir_path}/out/')
    
if __name__ == "__main__":

    if '-h' in sys.argv or '--help' in sys.argv:
        print("USAGE: %s [-sed_plots (indicates whether user would like the .pdf plots of each sed-fitted galaxy)] [-params (name of parameter.txt file, no single or double quotations marks)]")
        sys.exit()
    
    if '-sed_plots' in sys.argv:
        sed_plots = True
    else:
        sed_plots = False

    if '-params' in sys.argv:
        p = sys.argv.index('-params')
        param_file = str(sys.argv[p+1])
    else:
        print('-params argument not found. exiting.')
        sys.exit()
    
    #define ALL parameters using Params class (utils/param_utils.py)
    params = Params(param_file)
            
    #in order to save the probability distribution functions, ncores = nblocks = 1
    if params.create_pdfs:
        print('Create PDFs set to True! nblocks = ncores = 1.')
    
    print('Configuring input text files...')
    run_genconf(params.dir_path)

    add_params(params.dir_path, sed_plots, params.lim_flag, params.nblocks, 
               create_pdfs=params.create_pdfs)
    
    print('Executing CIGALE...')
    run_cigale(params.dir_path)
    
    print('CIGALE is Fin!')
    if sed_plots:
        
        print('Generating SED plots...')
        run_sed_plots(params.dir_path)
        
        print('Organizing output...')
        print('Removing SFH .fits files')
        
        os.chdir(params.dir_path+'/out/')
        os.mkdirs('best_SED_models', exist_ok=True)
        
        if sed_plots:
            os.system('rm *best_model*.fits')
        
        os.system('mv *best_model* best_SED_models')
        os.system('rm *_SFH*')
        
        try:
            os.mkdirs('PDF_fits', exist_ok=True)
            
            len_tab = len(params.main_tab)
            
            #4 for VFID (VFS), 5 for OBJID (WISESize)
            if len(str(params.id_col))==4:
                formatted_strings = np.array([f"{num:04}" for num in np.arange(0,len_tab)])
                ID_prefix='VFID'
            if len(str(params.id_col))==5:
                formatted_strings = np.array([f"{num:05}" for num in np.arange(0,len_tab)])
                ID_prefix='OBJID'
            
            #if I don't apply this loop (one mv command per OBJID), then there is a 
            #"too many arguments" error. I do not want a "too many arguments" error.
            for num in formatted_strings:           
                galID = f'{ID_prefix}{num}'
                os.system(f'mv {galID}*fits PDF_fits')
        
        except:
            pass
        
        print('SED Models are Fin!')
        
        
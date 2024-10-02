import sys
import os
from astropy.table import Table, vstack
homedir = os.getenv("HOME")

def run_cigale_all(filler='',sfh_type='sfh2exp'):

    print(f'Using SFH module {sfh_type}!')
    
    os.system(f'python run_cigale_{sfh_type}.py -dir_path ~/Desktop/cigale_vf_north{filler} -sed_plots')
    os.system(f'python run_cigale_{sfh_type}.py -dir_path ~/Desktop/cigale_vf_south{filler} -sed_plots')

def concatenate_ns(destination):
    north_results = Table.read(f'{destination}/out/results.fits')
    south_results = Table.read(f'{destination}/out/results.fits')
    
    combined_tab = vstack([north_results, south_results])    
    combined_tab.write(f'{destination}/results_NS.fits',overwrite=True)

def mv_files(filler, destination):
    
    #create directories if they do not already exist
    if not os.path.isdir(destination):
        os.system(f'mkdir {destination}')
    if not os.path.isdir(f'{destination}/SED_pdfs'):
        os.system(f'mkdir {destination}/SED_pdfs')
    
    #concatenate the north and south VF tables and move to the relevant directory
    concatenate_ns(destination)
    
    #moving PDF files from both cigale directories to the shiny new one.
    os.system(f'mv ~/Desktop/cigale_vf_north{filler}/out/best_SED_models/*.pdf {destination}/SED_pdfs/')
    os.system(f'mv ~/Desktop/cigale_vf_south{filler}/out/best_SED_models/*.pdf {destination}/SED_pdfs/')
    
if __name__ == "__main__":

    if '-h' in sys.argv or '--help' in sys.argv:
        print("USAGE: %s [-herschel (indicates that CIGALE wil run on Herschel galaxies)] %s ['sfh_type (sfh2exp or sfhdelayed)]")
    
    if '-herschel' in sys.argv:
        filler='_tom'
    else:
        filler=''
    
    if 'sfh_type' in sys.argv:
        p = sys.argv.index('-sfh_type')
        sfh_type = str(sys.argv[p+1])
    
    destination = f'{homedir}/Desktop/cigale_vf{filler}'
    
    
    
    run_cigale_all(filler,sfh_type)
    mv_files(filler, destination)
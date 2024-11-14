import sys
import os
from astropy.table import Table, vstack
homedir = os.getenv("HOME")

def run_cigale_all(north_path, south_path, sed_plots, herschel=false):
    
    sed_param = '-sed_plots' if sed_plots else ''
    
    #IF HERSCHEL BANDS, then user must manually complete this following step (e.g., generate 
    #their own .txt files...pending some sort of photometry catalog with row-matched Herschel
    #data.
    if not herschel:
        os.system('python generate_input_files.py -params params.txt')
    
    os.system(f'python run_cigale_one.py -dir_path {north_path} {sed_param} -params params.txt')
    os.system(f'python run_cigale_one.py -dir_path {south_path} {sed_param} -params params.txt')

def concatenate_ns(destination,north_path,south_path):
    
    north_results = Table.read(f'{north_path}/out/results.fits')
    south_results = Table.read(f'{south_path}/out/results.fits')
    
    combined_tab = vstack([north_results, south_results])    
    combined_tab.write(f'{destination}/results_NS.fits',overwrite=True)

def mv_files(north_path,south_path,destination,sed_plots):
    
    #create directories if they do not already exist
    if not os.path.isdir(destination):
        os.system(f'mkdir {destination}')
    
    if sed_plots:
        if not os.path.isdir(f'{destination}/SED_pdfs'):
            os.system(f'mkdir {destination}/SED_pdfs')

        #moving PDF files from both cigale directories to the shiny new one.
        os.system(f'mv {north_path}/out/best_SED_models/*.pdf {destination}/SED_pdfs/')
        os.system(f'mv {south_path}/out/best_SED_models/*.pdf {destination}/SED_pdfs/')

    #concatenate the north and south VF tables and move to the relevant directory
    concatenate_ns(destination,north_path,south_path)
    
if __name__ == "__main__":

    #unpack params.txt file here
    if '-h' in sys.argv or '--help' in sys.argv:
        print("USAGE: %s [-params (name of parameter.txt file, no single or double quotations marks)]")
    
    if '-params' in sys.argv:
        p = sys.argv.index('-params')
        param_file = str(sys.argv[p+1])

    #create dictionary with keyword and values from param textfile
    param_dict={}
    with open(param_file) as f:
        for line in f:
            try:
                key = line.split()[0]
                val = line.split()[1]
                param_dict[key] = val
            except:
                continue
        
        #extract parameters and assign to variables...
        north_path = param_dict['north_output_dir']
        south_path = param_dict['south_output_dir']
        sed_plots = bool(param_dict['sed_plots'])
        
        herschel=false
        if 'PACS' in param_dict['bands_north']:
            herschel=True

    destination = f'{homedir}/Desktop/cigale_vf'

    run_cigale_all(north_path, south_path, sed_plots)
    mv_files(north_path, south_path, destination, sed_plots)
    
    print(f'SEDs (if applicable) and merged results.fits moved to {destination}.')
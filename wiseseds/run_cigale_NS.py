import sys
import os
from astropy.table import Table, vstack
homedir = os.getenv("HOME")

def run_cigale_all(sed_plots,destination,herschel=False):
    
    sed_param = '-sed_plots' if sed_plots else ''
    
    #IF HERSCHEL BANDS, then user must manually complete this following step (e.g., generate 
    #their own .txt files...pending some sort of photometry catalog with row-matched Herschel
    #data).
    if not herschel:
        os.system('python generate_input_files.py -params params.txt')
    
    os.system(f'python run_cigale_one.py -dir_path {destination} {sed_param} -params params.txt')


def run_PDF_all(destination):
    
    os.system(f'python plot_PDF.py -params params.txt')
    
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
        sed_plots = bool(param_dict['sed_plots'])
        
        herschel=False
        if 'PACS' in param_dict['bands_north']:
            herschel=True
        
    destination = param_dict['destination']
    
    create_pdfs = param_dict['create_pdfs']
    
    run_cigale_all(sed_plots,destination,herschel)
    run_PDF_all(destination)
    
    print(f'SEDs+PDFs (if applicable) and results.fits located in {destination}/out/.')
    
    
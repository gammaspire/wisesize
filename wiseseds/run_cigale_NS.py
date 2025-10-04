import sys
import os
from astropy.table import Table, vstack
homedir = os.getenv("HOME")

sys.path.insert(0,'utils')
from param_utils import Params   #inherit Params class


def run_cigale_all(sed_plots, destination, herschel=False):
    
    sed_param = '-sed_plots' if sed_plots else ''
    
    #IF HERSCHEL BANDS, then user must manually complete this following step (e.g., generate 
    #their own .txt files...pending some sort of photometry catalog with row-matched Herschel
    #data).
    if not herschel:
        os.system('python generate_input_files.py -params params.txt')
    
    os.system(f'python cigale_setup.py -dir_path {destination} {sed_param} -params params.txt')


def run_PDF_all(destination):
    
    os.system(f'python plot_PDF.py -params params.txt')
    
if __name__ == "__main__":

    #unpack params.txt file here
    if '-h' in sys.argv or '--help' in sys.argv:
        print("USAGE: %s [-params (name of parameter.txt file, no single or double quotations marks)]")
        sys.exit()
    
    if '-params' in sys.argv:
        p = sys.argv.index('-params')
        param_file = str(sys.argv[p+1])
    else:
        print('-params argument not found. exiting.')
        sys.exit()

    params = Params(param_file)

    herschel=False
    if 'PACS' in params.bands_north:
        herschel=True
                
    run_cigale_all(params.sed_plots, params.destination, herschel)
    
    if params.create_pdfs:
        run_PDF_all(params.destination)
    
    print(f'SEDs+PDFs (if applicable) and results.fits located in {params.destination}/out/.')
    
    
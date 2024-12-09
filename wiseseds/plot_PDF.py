import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

import os
homedir = os.getenv("HOME")

def get_bayes_list(results):
    
    #grab the column names from results.fits
    header_list = results.colnames
    
    #all of the possible wavelength bands
    bands = ['FUV','NUV','WISE1','WISE2','WISE3','WISE4',
         'BASS-g','decamDR1-g','BASS-r','decamDR1-r','decamDR1-z',
            'PACS-blue','PACS-green','PACS-red']
    
    #this one is fun!
    #create a list of all bayes parameters which are NOT fluxes
    #conditions:
        #'bayes' must be in the header label
        #'_err' must not be in the header label
        #the header label cannot be a wavelength band
    
    bayes_list = [x for x in header_list if ('bayes' in x) & ('_err' not in x) & (x.replace('bayes.','') not in bands)]
    
    return bayes_list

def generate_pdf_pngs(results, destination, index, delete_fits=False):
    
    galaxy_id = results['id'][index]
    bayes_list = get_bayes_list(results)

    fig, ax = plt.subplots(3, 3,figsize=(26,16))
    fig.suptitle(f'{galaxy_id} Probability Distribution Functions',fontsize=30,y=0.92)
    ax = ax.flatten()

    for n, item in enumerate(bayes_list):

        ax[n].axvline(np.log10(results[item][index]),color='red',ls='-.',
                      label=f'Bayes Value = {np.log10(results[item][index]):.3f}')

        item = item.replace('bayes.','')
        prob_tab = Table.read(f'{destination}{galaxy_id}_{item}.fits')

        ax[n].plot(np.log10(prob_tab[0][:]),prob_tab['probability'])

        best_value = np.log10(results['best.'+item][index])
        ax[n].axvline(np.log10(results['best.'+item][index]),color='black',ls='--',
                      label=f'Best Value = {best_value:.3f}')

        ax[n].set_xlabel(f'log({item})',fontsize=20)
        ax[n].tick_params(axis='both', labelsize=14)

        ax[n].legend(fontsize=15)
        
        plt.savefig(f'{destination}/{galaxy_id}_PDF.fits', 
                    bbox_inches='tight', pad_inches=0.2, dpi=100)
        plt.close()
        
        #removes the .fits used for the PDFs (9 per galaxy)
        if delete_fits:
            os.system(f'rm {galaxy_id}*.fits')


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
        
    destination = param_dict['destination']
    delete_fits = param_dict['delete_PDF_fits']
    
    results = Table.read(f'{destination}/out/results.fits')
    
    for index in range(len(results)):
        generate_pdf_pngs(results, destination, index, delete_fits)
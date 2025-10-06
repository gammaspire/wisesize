import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

from seaborn import pairplot, load_dataset
import pandas as pd

import os
homedir = os.getenv("HOME")

sys.path.insert(0, 'utils')
from param_utils import Params

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
    
    #create empty table into which I will add all read-in variables and probabilities
    df = pd.DataFrame([])
    
    galaxy_id = results['id'][index]
    bayes_list = get_bayes_list(results)
    
    bayes_list.remove('bayes.sfh.tau_main')   #remove; I decided to replace this parameter with agn_frac
    bayes_list.remove('bayes.stellar.metallicity')   #remove also...I enjoy 3x3 grid

    fig, ax = plt.subplots(3, 3,figsize=(26,16))
    fig.suptitle(f'{galaxy_id} Probability Distribution Functions',fontsize=30,y=0.92)
    ax = ax.flatten()
    
    for n, item in enumerate(bayes_list):
        
        x_val = results[item][index]
        x_err = results[item+'_err'][index]
        
        ax[n].axvline(x_val,color='red',ls='-.',label=f'Bayes Value = {x_val:.3e}')
        ax[n].axvspan(x_val-x_err, x_val+x_err, alpha=0.1, color='red')   #shaded region

        item = item.replace('bayes.','')
        prob_tab = Table.read(f'{destination}out/{galaxy_id}_{item}.fits')

        xcol=[c for c in prob_tab.colnames if c != 'probability'][0]
        
        ax[n].plot(prob_tab[xcol],prob_tab['probability'],marker='o',color='tab:blue')
                
        #add row to the df
        df[f'probability_{item}'] = list(prob_tab['probability'])
        
        best_value = results['best.'+item][index]
        ax[n].axvline(results['best.'+item][index],color='black',ls='--',
                      label=f'Best Value = {best_value:.3e}')

        ax[n].set_xlabel(f'log({item})',fontsize=20)
        ax[n].tick_params(axis='both', labelsize=14)

        ax[n].legend(fontsize=15)

        fig.savefig(f'{destination}out/PDF_fits/{galaxy_id}_PDF.pdf', 
                    bbox_inches='tight', pad_inches=0.2, dpi=100)
        plt.close()
    
    corner_plots(df,destination,galaxy_id)
    
    #removes the .fits used for the PDFs (9 per galaxy)
    if delete_fits:
        os.system(f'rm {destination}out/{galaxy_id}*.fits')
    else:
        os.system(f'mv {destination}out/{galaxy_id}*.fits {destination}out/PDF_fits')

    
def corner_plots(df,destination,galaxy_id,color='orange'):
    
    print(f'PDF finished. generating cornerplot for {galaxy_id}...')
    cplot = pairplot(df,kind='kde',corner=True,diag_kws={'color': color},
            plot_kws={'color': f'dark{color}'})
    cplot.savefig(f'{destination}out/PDF_fits/{galaxy_id}_corner.pdf')
    plt.close()
    
    
if __name__ == "__main__":

    #unpack params.txt file here
    if '-h' in sys.argv or '--help' in sys.argv:
        print("USAGE: %s [-params (name of parameter.txt file, no single or double quotations marks)]")
    
    if '-params' in sys.argv or '--params' in sys.argv:
        p = sys.argv.index('-params')
        param_file = str(sys.argv[p+1])
    else:
        print('-params not found. exiting.')
        sys.exit()
    
    params = Params(param_file)
    
    results = Table.read(f'{params.destination}out/results.fits')
    
    #ensure output directory exists...
    pdf_dir = os.path.join(params.destination, 'out', 'PDF_fits')
    os.makedirs(pdf_dir, exist_ok=True)
    
    for index in range(len(results)):
        generate_pdf_pngs(results, params.destination, index, params.delete_pdf_fits)
        print('Finished!')
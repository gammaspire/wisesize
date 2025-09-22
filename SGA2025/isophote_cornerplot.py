'''
~Galaxy Isophotal Radius Corner Plot Generator~
- for helping diagnose SGA2025 galaxy photometric profiles
- in particular, exploring how SMA_MOMENT correlates with isophotal radii for each wavelength band

- creates a fancy cornerplot for the user-inputted data and band which illustrates how R24, R25, and R26 
  correlate with each other as well as with the SMA_MOMENT.
'''

import sys
import numpy as np
from rich import print
from matplotlib import pyplot as plt
from astropy.table import Table
import pandas as pd
import seaborn as sns

import os
homedir=os.getenv("HOME")


def get_colnames(band='R'):
    """
    Return the column names for isophotal radii in a given photometric band, with SMA_MOMENT prepended.

    Parameters
    ----------
    band : str, optional
        Default = 'R'

    Returns
    -------
    colnames : list of str
        List of column names which will be directly used for the corner plot
    """
    isophotes=['24','25','26']
    band_isophotes = [f'R{isophote}_{band}' for isophote in isophotes]

    colnames = ['SMA_MOMENT'] + band_isophotes
    
    return colnames


def remove_zeros(data_table, colnames):
    """
    Create a boolean mask to remove rows with zero-valued isophotal radii.

    Parameters
    ----------
    data_table : astropy.table.Table
        Input table containing SGA2025 isophotal measurements
    colnames : list of str

    Returns
    -------
    no_zeros_flag : np.ndarray (bool)
        Boolean mask where True means the row contains valid (non-zero) data.
    """
    
    #global 'no zeros' flag -- removes any rows in which there is no isophote data for that galaxy
    no_zeros_flag = (data_table[colnames[0]]>0) & (data_table[colnames[1]]>0) &\
                    (data_table[colnames[2]]>0)

    return no_zeros_flag
    

def generate_cornerplot(data_table=None, band='R'):
    """
    Generate a corner plot comparing SMA_MOMENT with isophotal radii at
    24, 25, and 26 mag in the specified band.

    - Off-diagonal panels show pairwise scatter plots with fitted slopes
    - Diagonal panels show histograms with vertical lines for mean and median

    Parameters
    ----------
    data_table : astropy.table.Table, optional
        Input table containing galaxy measurements. If None, the function
        attempts to load a default FITS file from my (the author's) Desktop
    band : str, optional
        Photometric band (e.g., 'R')

    Returns
    -------
    None
        Displays a seaborn pairplot with colors and stats and data
    """
    if band not in ['G', 'R', 'I', 'Z']:
        print('Please only use the following optical bands: G, R, I, or Z.')
        sys.exit()

    if data_table is None:
        #if table args not given,
        #read in my FITS file
        try:
            path=homedir+'/Desktop/wisesize/SGA2025_tables/SGA2025-ellipse-v0.2.fits'
            data_table=Table.read(path,hdu=1)
        except:
            print(f'Data table not found at {path}. Define the data_table arg or try again.')
            sys.exit()
    
    #grab length of full data table
    len_all = len(data_table)
    
    #get column names
    colnames = get_colnames(band)
    
    #define the "zeros" flag
    no_zeros_flag = remove_zeros(data_table, colnames)
    
    #create df with 'colnames' columns
    tab = Table(data_table[colnames]).to_pandas()

    #create the cornerplot! 
    #filter out galaxies with isophote=0
    p = sns.pairplot(tab[no_zeros_flag], corner=True, height=2, kind='hist', diag_kind='kde')

    #pull the axes from the corner plot
    axes_flat = [ax for ax in p.axes.flat if ax is not None]

    #each diagonal (axis 0, 2, 5, 9, ...) shows the variable ellipse_tab[colname]
    #that colname is index 0, 1, 2, 3, ... as you go down the diagonal
    #this dictionary converts the axis number of the diagonal with the index of the corresponding colname
    ax_dict={0:'0', 2:'1', 5:'2', 9:'3', 14:'4', 20:'5', 27:'6'}

    for i, ax in enumerate(axes_flat):

        if i not in [0,2,5,9,14,20,27]:   #omit histograms; note that hidden axes are None but also have an index

            #grab labels (which are, in fact, the column names!)
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()

            #create 3-sigma clip flag to remove significant outliers when calculating ratios
            clip_flag_x = (tab[x_label] > np.mean(tab[x_label]) - 3*np.std(tab[x_label])) &\
                          (tab[x_label] < np.mean(tab[x_label]) + 3*np.std(tab[x_label]))

            clip_flag_y = (tab[y_label] > np.mean(tab[y_label]) - 3*np.std(tab[y_label])) &\
                          (tab[y_label] < np.mean(tab[y_label]) + 3*np.std(tab[y_label]))

            tab_cut = tab[no_zeros_flag & clip_flag_x & clip_flag_y]
            ratios = tab_cut[y_label] /  tab_cut[x_label]

            #remove ratios with NaN entries
            #these happen to correspond to cases with 0/0, so where there are no isophotal measurements
            #ratios_clean = ratios.dropna()

            median_ratio = np.median(ratios)

            ax.axline([0,0],slope=1, color='red',ls='--',alpha=0.3, label='1-to-1')
            ax.axline([0,0],slope=median_ratio, color='purple', ls='-.', 
                      alpha=0.7, label=f'Slope: {median_ratio:.3f}')

            ax.set_xlim(0,)
            ax.set_ylim(0,)

            ax.legend(fontsize=6,loc='upper left')

        else:

            ax_index = int(ax_dict[i])

            x_label = colnames[ax_index]

            x_label_zeros = (tab[x_label]==0)

            mean = np.mean(tab[x_label])
            median = np.median(tab[x_label])
            stdev = np.std(tab[x_label])

            ax.axvline(mean,color='blue',ls=':',alpha=0.8,label=f'Mean: {mean:.3f} $\pm$ {stdev:.3f}')
            ax.axvline(median,alpha=0,label=f'Median: {median:.3f}')

            ax.text(0.4, 0.3, 
                    f"{100 - len(data_table[x_label][x_label_zeros])*100/len_all:.2f}% with no {x_label}\n",
                    transform=ax.transAxes, ha='left', va='top', 
                    fontsize=7, color='green')

            ax.legend(fontsize=6,handlelength=0, handletextpad=0)

    p.fig.suptitle(f"SMA_MOMENT vs. {band} Isophotal Radii", y=1)     

    plt.show()
    
    
if __name__ == '__main__':
    
    print("Use: generate_cornerplot(data_table=None, band='R')")
    print("Enter your own astropy.table object for data_table; defaults to"
          " my (the author's) path.")
    print('Band can be G, R, I, Z')
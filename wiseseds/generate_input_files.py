#aim: generate input files compatible with CIGALE for a given sample --> __.txt with sample params, and pcigale.ini

from astropy.table import Table

import os
homedir = os.getenv("HOME")

import numpy as np
import sys

sys.path.insert(0,'utils')
from param_utils import Params   #inherit the parameters class
from init_utils import define_flux_dict, check_dir, trim_tables
from conversion_utils import clip_negative_outliers, apply_error_floor

filter_names_all = ['FUV','NUV','G','R','Z','W1','W2','W3','W4']

#return a table which contains galaxy IDs, redshifts, fluxes, and flux errors
def create_fauxtab(params_class, flux_tab, ext_tab, IDs, redshifts):
    
    #isolate needed flux_tab fluxes; convert from nanomaggies to mJy
    #order: FUV, NUV, g, r, (z,) W1, W2, W3, W4
    try:
        flag_n = flux_tab['DEC_MOMENT']>32   #isolates north galaxies
        flag_s = flux_tab['DEC_MOMENT']<32   #isolates south galaxies
    except:
        flag_n = flux_tab['DEC']>32   #isolates north galaxies
        flag_s = flux_tab['DEC']<32   #isolates south galaxies
    
    N=len(flux_tab) #all galaxies...north and south.
    dtype=[('OBJID','str'),('redshift','f4'),('FUV','f4'),('FUV_err','f4'),
           ('NUV','f4'),('NUV_err','f4'),('G','f4'),('G_err','f4'),
          ('R','f4'),('R_err','f4'),('Z','f4'),('Z_err','f4'),
          ('W1','f4'),('W1_err','f4'),('W2','f4'),('W2_err','f4'),
          ('W3','f4'),('W3_err','f4'),('W4','f4'),('W4_err','f4')]
    
    faux_tab = Table(data=np.zeros(N,dtype=dtype))
    faux_tab['OBJID']=IDs
    faux_tab['redshift']=redshifts
        
    #define conversion factor for flux
    conversion_factor = 1.
    #if True, convert fluxes from nanomaggies to mJy
    if params_class.convert_flux:
        conversion_factor = 3.631e-3
        
    #for all filters in filter_names_all...populate the respective data column
    for i in filter_names_all:
        
        fluxes = flux_tab[params_class.flux_id_col + i] * conversion_factor
        flux_errs = flux_tab[params_class.flux_id_col_err + i]   #do not apply conversion factor just yet
        
        ###################
        # Clean the data! #
        ###################
        
        #first create flags to identify every row with no photometry
        no_flux_flag = (fluxes==0.) & (flux_errs==0.)

        #any row with no fluxes will be assigned an np.nan
        fluxes[no_flux_flag] = np.nan
        flux_errs[no_flux_flag] = np.nan
        
        #for galaxies WITH photometry!
        
        #if need to convert invariance to an error...do so
        if params_class.ivar_to_err:
            flux_errs[~no_flux_flag] = np.sqrt(1/flux_errs[~no_flux_flag]) * conversion_factor
            
            #if neither condition is met, convert error as normal
        else:
            flux_errs[~no_flux_flag] = flux_errs[~no_flux_flag] * conversion_factor

        ###################
        # EXTINCTION CORR #
        ###################
        
        #check for flag indicating conversion from transmission to extinction (in magnitudes) is needed
        if params_class.transmission_to_extinction:
            ext_values = -2.5 * np.log10(ext_tab[params_class.extinction_col+i.lower()])
        else:
            ext_values = ext_tab[params_class.extinction_col+i.lower()]
        
        #Milky Way (MW) extinction corrections (SFD) for each band, given in magnitudes.
        ext_corrections = 10.**(ext_values/2.5)   #converting to linear scale factors
                
        #now apply SFD extinction correction (per Legacy Survey)
        flux_errs[~no_flux_flag] *= ext_corrections[~no_flux_flag]
        fluxes[~no_flux_flag] *= ext_corrections[~no_flux_flag]

        #If the relative error dF/F < 0.10, then let dF = 0.10*F
        #the idea is that our MINIMUM error floor for fluxes will be set as 10% of the flux value
        #for grz and 15% for W1-4 & NUV+FUV. 
        #CURRENTLY --> using 10% for grz; 13% for FUV, NUV, W1-4.
        flux_errs[~no_flux_flag] = apply_error_floor(fluxes[~no_flux_flag], flux_errs[~no_flux_flag], band=i)


        #...another conditional statement. if zero is within the 4-sigma confidence interval of the flux value, keep the negative value. if the flux is OUTSIDE of this limit, then set to NaN. 
        fluxes, flux_errs = clip_negative_outliers(fluxes, flux_errs)
          

        #appending arrays to faux table
        faux_tab[i] = fluxes
        faux_tab[f'{i}_err'] = flux_errs
        
    faux_tab.add_columns([flag_n,flag_s],names=['flag_north','flag_south'])

    return faux_tab
        

def create_input_files(params_class, trim=True):
        
    #define flux table, extinction table
    ext_tab = params_class.ext_tab
    flux_tab = params_class.flux_tab
    
    IDs = params_class.IDs
    redshifts = params_class.redshifts
    
    #re-define variables with trimmed data
    if trim:
        IDs, redshifts, flux_tab, ext_tab = trim_tables(IDs, redshifts, flux_tab, ext_tab)
    
    #contains FUV, NUV, G, R, Z, W1, W2, W3, W4, north flag, south flag for all galaxies
    faux_table = create_fauxtab(params_class, flux_tab=flux_tab, ext_tab=ext_tab, IDs=IDs, redshifts=redshifts)
    
    #generate flux dictionaries to map the params.txt flux bands to their CIGALE labels
    flux_dict_north = define_flux_dict('n')
    flux_dict_south = define_flux_dict('s')
    
    bands_north = list(flux_dict_north.keys())
    bands_south = list(flux_dict_south.keys())
    
    #write files...
    check_dir(params_class.dir_path)
        
    with open(params_class.dir_path+'/galaxy_data.txt', 'w') as file:
        
        #create file header!
        s = '# id redshift '
        
        #if the band appears in both hemispheres (like G, R), write only once.
        #otherwise, write labels only if they exist in north/south dict.
        for flux in filter_names_all:
            
            #len(flux)>=2 isolates NUV, FUV, W1-4 bands. excludes gr(z) bands
            if (flux in bands_north) & (flux in bands_south) & (len(flux)>=2):
                s = s + f'{flux_dict_north[flux]} {flux_dict_north[flux]}_err ' #same for N & S
            
            #for G and R, add twice (as we will need one label each for BASS-g north and DECam-g south
            #add labels if needed, else ''.
            #adding them consecutively ensures the file will read something like BASS-g BASS-g_err DECam-g DECam-g_err ...
            #I guess '' is a failsafe in case the band is not in north or south...effectively skips the header label
            #maybe I set that up once upon a time to accommodate Z-band? I don't know.
            else:
                s = s + f'{flux_dict_north[flux]} {flux_dict_north[flux]}_err ' if flux in bands_north else s + ''
                s = s + f'{flux_dict_south[flux]} {flux_dict_south[flux]}_err ' if flux in bands_south else s + ''
            
        s = s + ' \n'
        file.write(s)
        
        #storing header informtaion in a list so I can interate over it
        #however, I am only keeping the CIGALE FILTER LABELS. no #, no id, no redshift
        filter_labels_all = [x for x in s.split() if ('_err' not in x) & (x!='#') & (x!='id') & (x!='redshift')]
        
        #you may wonder why G and R are included TWICE!
            #There are two sources of G and R depending on whether galaxy is in northern
            #or southern hemisphere. The first G (from flux_dict_north) becomes 'BASS-g,' the second 'decamDR1-g'
            #this is how I set up the header file earlier!
        filter_comp_names = ['FUV','NUV','G','G','R','R','Z','W1','W2','W3','W4']
        print(filter_labels_all)
        
        #for every "good" galaxy in flux_tab, add a row to the text file with relevant information
        
        ####################
        ###NORTH GALAXIES###
        ####################
        
        for n in faux_table[faux_table['flag_north']]:
                            
            #the first two will be ID [0] and redshift [1], by design.
            s_gal = f'{n[0]} {round(n[1],4) } '
            
            for i in range(len(filter_labels_all)):
                if (filter_comp_names[i]!='Z'):
                    if filter_labels_all[i] == flux_dict_north[filter_comp_names[i]]:
                        flux_val = n[filter_comp_names[i]]
                        flux_err = n[f'{filter_comp_names[i]}_err']
                        #for the flux values and errors, round to 3 decimal places
                        s_gal = s_gal + "%.3f "%flux_val + "%.3f "%flux_err
                    else:
                        s_gal = s_gal + 'nan nan '
                else:
                    s_gal = s_gal + 'nan nan '
            
            s_gal = s_gal + '\n'
            file.write(s_gal)
        print('n galaxies finished', len(faux_table[faux_table['flag_north']]))
        
        ####################
        ###SOUTH GALAXIES###
        ####################
        
        for n in faux_table[faux_table['flag_south']]:
                
            #the first two will be ID and redshift, by design.
            s_gal = f'{n[0]} {round(n[1],4) } '

            for i in range(len(filter_labels_all)):
                if filter_labels_all[i] == flux_dict_south[filter_comp_names[i]]:
                    flux_val = n[filter_comp_names[i]]
                    flux_err = n[f'{filter_comp_names[i]}_err']
                    #for the flux values and errors, round to 3 decimal places
                    s_gal = s_gal + "%.3f "%flux_val + "%.3f "%flux_err
                else:
                    s_gal = s_gal + 'nan nan '

            s_gal = s_gal + '\n'
            file.write(s_gal)
            
        print('s galaxies finished', len(faux_table[faux_table['flag_south']]))   


def create_ini_file(params_class): #dir_path, sfh_module, dust_module, ncores):
    
    check_dir(params_class.dir_path)
    
    #create pcigale.ini files
    with open(params_class.dir_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = galaxy_data.txt \n')
        file.write('parameters_file = \n')
        file.write(f'sed_modules = {params_class.sfh_module}, bc03, nebular, dustatt_modified_CF00, {params_class.dust_module}, skirtor2016, redshifting \n')
        file.write('analysis_method = pdf_analysis \n')
        file.write(f'cores = {params_class.ncores} \n')  

    #create pcigale.ini.spec files
    
    with open(params_class.dir_path+'/pcigale.ini.spec', 'w') as file:
        file.write('data_file = string() \n')
        file.write('parameters_file = string() \n')
        file.write('sed_modules = cigale_string_list() \n')
        file.write('analysis_method = string() \n')
        file.write('cores = integer(min=1)')
        
        
if __name__ == "__main__":

    #unpack args here
    if '-h' in sys.argv or '--help' in sys.argv:
        print("USAGE: %s [-params (name of parameter.txt file, no single or double quotations marks)]")
        sys.exit()
    
    if '-params' in sys.argv:
        p = sys.argv.index('-params')
        param_file = str(sys.argv[p+1])
    else:
        print('-params argument not found. exiting.')
        sys.exit()
    
    #define ALL parameters using Params class (utils/param_utils.py)
    params = Params(param_file)
    trim = True
    
    params.load_tables()   #load the tables ext_tab, main_tab, phot_tab
    
    params.load_columns()  #defines galaxy IDs (params.IDs) and redshifts (params.redshifts)
    
    
    #annnnnd, run
    create_ini_file(params)
    create_input_files(params, trim)
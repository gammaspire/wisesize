#aim: generate input files compatible with CIGALE for a given sample --> __.txt with sample params, and pcigale.ini

from astropy.table import Table

import os
homedir = os.getenv("HOME")

import numpy as np
import sys


def define_flux_dict(n_or_s):
    
    if n_or_s=='n':
        flux_dict = {'FUV':'FUV', 'NUV':'NUV', 'G':'BASS-g', 'R':'BASS-r',
                     'W1':'WISE1', 'W2':'WISE2', 'W3':'WISE3', 'W4':'WISE4'} 
                     #'PACS1':'PACS_blue', 'PACS2':'PACS_green', 'PACS3':'PACS_red'}
    else:
        flux_dict = {'FUV':'FUV', 'NUV':'NUV', 'G':'decamDR1-g', 'R':'decamDR1-r',
                     'Z':'decamDR1-z', 'W1':'WISE1', 'W2':'WISE2', 'W3':'WISE3', 'W4':'WISE4'}
                     #'PACS1':'PACS_blue', 'PACS2':'PACS_green', 'PACS3':'PACS_red'}
    return flux_dict

#for low-z objects, v = cz
def get_redshift(Vcosmic_array):
    
    try:
        z=Vcosmic_array/3e5
    except:
        z=np.asarray(Vcosmic_array)/3e5
    return z


#trim flags according to redshift values (must be positive) and whether the galaxies contain photometry data
def trim_tables(IDs, redshifts, flux_tab, ext_tab):
    
    all_flags = (redshifts>0.) & (flux_tab['photFlag'])
    
    return IDs[all_flags], redshifts[all_flags], flux_tab[all_flags], ext_tab[all_flags]


#return a table which contains galaxy IDs, redshifts, fluxes, and flux errors
def create_fauxtab(IDs, redshifts, flux_tab, ext_tab, n_or_s, bands):
    
    #isolate needed flux_tab fluxes; convert from nanomaggies to mJy
    #order: FUV, NUV, g, r, (z,) W1, W2, W3, W4
    if n_or_s == 'n':
        flag = flux_tab['DEC_MOMENT']>32   #isolates north galaxies
        filter_names = bands
    elif n_or_s == 's':
        flag = flux_tab['DEC_MOMENT']<32   #isolates south galaxies
        filter_names = bands
    else:
        print('Please enter n or s for the n_or_s argument!')
        sys.exit
        
    fluxes = []
    flux_ivars = []
    flux_errs = []
    ext_corrections = []
    
    for i in filter_names:
        fluxes.append(flux_tab[f'FLUX_AP06_{i}']*3.631e-3)
        flux_ivars.append(flux_tab[f'FLUX_IVAR_AP06_{i}'])
        #Milky Way (MW) extinction corrections (SFD) for each band, given in magnitudes.
        ext_corrections.append(10.**(ext_tab[f'A({i})_SFD']/2.5))   #converting to linear scale factors
        #for converting invariances to errors
        flux_errs.append(np.zeros(len(flux_tab)))         

    #for every list of fluxes...
    for index in range(len(flux_ivars)):
        #for every element in that list of fluxes...
        for n in range(len(flux_errs[index])):
            
            #if zero (indicates that there are no data for these filters), replace with NaNs
            
            if ((fluxes[index][n]==0.) & (flux_errs[index][n]==0.)):
                fluxes[index][n] = 'NaN'
                flux_errs[index][n] = 'NaN'
            
            #if neither condition is met, calculate error as normal
            else:
                flux_errs[index][n] = np.sqrt(1/flux_ivars[index][n])*3.631e-3
                
                #now apply SFD extinction correction (per Legacy Survey)
                flux_errs[index][n] *= ext_corrections[index][n]
                fluxes[index][n] *= ext_corrections[index][n]  

                #If the relative error dF/F < 0.10, then let dF = 0.10*F
                #the idea is that our MINIMUM error floor for fluxes will be set as 10% of the flux value
                #for grz and 15% for W1-4 & NUV+FUV. 
                #CURRENTLY --> using 10% for grz; 13% for FUV, NUV, W1-4.
                if index in [0,1,4,5,6,7]:   #FUV, NUV, W1, W2, W3, W4
                    if (flux_errs[index][n]/np.abs(fluxes[index][n])) < 0.13:
                        flux_errs[index][n] = np.abs(0.13*fluxes[index][n])
                else:   #legacy grz
                    if (flux_errs[index][n]/np.abs(fluxes[index][n])) < 0.10:
                        flux_errs[index][n] = np.abs(0.10*fluxes[index][n])

                #...another conditional statement. if zero is within the 4-sigma confidence interval of the flux value, keep the negative value. if the flux is OUTSIDE of this limit, then set to NaN. 

                if (fluxes[index][n]<0.) & ~((0.<(fluxes[index][n]+flux_errs[index][n]*4)) & (0.>(fluxes[index][n]-flux_errs[index][n]*4))):

                    fluxes[index][n] = 'NaN'
                    flux_errs[index][n] = 'NaN'

    #create table to organize results

    #I will need these flags as well. humph.
    north_flag = flux_tab['DEC_MOMENT']>32
    south_flag = flux_tab['DEC_MOMENT']<32
    
    if n_or_s == 'n':
        flag=north_flag       
    else:
        flag=south_flag

    faux_table = Table([flux_tab['VFID'][flag],np.round(redshifts,4)[flag]],
                           names=['VFID','redshift'])   
    for index,label in enumerate(filter_names):
        faux_table.add_column(fluxes[index][flag],name=label)
        faux_table.add_column(flux_errs[index][flag],name=f'{label}_err')

    return faux_table


def check_dir(north_path, south_path):
    
    if not os.path.isdir(north_path):
        os.mkdir(north_path)
        print(f'Created {north_path}')
    if not os.path.isdir(south_path):
        os.mkdir(south_path)
        print(f'Created {south_path}')
        

def create_input_files(IDs, redshifts, flux_tab, ext_tab, north_path, south_path, 
                       bands_north, bands_south, trim=True):
    
    if trim:
        IDs, redshifts, flux_tab, ext_tab = trim_tables(IDs,redshifts,flux_tab,ext_tab)
    
    faux_table_north = create_fauxtab(IDs, redshifts, flux_tab, ext_tab, 'n', bands_north)
    faux_table_south = create_fauxtab(IDs, redshifts, flux_tab, ext_tab, 's', bands_south)
    
    #generate flux dictionaries to map the params.txt flux bands to their CIGALE labels
    flux_dict_north = define_flux_dict('n')
    flux_dict_south = define_flux_dict('s')
    
    #write files...
    check_dir(north_path, south_path)
    
    with open(north_path+'/vf_data_north.txt', 'w') as file:
        
        #create file header
        s = '# id redshift'
        for flux in bands_north:
            s = s + f' {flux_dict_north[flux]} {flux_dict_north[flux]}_err'
        s = s + ' \n'
        file.write(s)
        
        #for every "good" galaxy in flux_tab, add a row to the text file with relevant information
        for n in faux_table_north:
            
            #create empty string variable which I will use to iteratively update the new row
            s_gal = ''
            
            for i in range(len(n)):
                
                #the first two will be ID and redshift, by design.
                if (i==0) or (i==1):
                    s_gal = s_gal + f'{n[i]} '
                #for the flux values and errors, round to 3 decimal places
                else:
                    s_gal = s_gal + "%.3f "%n[i]
            
            s_gal = s_gal + '\n'
            
            file.write(s_gal)

        file.close()    

    with open(south_path+'/vf_data_south.txt', 'w') as file:
        #create file header
        s = '# id redshift'
        for flux in bands_south:
            s = s + f' {flux_dict_south[flux]} {flux_dict_south[flux]}_err'
        s = s + ' \n'
        file.write(s)

        #for every "good" galaxy in flux_tab, add a row to the text file with relevant information
        for n in faux_table_south:
            
            #create empty string variable which I will use to iteratively update the new row
            s_gal = ''
            
            for i in range(len(n)):
                if (i==0) or (i==1):
                    s_gal = s_gal + f'{n[i]} '
                else:
                    s_gal = s_gal + "%.3f "%n[i]
            
            s_gal = s_gal + '\n'
            file.write(s_gal)
        
        file.close() 
        

def create_ini_file(north_path, south_path, sfh_module, dust_module, ncores):
    
    check_dir(north_path, south_path)    
    
    #create pcigale.ini files
    with open(north_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = vf_data_north.txt \n')
        file.write('parameters_file = \n')
        file.write(f'sed_modules = {sfh_module}, bc03, nebular, dustatt_modified_CF00, {dust_module}, skirtor2016, redshifting \n')
        file.write('analysis_method = pdf_analysis \n')
        file.write(f'cores = {ncores} \n')
        file.close()    
    with open(south_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = vf_data_south.txt \n')
        file.write('parameters_file = \n')
        file.write(f'sed_modules = {sfh_module}, bc03, nebular, dustatt_modified_CF00, {dust_module}, skirtor2016, redshifting \n')
        file.write('analysis_method = pdf_analysis \n')
        file.write(f'cores = {ncores} \n')
        file.close()    
        
    #create pcigale.ini.spec files
    
    with open(north_path+'/pcigale.ini.spec', 'w') as file:
        file.write('data_file = string() \n')
        file.write('parameters_file = string() \n')
        file.write('sed_modules = cigale_string_list() \n')
        file.write('analysis_method = string() \n')
        file.write('cores = integer(min=1)')
        file.close()
    with open(south_path+'/pcigale.ini.spec', 'w') as file:    
        file.write('data_file = string() \n')
        file.write('parameters_file = string() \n')
        file.write('sed_modules = cigale_string_list() \n')
        file.write('analysis_method = string() \n')
        file.write('cores = integer(min=1) \n')
        file.close()        
        
def run_all(Vcosmic_array, IDs, flux_tab, ext_tab, north_path, south_path,
           bands_north, bands_south, sfh_module='sfh2exp', dust_module='dl2014', ncores=1,
           trim=True):

    create_ini_file(north_path, south_path, sfh_module=sfh_module, dust_module=dust_module, 
                    ncores=ncores)
    redshifts = get_redshift(Vcosmic_array)
    create_input_files(IDs, redshifts, flux_tab, ext_tab, north_path, south_path, 
                      bands_north, bands_south, trim)
    

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
        path_to_repos = param_dict['path_to_repos']
        vcosmic_table = param_dict['vcosmic_table']
        phot_table = param_dict['phot_table']
        extinction_table = param_dict['extinction_table']
        
        id_col = param_dict['galaxy_ID_col']
        
        bands_north = param_dict['bands_north'].split("-")   #list!
        bands_south = param_dict['bands_south'].split("-")   #list!
        
        ncores = param_dict['ncores']
        
        sfh_module = param_dict['sfh_module']
        dust_module = param_dict['dust_module']
    
        north_path = param_dict['north_output_dir']
        south_path = param_dict['south_output_dir']
    
    trim = True
    
    #load tables
    vf = Table.read(path_to_repos+vcosmic_table)
    flux_tab = Table.read(path_to_repos+phot_table)
    ext_tab = Table.read(path_to_repos+extinction_table)
    
    Vcosmic_array = vf['Vcosmic']
    IDs = vf[id_col]

    run_all(Vcosmic_array, IDs, flux_tab, ext_tab, north_path, south_path,
            bands_north, bands_south, sfh_module=sfh_module, dust_module=dust_module, 
            ncores=ncores,trim=True)
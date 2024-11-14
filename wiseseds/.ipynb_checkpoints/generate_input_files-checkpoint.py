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
    else:
        flux_dict = {'FUV':'FUV', 'NUV':'NUV', 'G':'decamDR1-g', 'R':'decamDR1-r',
                     'Z':'decamDR1-z', 'W1':'WISE1', 'W2':'WISE2', 'W3':'WISE3', 'W4':'WISE4'}
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
def create_fauxtab(IDs, redshifts, flux_tab, ext_tab):
    
    #isolate needed flux_tab fluxes; convert from nanomaggies to mJy
    #order: FUV, NUV, g, r, (z,) W1, W2, W3, W4
    flag_n = flux_tab['DEC_MOMENT']>32   #isolates north galaxies
    flag_s = flux_tab['DEC_MOMENT']<32   #isolates south galaxies
    
    filter_names_all = ['FUV','NUV','G','R','Z','W1','W2','W3','W4']
    
    faux_table = Table([flux_tab['VFID'],np.round(redshifts,4)],
                           names=['VFID','redshift']) 
    
    N=len(flux_tab) #all VF galaxies...north and south.
    dtype=[('VFID','str'),('redshift','f4'),('FUV','f4'),('FUV_err','f4'),
           ('NUV','f4'),('NUV_err','f4'),('G','f4'),('G_err','f4'),
          ('R','f4'),('R_err','f4'),('Z','f4'),('Z_err','f4'),
          ('W1','f4'),('W1_err','f4'),('W2','f4'),('W2_err','f4'),
          ('W3','f4'),('W3_err','f4'),('W4','f4'),('W4_err','f4')]
    
    faux_tab = Table(data=np.zeros(N,dtype=dtype))
    faux_tab['VFID']=IDs
    faux_tab['redshift']=redshifts
        
    for i in filter_names_all:
        
        fluxes = flux_tab[f'FLUX_AP06_{i}']*3.631e-3
        flux_ivars = flux_tab[f'FLUX_IVAR_AP06_{i}']
        #Milky Way (MW) extinction corrections (SFD) for each band, given in magnitudes.
        ext_corrections = 10.**(ext_tab[f'A({i})_SFD']/2.5)   #converting to linear scale factors
        flux_errs = np.zeros(len(fluxes)) 
        
        #for every element in that list of fluxes...convert invariance to err if able
        for n in range(len(fluxes)):
            
            #if zero (indicates that there are no data for these filters), replace with NaNs
            
            if ((fluxes[n]==0.) & (flux_errs[n]==0.)):
                fluxes[n] = 'NaN'
                flux_errs[n] = 'NaN'
            
            #if neither condition is met, calculate error as normal
            else:
                flux_errs[n] = np.sqrt(1/flux_ivars[n])*3.631e-3
                
                #now apply SFD extinction correction (per Legacy Survey)
                flux_errs[n] *= ext_corrections[n]
                fluxes[n] *= ext_corrections[n]  

                #If the relative error dF/F < 0.10, then let dF = 0.10*F
                #the idea is that our MINIMUM error floor for fluxes will be set as 10% of the flux value
                #for grz and 15% for W1-4 & NUV+FUV. 
                #CURRENTLY --> using 10% for grz; 13% for FUV, NUV, W1-4.
                if n in [0,1,4,5,6,7]:   #FUV, NUV, W1, W2, W3, W4
                    if (flux_errs[n]/np.abs(fluxes[n])) < 0.13:
                        flux_errs[n] = np.abs(0.13*fluxes[n])
                else:   #legacy grz
                    if (flux_errs[n]/np.abs(fluxes[n])) < 0.10:
                        flux_errs[n] = np.abs(0.10*fluxes[n])

                #...another conditional statement. if zero is within the 4-sigma confidence interval of the flux value, keep the negative value. if the flux is OUTSIDE of this limit, then set to NaN. 

                if (fluxes[n]<0.) & ~((0.<(fluxes[n]+flux_errs[n]*4)) & (0.>(fluxes[n]-flux_errs[n]*4))):

                    fluxes[n] = 'NaN'
                    flux_errs[n] = 'NaN'
            
        #appending arrays to faux table
        faux_tab[i] = fluxes
        faux_tab[f'{i}_err'] = flux_errs
        
    faux_tab.add_columns([flag_n,flag_s],names=['flag_north','flag_south'])

    return faux_tab


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        print(f'Created {dir_path}')
        

def create_input_files(IDs, redshifts, flux_tab, ext_tab, dir_path, bands_north, bands_south, trim=True):
    
    filter_names_all = ['FUV','NUV','G','R','Z','W1','W2','W3','W4']

    if trim:
        IDs, redshifts, flux_tab, ext_tab = trim_tables(IDs,redshifts,flux_tab,ext_tab)
    
    #contains FUV, NUV, G, R, Z, W1, W2, W3, W4, north flag, south flag for ALL VF galaxies
    faux_table = create_fauxtab(IDs, redshifts, flux_tab, ext_tab)
    
    #generate flux dictionaries to map the params.txt flux bands to their CIGALE labels
    flux_dict_north = define_flux_dict('n')
    flux_dict_south = define_flux_dict('s')
    
    #write files...
    check_dir(dir_path)
        
    with open(dir_path+'/vf_data.txt', 'w') as file:
        
        #create file header
        s = '# id redshift '
        
        #for flux in bands_all
        
        for flux in filter_names_all:
            
            #if flux included twice and is NOT GR(Z), only count once.
            if (flux in bands_north) & (flux in bands_south) & (len(flux)>=2):
                s = s + f'{flux_dict_north[flux]} {flux_dict_north[flux]}_err ' #same for N&S
            
            #otherwise, proceed as normal. add labels if needed, else ''.
            else:
                s = s + f'{flux_dict_north[flux]} {flux_dict_north[flux]}_err ' if flux in bands_north else s + ''
                s = s + f'{flux_dict_south[flux]} {flux_dict_south[flux]}_err ' if flux in bands_south else s + ''
            
        s = s + ' \n'
        file.write(s)
        
        #recreating list of header information...BUT FILTERS ONLY!
        filter_labels_all = [x for x in s.split() if ('_err' not in x) & (x!='#') & (x!='id') & (x!='redshift')]
        filter_comp_names = ['FUV','NUV','G','G','R','R','Z','W1','W2','W3','W4']
        print(filter_labels_all)
        
        #for every "good" galaxy in flux_tab, add a row to the text file with relevant information
        ###NORTH GALAXIES###
        for n in faux_table[faux_table['flag_north']]:
                            
            #the first two will be ID and redshift, by design.
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
        
        ###SOUTH GALAXIES###
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
           
        
        file.close()    


def create_ini_file(dir_path, sfh_module, dust_module, ncores):
    
    check_dir(dir_path)
    
    #create pcigale.ini files
    with open(dir_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = vf_data.txt \n')
        file.write('parameters_file = \n')
        file.write(f'sed_modules = {sfh_module}, bc03, nebular, dustatt_modified_CF00, {dust_module}, skirtor2016, redshifting \n')
        file.write('analysis_method = pdf_analysis \n')
        file.write(f'cores = {ncores} \n')
        file.close()    

    #create pcigale.ini.spec files
    
    with open(dir_path+'/pcigale.ini.spec', 'w') as file:
        file.write('data_file = string() \n')
        file.write('parameters_file = string() \n')
        file.write('sed_modules = cigale_string_list() \n')
        file.write('analysis_method = string() \n')
        file.write('cores = integer(min=1)')
        file.close() 
        
def run_all(Vcosmic_array, IDs, flux_tab, ext_tab, dir_path, bands_north, bands_south,
            sfh_module='sfh2exp', dust_module='dl2014', ncores=1, trim=True):

    create_ini_file(dir_path, sfh_module=sfh_module, dust_module=dust_module, 
                    ncores=ncores)
    redshifts = get_redshift(Vcosmic_array)
    create_input_files(IDs, redshifts, flux_tab, ext_tab, dir_path, bands_north, 
                       bands_south, trim)
    

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
        
        dir_path = param_dict['destination']
                
        id_col = param_dict['galaxy_ID_col']
        
        bands_north = param_dict['bands_north'].split("-")   #list!
        bands_south = param_dict['bands_south'].split("-")   #list!
        
        ncores = param_dict['ncores']
        
        #in order to save the probability distribution functions, ncores = nblocks = 1
        if bool(param_dict['create_pdfs']):
            ncores = 1
        
        sfh_module = param_dict['sfh_module']
        dust_module = param_dict['dust_module']
    
    trim = True
    
    #load tables
    vf = Table.read(path_to_repos+vcosmic_table)
    flux_tab = Table.read(path_to_repos+phot_table)
    ext_tab = Table.read(path_to_repos+extinction_table)
    
    Vcosmic_array = vf['Vcosmic']
    IDs = vf[id_col]

    run_all(Vcosmic_array, IDs, flux_tab, ext_tab, dir_path, bands_north, 
            bands_south, sfh_module=sfh_module, dust_module=dust_module, 
            ncores=ncores,trim=True)
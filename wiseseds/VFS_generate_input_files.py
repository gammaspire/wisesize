#aim: generate input files compatible with CIGALE for a given sample --> __.txt with sample params, and pcigale.ini


from astropy.table import Table
import numpy as np
import os

homedir = os.getenv("HOME")

#for low-z objects, v = cz
def get_redshift(Vcosmic_array):
    
    try:
        z = Vcosmic_array/3e5
    except:
        z = np.asarray(Vcosmic_array)/3e5
    return z
        
def trim_tables(IDs, redshifts, flux_tab):
    
    all_flags = (redshifts>0.) & (flux_tab['photFlag'])
    return IDs[all_flags], redshifts[all_flags], flux_tab[all_flags]
    
    
def create_fauxtab(IDs, redshifts, flux_tab):
    
    #isolate needed flux_tab fluxes; convert from nanomaggies to mJy
    #order: FUV, NUV, g, r, W1, W2, W3, W4
    fluxes = [flux_tab['FLUX_AP06_FUV']*3.631e-3, flux_tab['FLUX_AP06_NUV']*3.631e-3, flux_tab['FLUX_AP06_G']*3.631e-3,
             flux_tab['FLUX_AP06_R']*3.631e-3, flux_tab['FLUX_AP06_Z']*3.631e-3, 
              flux_tab['FLUX_AP06_W1']*3.631e-3, flux_tab['FLUX_AP06_W2']*3.631e-3,
             flux_tab['FLUX_AP06_W3']*3.631e-3, flux_tab['FLUX_AP06_W4']*3.631e-3]

    flux_ivars = [flux_tab['FLUX_IVAR_AP06_FUV'], flux_tab['FLUX_IVAR_AP06_NUV'],
                 flux_tab['FLUX_IVAR_AP06_G'], flux_tab['FLUX_IVAR_AP06_R'],
                  flux_tab['FLUX_IVAR_AP06_Z'],
                 flux_tab['FLUX_IVAR_AP06_W1'], flux_tab['FLUX_IVAR_AP06_W2'],
                 flux_tab['FLUX_IVAR_AP06_W3'], flux_tab['FLUX_IVAR_AP06_W4']]
    
    flux_errs = [np.zeros(len(flux_tab)),np.zeros(len(flux_tab)),np.zeros(len(flux_tab)),np.zeros(len(flux_tab)),
             np.zeros(len(flux_tab)),np.zeros(len(flux_tab)),np.zeros(len(flux_tab)),np.zeros(len(flux_tab)),
                 np.zeros(len(flux_tab))]

    #for every list of fluxes...
    for index in range(len(flux_ivars)):
        #for every element in that list of fluxes...
        for n in range(len(flux_errs[index])):
            #if zero or negative, replace with NaN
            #can do for both fluxes AND flux errors!
            if (flux_ivars[index][n]==0.) | (flux_ivars[index][n]<0.) | (fluxes[index][n]<0.) | (fluxes[index][n]==0.):
                flux_errs[index][n] = 'NaN'  
                fluxes[index][n] = 'NaN'
            #if not zero, calculate error as normal
            else:
                flux_errs[index][n] = np.sqrt(1/flux_ivars[index][n])*3.631e-3
            
            #ANOTHER conditional statement. If the relative error dF/F < 0.05, then let dF = 0.05*F
            #the idea is that our MINIMUM error floor for fluxes will be set as 5% of the flux value.
            if (flux_errs[index][n]/fluxes[index][n]) < 0.05:
                flux_errs[index][n] = 0.05
                
    
    #create table to organize results

    faux_table = Table([flux_tab['VFID'],np.round(redshifts,4),fluxes[0],
                              flux_errs[0],fluxes[1],flux_errs[1],
                              fluxes[2],flux_errs[2],fluxes[3],
                              flux_errs[3],fluxes[4],flux_errs[4],
                              fluxes[5],flux_errs[5],fluxes[6],
                              flux_errs[6],fluxes[7],flux_errs[7],fluxes[8],flux_errs[8]],
                              names=['VFID','redshift','FUV','FUV_err','NUV','NUV_err',
                                     'g', 'g_err', 'r', 'r_err','z','z_err',
                                     'WISE1','WISE1_err','WISE2','WISE2_err','WISE3','WISE3_err','WISE4','WISE4_err'])
    
    return faux_table

def check_dir(north_path, south_path):
    
    if not os.path.isdir(north_path):
        os.mkdir(homedir+'/Desktop/cigale_vf_north')
        print('Created '+ homedir + '/Desktop/cigale_vf_north')
    if not os.path.isdir(south_path):
        os.mkdir(homedir+'/Desktop/cigale_vf_south')
        print('Created '+ homedir + '/Desktop/cigale_vf_south')

def create_input_files(IDs, redshifts, flux_tab, north_path, south_path, trim=True):
    
    if trim:
        IDs, redshifts, flux_tab = trim_tables(IDs,redshifts,flux_tab)
    
    faux_table = create_fauxtab(IDs, redshifts, flux_tab)
    
    #I will need these flags as well. humph.
    north_flag = flux_tab['DEC_MOMENT']>32
    south_flag = flux_tab['DEC_MOMENT']<32
    
    #write files...
    
    check_dir(north_path, south_path)
    
    with open(homedir+'/Desktop/cigale_vf_north/vf_data_north.txt', 'w') as file:
        #create file header
        s = '# id redshift FUV FUV_err NUV NUV_err BASS-g BASS-g_err BASS-r BASS-r_err MzLS-z MzLS-z_err WISE1 WISE1_err WISE2 WISE2_err WISE3 WISE3_err WISE4 WISE4_err'+' \n'
        file.write(s)

        #for every "good" galaxy in flux_tab, add a row to the text file with relevant information
        for n in faux_table[north_flag]:

            s_gal = f"{n[0]} {n[1]} %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f"%(n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],n[12],n[13],n[14],n[15],n[16],n[17],n[18],n[19]) + '\n'
            file.write(s_gal)

        file.close()    

    with open(homedir+'/Desktop/cigale_vf_south/vf_data_south.txt', 'w') as file:
        #create file header
        s = '# id redshift FUV FUV_err NUV NUV_err decamDR1-g decamDR1-g_err decamDR1-r decamDR1-r_err decamDR1-z decamDR1-z_err WISE1 WISE1_err WISE2 WISE2_err WISE3 WISE3_err WISE4 WISE4_err'+' \n'
        file.write(s)

        #for every "good" galaxy in flux_tab, add a row to the text file with relevant information
        for n in faux_table[south_flag]:

            s_gal = f"{n[0]} {n[1]} %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f"%(n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],n[12],n[13],n[14],n[15],n[16],n[17],n[18],n[19]) + '\n'
            file.write(s_gal)

        file.close()    
        
def create_ini_file(north_path, south_path):
    
    check_dir(north_path, south_path)    
    
    #create pcigale.ini files
    with open(north_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = vf_data_north.txt \n')
        file.write('parameters_file = \n')
        file.write('sed_modules = sfhdelayed, bc03, nebular, dustatt_modified_CF00, dl2014, redshifting \n')
        file.write('analysis_method = pdf_analysis \n')
        file.write('cores = 1 \n')
        file.close()    
    with open(south_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = vf_data_south.txt \n')
        file.write('parameters_file = \n')
        file.write('sed_modules = sfhdelayed, bc03, nebular, dustatt_modified_CF00, dl2014, redshifting \n')
        file.write('analysis_method = pdf_analysis \n')
        file.write('cores = 1 \n')
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
        
def run_all(Vcosmic_array, IDs, flux_tab, north_path, south_path, trim=True):

    create_ini_file(north_path, south_path)
    redshifts = get_redshift(Vcosmic_array)
    create_input_files(IDs, redshifts, flux_tab, north_path, south_path, trim)

        
if __name__ == "__main__":
    
    north_path = homedir+'/Desktop/cigale_vf_north'
    south_path = homedir+'/Desktop/cigale_vf_south'
    
    trim = True
    
    #load tables
    vf = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_environment.fits')
    flux_tab = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_legacy_ephot.fits')
    
    Vcosmic_array = vf['Vcosmic']
    IDs = vf['VFID']
    
    run_all(Vcosmic_array, IDs, flux_tab, north_path, south_path, trim)
    
        
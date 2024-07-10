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
        
def trim_tables(IDs, redshifts, flux_table):
    
    all_flags = (redshifts>0.) | (flux_table['photFlag'])
    return IDs[all_flags], redshifts[all_flags], flux_table[all_flags]
    
    
def create_fauxtab(IDs, redshifts, flux_table, trim=True):
    
    phot = flux_table
    
    if trim:
        IDs, redshifts, flux_table = trim_tables(IDs,redshifts,phot)
    
    #isolate needed phot fluxes; convert from nanomaggies to mJy
    #order: W1, W2, W3, W4, NUV, FUV, g, r

    fluxes = [phot['FLUX_AP06_W1']*3.631e-3, phot['FLUX_AP06_W2']*3.631e-3, phot['FLUX_AP06_W3']*3.631e-3,
             phot['FLUX_AP06_W4']*3.631e-3, phot['FLUX_AP06_NUV']*3.631e-3, phot['FLUX_AP06_FUV']*3.631e-3,
             phot['FLUX_AP06_G']*3.631e-3, phot['FLUX_AP06_R']*3.631e-3]

    flux_ivars = [phot['FLUX_IVAR_AP06_W1'], phot['FLUX_IVAR_AP06_W2'],
                 phot['FLUX_IVAR_AP06_W3'], phot['FLUX_IVAR_AP06_W4'],
                 phot['FLUX_IVAR_AP06_NUV'], phot['FLUX_IVAR_AP06_FUV'],
                 phot['FLUX_IVAR_AP06_G'], phot['FLUX_IVAR_AP06_R']]
    
    flux_errs = [np.zeros(len(phot)),np.zeros(len(phot)),np.zeros(len(phot)),np.zeros(len(phot)),
             np.zeros(len(phot)),np.zeros(len(phot)),np.zeros(len(phot)),np.zeros(len(phot))]

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
    
    #create table to organize results

    faux_table = Table([phot['VFID'],np.round(redshifts,4),fluxes[0],flux_errs[0],fluxes[1],flux_errs[1],
                   fluxes[2],flux_errs[2],fluxes[3],flux_errs[3],fluxes[4],flux_errs[4],
                   fluxes[5],flux_errs[5],fluxes[6],flux_errs[6],fluxes[7],flux_errs[7]],
                   names=['VFID','redshift','WISE1','WISE1_err','WISE2','WISE2_err','WISE3','WISE3_err',
                          'WISE4','WISE4_err','NUV','NUV_err','FUV','FUV_err','g','g_err','r','r_err'])
    return faux_table

def check_dir(north_path, south_path):
    
    if not os.path.isdir(north_path):
        os.mkdir(homedir+'/Desktop/cigale_vf_north')
        print('Created '+ homedir + '/Desktop/cigale_vf_north')
    if not os.path.isdir(south_path):
        os.mkdir(homedir+'/Desktop/cigale_vf_south')
        print('Created '+ homedir + '/Desktop/cigale_vf_south')

def create_input_files(IDs, redshifts, flux_table, north_path, south_path, trim=True):
    
    faux_table = create_fauxtab(IDs, redshifts, flux_table, trim=True)
    
    #I will need these flags as well. humph.
    north_flag = phot['DEC_MOMENT']>32
    south_flag = phot['DEC_MOMENT']<32
    
    #write files...
    
    check_dir(north_path, south_path)
    
    with open(homedir+'/Desktop/cigale_vf_north/vf_data_north.txt', 'w') as file:
        #create file header
        s = '# id redshift FUV FUV_err NUV NUV_err g g_err r r_err WISE1 WISE1_err WISE2 WISE2_err WISE3 WISE3_err WISE4 WISE4_err'+' \n'
        file.write(s)

        #for every "good" galaxy in phot, add a row to the text file with relevant information
        for n in faux_table[north_flag]:

            s_gal = f"{n[0]} {n[1]} %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f"%(n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],n[12],n[13],n[14],n[15],n[16],n[17]) + '\n'
            file.write(s_gal)

        file.close()    

    with open(homedir+'/Desktop/cigale_vf_south/vf_data_south.txt', 'w') as file:
        #create file header
        s = '# id redshift FUV FUV_err NUV NUV_err g g_err r r_err WISE1 WISE1_err WISE2 WISE2_err WISE3 WISE3_err WISE4 WISE4_err'+' \n'
        file.write(s)

        #for every "good" galaxy in phot, add a row to the text file with relevant information
        for n in faux_table[south_flag]:

            s_gal = f"{n[0]} {n[1]} %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f"%(n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],n[12],n[13],n[14],n[15],n[16],n[17]) + '\n'
            file.write(s_gal)

        file.close()    
        
def create_ini_file(north_path, south_path):
    
    check_dir(north_path, south_path)    
    
    #create pcigale.ini files
    with open(north_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = vf_data_north.txt \n')
        file.write('parameters_file = \n')
        file.write('sed_modules = sfh2exp, bc03, nebular, dustatt_modified_CF00, dl2014, redshifting \n')
        file.write('analysis_method = pdf_analysis \n')
        file.write('cores = 1 \n')
        file.close()    
    with open(south_path+'/pcigale.ini', 'w') as file:
        file.write('data_file = vf_data_south.txt \n')
        file.write('parameters_file = \n')
        file.write('sed_modules = sfh2exp, bc03, nebular, dustatt_modified_CF00, dl2014, redshifting \n')
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
        
def run_all(Vcosmic_array, IDs, flux_table, north_path, south_path, trim=True):

    create_ini_file(north_path, south_path)
    redshifts = get_redshift(Vcosmic_array)
    create_input_files(IDs, redshifts, flux_table, north_path, south_path, trim)

        
if __name__ == "__main__":
    
    north_path = homedir+'/Desktop/cigale_vf_north'
    south_path = homedir+'/Desktop/cigale_vf_south'
    
    trim = True
    
    #load tables
    vf = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_environment.fits')
    phot = Table.read(homedir+'/Desktop/v2-20220820/vf_v2_legacy_ephot.fits')
    
    Vcosmic_array = vf['Vcosmic']
    IDs = vf['VFID']
    flux_table = phot
    
    run_all(Vcosmic_array, IDs, flux_table, north_path, south_path, trim)
    
        
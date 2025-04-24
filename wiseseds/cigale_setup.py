#this script assumes that generate_input_files.py is successfully run, and that the user has activated their cigale (conda) environment

import os
homedir = os.getenv("HOME")

import sys
import fileinput
import re
from astropy.table import Table

def run_genconf(dir_path):
    os.chdir(dir_path)
    #will generate configuration files pcigale.ini and pcigale.ini.spec
    #can check/edit the various parameters in the file before the next step
    os.system('pcigale genconf')

def change_sedplot(dir_path):
    with open(dir_path+'/pcigale.ini', 'r') as file:
        lines = file.readlines()

    #modify the lines
    modified_lines = []
    for line in lines:
        if re.match(r'^\s*save_best_sed\s*=', line):
            modified_lines.append("save_best_sed = True\n")
            print('line changed: save_best_sed = True')
        else:
            modified_lines.append(line)

    #write the modified lines back to the file
    with open(dir_path+'/pcigale.ini', 'w') as file:
        file.writelines(modified_lines)

def add_params(dir_path,sed_plots=False,lim_flag='noscaling',nblocks=1, create_pdfs=False):
    
    #different modules, different naming schemes...
    #sfhdelayed --> age_main, age_burst
    #sfh2exp --> age, burst_age

    with open(dir_path+'/pcigale.ini','r') as file:
        lines = file.readlines()
        
    #modify lines...enjoy.
    modified_lines = []
    for line in lines:
        if re.match(r'^\s*save_best_sed\s*=', line):
            if sed_plots:
                modified_lines.append("  save_best_sed = True \n")
                print('line changed: save_best_sed = True')
            else:
                modified_lines.append(line)
        elif re.match(r'^\s*tau_main\s*=', line):
            modified_lines.append('   tau_main = 300, 500, 1000, 3000, 6000, 1e5 \n')
        elif re.match(r'^\s*age\s*=', line):
            modified_lines.append('   age = 1e3, 3e3, 5e3, 7e3, 1e4, 13000 \n') 
        elif re.match(r'^\s*age_main\s*=', line):
            modified_lines.append('   age_main = 1e3, 3e3, 5e3, 7e3, 1e4, 13000 \n') 
           
        elif re.match(r'^\s*age_main\s*=', line):
            modified_lines.append('   age = 1e3, 3e3, 5e3, 7e3, 1e4, 13000 \n')     
           
        elif re.match(r'^\s*tau_burst\s*=', line):
            modified_lines.append('   tau_burst = 100, 200, 400 \n')
        elif re.match(fr'^\s*burst_age\s*=', line):
            modified_lines.append('   burst_age = 20, 80, 200, 400, 800, 1e3 \n')
        elif re.match(fr'^\s*age_burst\s*=', line):
            modified_lines.append('   age_burst = 20, 80, 200, 400, 800, 1e3 \n')    

        elif re.match(r'^\s*f_burst\s*=', line):
            modified_lines.append('   f_burst = 0, 0.001, 0.005, 0.01, 0.05, 0.1 \n')
        elif re.match(r'^\s*imf\s*=', line):
            modified_lines.append('   imf = 1 \n')
        elif re.match(r'^\s*metallicity\s*=', line):
            modified_lines.append('   metallicity = 0.004, 0.02, 0.05 \n')
        elif re.match(r'^\s*variables\s*=',line):
            #modified_lines.append('  variables = sfh.sfr, stellar.m_star, sfh.burst_age, sfh.age, sfh.f_burst, sfh.tau_burst, sfh.tau_main, attenuation.Av_ISM, dust.alpha, dust.gamma, dust.qpah, dust.umean, dust.umin, dust.mass \n') 
            modified_lines.append('  variables = sfh.sfr, stellar.m_star, stellar.metallicity, sfh.burst_age, sfh.age, sfh.f_burst, sfh.tau_burst, sfh.tau_main, agn.fracAGN, attenuation.Av_ISM, dust.mass \n') 
        elif re.match(r'^\s*normalise\s*=',line):
            modified_lines.append('   normalise = True')
        elif re.match(r'^\s*Av_ISM\s*=',line):
            modified_lines.append('  Av_ISM = 0.0, 0.01, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.12, 0.15, 1.0, 1.3, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3 \n')
        elif re.match(r'^\s*fracAGN\s*=',line):
            modified_lines.append('  fracAGN = 0.0, 0.05, 0.1, 0.5 \n')
        elif re.match(r'^\s*umin\s*=',line):
            modified_lines.append('  umin = 1.0, 5.0, 10.0 \n')
        elif re.match(r'^\s*alpha\s*=',line):
            modified_lines.append('  alpha = 1.0, 2.0, 2.8 \n')
        elif re.match(r'^\s*gamma\s*=',line):
            modified_lines.append('  gamma = 0.02, 0.1 \n')  
        elif re.match(r'^\s*blocks\s*=',line):
            modified_lines.append(f'  blocks = {nblocks} \n')  
        elif re.match(r'^\s*lim_flag\s*=',line):
            modified_lines.append(f'  lim_flag = {lim_flag} \n')
        elif re.match(r'^\s*save_chi2\s*=',line):
            if create_pdfs:
                modified_lines.append("  save_chi2 = properties \n")
                print('line changed: save_chi2 = properties')
            else:
                modified_lines.append(line)            
        else:
            modified_lines.append(line)
    
    #write modified lines back into the file...or, rather, recreate the file with ALL lines, modified or otherwise
    with open(dir_path+'/pcigale.ini', 'w') as file:
        file.writelines(modified_lines)
        
def run_cigale(dir_path):

    os.chdir(dir_path)   #just in case...
    
    #once completed, this line will generate an 'out' directory containing 'results.fits', among other files.
    os.system('pcigale run')
    
def run_sed_plots(dir_path):
    
    os.chdir(dir_path)  #AGAIN, just in case...
    os.system('pcigale-plots sed')
    print(f'Find .pdf plots in {dir_path}/out/')
    
if __name__ == "__main__":

    if '-h' in sys.argv or '--help' in sys.argv:
        print("USAGE: %s [-dir_path (path to directory harboring relevant .txt and .ini files, no quotation marks)] %s [-sed_plots (indicates whether user would like the .pdf plots of each sed-fitted galaxy)] [-params (name of parameter.txt file, no single or double quotations marks)]")
    
    if '-dir_path' in sys.argv:
        p = sys.argv.index('-dir_path')
        dir_path = str(sys.argv[p+1])
        print(dir_path)
    
    if '-sed_plots' in sys.argv:
        sed_plots = True
    else:
        sed_plots = False

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
        nblocks = param_dict['nblocks']
        lim_flag = param_dict['lim_flag']
        path_to_repos = param_dict['path_to_repos']
        phot_table = param_dict['phot_table']
        len_tab = len(Table.read(path_to_repos+phot_table))
        
        #in order to save the probability distribution functions, ncores = nblocks = 1
        if bool(int(param_dict['create_pdfs'])):
            nblocks = 1
            ncores = 1
            print('Create PDFs set to True! nblocks = ncores = 1.')
    
    print('Configuring input text files...')
    run_genconf(dir_path)

    add_params(dir_path,sed_plots,lim_flag,nblocks,create_pdfs=bool(int(param_dict['create_pdfs'])))
    print('Executing CIGALE...')
    run_cigale(dir_path)
    print('CIGALE is Fin!')
    if sed_plots:
        
        print('Generating SED plots...')
        run_sed_plots(dir_path)
        
        print('Organizing output...')
        print('Removing SFH .fits files')
        os.chdir(dir_path+'/out/')
        os.mkdir('best_SED_models')
        if sed_plots:
            os.system('rm *best_model*.fits')
        os.system('mv *best_model* best_sed_models')
        os.system('rm *_SFH*')
        
        try:
            os.mkdir('PDF_fits')
            
            #4 for VFID (VFS), 5 for OBJID (WISESize)
            if len(str(len_tab))==4:
                formatted_strings = np.array([f"{num:04}" for num in np.arange(0,len_tab)])
                ID_prefix='VFID'
            if len(str(len_tab))==5:
                formatted_strings = np.array([f"{num:05}" for num in np.arange(0,len_tab)])
                ID_prefix='OBJID'
            
            #if I don't apply this loop (one mv command per OBJID), then there is a 
            #"too many arguments" error. I do not want a "too many arguments" error.
            for num in formatted_strings:           
                galID = f'{ID_prefix}{num}'
                os.system(f'mv {ID_prefix}*fits PDF_fits')
        except:
            pass
        
        print('SED Models are Fin!')
        
        
#this script assumes that generate_input_files.py is successfully run, and that the user has activated their cigale (conda) environment

import os
homedir = os.getenv("HOME")

import sys
import fileinput
import re

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

def add_params(dir_path,sed_plots=False):
    with open(dir_path+'/pcigale.ini','r') as file:
        lines = file.readlines()
        
    #modify lines...enjoy.
    modified_lines = []
    for line in lines:
        if re.match(r'^\s*save_best_sed\s*=', line):
            if sed_plots:
                modified_lines.append("save_best_sed = True\n")
                print('line changed: save_best_sed = True')
            else:
                modified_lines.append(line)
        elif re.match(r'^\s*tau_main\s*=', line):
            modified_lines.append('tau_main = 1, 300, 500, 800, 1000, 3000, 6000, 1e4, 1e5')
        elif re.match(r'^\s*age_main\s*=', line):
            modified_lines.append('age_main = 3e3, 5e3, 7e3, 1e4, 13000')
        elif re.match(r'^\s*tau_burst\s*=', line):
            modified_lines.append('tau_burst = 10, 20, 40, 80, 200, 400, 800, 1e3, 3e3')
        elif re.match(r'^\s*age_burst\s*=', line):
            modified_lines.append('age_burst = 10, 20, 40, 80, 200, 400, 800, 1e3, 3e3')
        elif re.match(r'^\s*f_burst\s*=', line):
            modified_lines.append('f_burst = 0, 0.001, 0.005, 0.01, 0.05, 0.1')
        elif re.match(r'^\s*imf\s*=', line):
            modified_lines.append('imf = 1')
        elif re.match(r'^\s*metallicity\s*=', line):
            modified_lines.append('metallicity = 0.004, 0.02, 0.05')
        elif re.match(r'^\s*variables\s*=',line):
            modified_lines.append('variables = sfh.sfr, stellar.m_star, sfh.age_burst, sfh.age_main, sfh.f_burst, sfh.tau_burst, sfh.tau_main')
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
        print("USAGE: %s [-dir_path (path to directory harboring relevant .txt and .ini files, no quotation marks)] %s [-sed_plots (indicates whether user would like the .pdf plots of each sed-fitted galaxy)]")
    
    if '-dir_path' in sys.argv:
        p = sys.argv.index('-dir_path')
        dir_path = str(sys.argv[p+1])
        print(dir_path)
    
    if '-sed_plots' in sys.argv:
        sed_plots = True
    else:
        sed_plots = False
    
    print('Configuring input text files...')
    run_genconf(dir_path)
    #if sed_plots:
    #    change_sedplot(dir_path)   #changes line to save_best_sed = True 
    add_params(dir_path,sed_plots)
    print('Executing CIGALE...')
    run_cigale(dir_path)
    print('CIGALE is Fin!')
    if sed_plots:
        print('Generating SED plots...')
        run_sed_plots(dir_path)
        print('Organizing output...')
        os.chdir(dir_path+'/out/')
        os.mkdir('SFH_outputs')
        os.mkdir('best_SED_models')
        os.system('mv *best_model* best_sed_models')
        os.system('mv *_SFH* SFH_outputs')
        print('SED Models are Fin!')
        
        
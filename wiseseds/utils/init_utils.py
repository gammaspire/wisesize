import os
import sys
import re
import numpy as np

#####################################################
# Names of each band in the cigale filter textfile! #
#####################################################
def define_flux_dict(n_or_s):
    
    if n_or_s not in ['n','s']:
        print('Please put "n" for north and "s" for south.')
        sys.exit()
    
    if n_or_s=='n':
        flux_dict_north = {'FUV':'FUV', 'NUV':'NUV', 'G':'BASS-g', 'R':'BASS-r',
                     'W1':'WISE1', 'W2':'WISE2', 'W3':'WISE3', 'W4':'WISE4'}
        return flux_dict_north
    
    flux_dict_south = {'FUV':'FUV', 'NUV':'NUV', 'G':'decamDR1-g', 'R':'decamDR1-r',
                     'Z':'decamDR1-z', 'W1':'WISE1', 'W2':'WISE2', 'W3':'WISE3', 'W4':'WISE4'}
    return flux_dict_south


##############
# Trim Table #
##############
def trim_tables(IDs, redshifts, flux_tab, ext_tab):
    '''
    trim flags according to redshift values (must be positive) and whether the galaxies contain photometry data
    '''
    all_flags = (redshifts>0.) & (flux_tab['photFlag'])
    
    return IDs[all_flags], redshifts[all_flags], flux_tab[all_flags], ext_tab[all_flags]


####################################################
# Check whether dir_path exists...create it if not #
####################################################
def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        print(f'Created {dir_path}')


######################################
# Add a SLEW of items to pcigale.ini #
######################################

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
            modified_lines.append('   normalise = True \n')
        
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
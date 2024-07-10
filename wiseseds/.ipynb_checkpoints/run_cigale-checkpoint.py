#this script assumes that generate_input_files.py is successfully run, and that the user has activated their cigale (conda) environment

import os
homedir = os.getenv("HOME")

import sys
import fileinput

def run_genconf(dir_path, sed_plots=False):
    os.chdir(dir_path)
    os.system('pcigale genconf')  #will generate configuration files pcigale.ini and pcigale.ini.spec
                                  #can check/edit the various parameters in the file before the next step
    
    #the pcigale genconf default save_best_sed to False...however, we need these files in order to generate plots!
    if sed_plots:
        
        for line in fileinput.input(dir_path, inplace=True):
            if line.startswith('save_best_sed'):
                #quite curious...the print statement is what the line will change to!
                print("save_best_sed = True")
            else:
                #this statement is included so that lines not including save_best_sed are ignored.
                #if I remove this line, in fact, the pcigale.ini file will ONLY contain the "save_best_sed = True" text.
                print(line.strip()) 
                
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
    
    if '-sed_plots' in sys.argv:
        sed_plots = True
    
    print('Configuring input text files...')
    run_genconf(dir_path, sed_plots)
    print('Fin!')
    print('Execuring CIGALE...')
    run_cigale(dir_path)
    print('Fin!')
    if sed_plots:
        print('Generating SED plots...')
        run_sed_plots(dir_path)
        print('Fin!')
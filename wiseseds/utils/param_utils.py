#create dictionary with keyword and values from param textfile

from astropy.table import Table
from conversion_utils import get_redshift

def read_params(param_file):

    param_dict={}
    with open(param_file) as f:
        for line in f:
            try:
                key = line.split()[0]
                val = line.split()[1]
                param_dict[key] = val
            except:
                continue
                
    return param_dict


#define a class...easier for me to organize parameters!
class Params():
    
    #################################################
    # extract parameters and assign to variables... #
    #################################################
    
    def __init__(self, param_file):
        
        param_dict = read_params(param_file)

        self.path_to_repos = param_dict['path_to_repos']
        self.main_table = param_dict['main_table']
        self.phot_table = param_dict['phot_table']
        self.extinction_table = param_dict['extinction_table']

        self.dir_path = param_dict['destination']

        self.id_col = param_dict['galaxy_ID_col']

        self.bands_north = param_dict['bands_north'].split("-")   #list!
        self.bands_south = param_dict['bands_south'].split("-")   #list!

        self.ncores = param_dict['ncores']
        self.nblocks = param_dict['nblocks']

        #in order to save the probability distribution functions, ncores = nblocks = 1
        if bool(int(param_dict['create_pdfs'])):
            self.ncores = 1
            self.nblocks = 1

        self.lim_flag = param_dict['lim_flag']
        
        self.sfh_module = param_dict['sfh_module']
        self.dust_module = param_dict['dust_module']
        
        self.Vcosmic_column = param_dict['Vcosmic_column']
        self.redshift_column = param_dict['redshift_column']
        
        self.flux_id_col = param_dict['flux_ID_col']
        self.flux_id_col_err = param_dict['flux_ID_col_err']
        
        self.extinction_col = param_dict['extinction_col']
        
        self.convert_flux = bool(int(param_dict['nanomaggies_to_mJy']))
        self.ivar_to_err = bool(int(param_dict['IVAR_to_ERR']))
        self.transmission_to_extinction = bool(int(param_dict['transmission_to_extinction']))
        
    ##################################################
    # class functions for loading tables and columns #
    ##################################################

    def load_tables(self):
        #load the tables
        self.main_tab = Table.read(self.path_to_repos + self.main_table)
        self.flux_tab = Table.read(self.path_to_repos + self.phot_table)
        self.ext_tab = Table.read(self.path_to_repos + self.extinction_table)
        
        #I need to isolate just one row. JUST one row.
        galaxy_flag = (self.ext_tab['maskbits']==4096)
        ext_tab = ext_tab[galaxy_flag][0]     #...this one. sure.

    def load_columns(self):
        self.IDs = self.main_tab[self.id_col]

        #if "vf" in the vcosmic_table name, then must be using Virgo catalogs...thus, Vcosmic column is available
        if 'vf' in self.phot_table:
            Vcosmic_array = self.main_tab[self.Vcosmic_column]
            redshifts = get_redshift(Vcosmic_array)

        #otherwise, just use the redshift column
        else:
            redshifts = self.main_tab[self.redshift_column]
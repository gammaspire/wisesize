'''
AIM: create halo mass predictions for all NED-LVS galaxies in the mass complete WISESize volume, using the Random Forest Regression model with parameters defined in rf_regression_parameters.txt.
'''

import numpy as np
import pandas as pd
import sys
import os
homedir=os.getenv("HOME")

from astropy.table import Table

from ML_M200_functions import rowmatch_to_catalog
from ML_M200_regression import parse_force_features, read_params, RFR_model


def create_volume_flag(parent_sample):
    '''
    Aim: return row-matched parent sample flag to isolate galaxies within the mass complete WISESize volume.
    '''
    #these flags will be important later on when creating the ML models!
    raflag = (parent_sample['RA']>87) & (parent_sample['RA']<300)
    decflag = (parent_sample['DEC']>-10) & (parent_sample['DEC']<85)
    mstarflag = parent_sample['Mstar_all_flag']
    zflag = (parent_sample['Z']>0.002) & (parent_sample['Z']<0.025)

    #these are ALL flags applied to the input catalog for Sigma_*
    volume_flag = (mstarflag) & (zflag) & (raflag) & (decflag)
    
    return volume_flag


def create_volume_sample(parent_sample, volume_flag):
    '''
    Aim: trim nedlvs_parent such that the only galaxies remaining are those withing the mass complete WISESize volume.
    '''

    #remember the .npy files of features and labels? we need them now.
    SigmaM_features = np.load(homedir+'/Desktop/SigmaMfeatures.npy', allow_pickle=True).tolist()
    SigmaM_names = np.load(homedir+'/Desktop/SigmaMnames.npy', allow_pickle=True).tolist()
    Sigmak_features = np.load(homedir+'/Desktop/Sigmakfeatures.npy', allow_pickle=True).tolist()
    Sigmak_names = np.load(homedir+'/Desktop/Sigmaknames.npy', allow_pickle=True).tolist()

    #row match the above feature columns to parent galaxy catalog
    parent_sample_ = rowmatch_to_catalog(parent_sample, SigmaM_features, SigmaM_names, Sigmak_features, Sigmak_names)

    print(f'{len(parent_sample_)} galaxies in NEDLVS sample.')

    #apply the wisesize volume + mass completeness flags
    wisesize_volume = parent_sample_[volume_flag]
    print(f'{len(wisesize_volume)} galaxies in volume-limited NEDLVS sample.')

    #and then...convert to dataframe
    df_vol = wisesize_volume.to_pandas()

    #replace -999 with NaNs
    df_vol.replace(-999, np.nan, inplace=True)

    #REMOVE NaNs
    df_vol = df_vol.dropna()
    print(f'{len(wisesize_volume) - len(df_vol)} galaxies removed after dropping NaN rows.')
    print(f'Final count: {len(df_vol)} galaxies.')

    #save progress...just in case.
    df_vol.to_csv(homedir+'/Desktop/ML_features_wisesizevolume.csv', index=False)
    print('Dataframe contains galaxies in the mass complete WISESize volume is created.')
    
    return df_vol


def volume_sample_x(df_vol, feature_list):
    X_vol = df_vol[feature_list]    
    return X_vol


def make_m200_model(df, feature_names, param_dict):
    '''
    Aim: generate the M200 model using the original Tempel+2017 galaxies within mass complete WISESize volume.
    '''
    
    #generate the original model created with the Tempel+2017 galaxies in the mass complete WISESize volume
    _, model, _, _ = RFR_model(df=df, feature_list=feature_names, use_pca=bool(int(param_dict['use_pca'])), 
                              use_optimal_features=bool(int(param_dict['use_optimal_features'])), 
                              pca_threshold=float(param_dict['correlation_threshold']), 
                              Ngal_threshold=float(param_dict['Ngal_threshold']), test_size=float(param_dict['test_size']), 
                              n_trees=int(param_dict['n_trees']), max_depth=int(param_dict['max_depth']), 
                              random_state=int(param_dict['random_state']),
                              bin_width=float(param_dict['bin_width']), threshold_width=float(param_dict['threshold_width']),
                              min_bin_count=float(param_dict['min_bin_count']), method=str(param_dict['method']),
                              regression_plot=False, importances_plot=False)
    
    return model


def add_parent_m200(parent_sample, df_vol):
    '''
    Aim: map the predicted logM200 values from the mass complete WISESize volume sample to the parent sample.
        * this mapping uses the OBJNAME column to determine which logM200 belongs to which galaxy
        * all non-df_vol galaxies will receive logM200=NaN
    '''

    #in case the column is byte or bytearray type, convert to str
    df_vol['OBJNAME'] = df_vol['OBJNAME'].astype('str')
    parent_sample['OBJNAME'] = parent_sample['OBJNAME'].astype('str')

    #helper function to convert byte string ("b'VFID0000'") to a proper string ("VFID0000")
    def decode_if_bytes(x):
        x = x.replace('b','')
        x = x.replace("'","")
        return x
    
    #convert all byte strings to strings, if applicable.
    df_vol['OBJNAME'] = df_vol['OBJNAME'].apply(decode_if_bytes)
    parent_sample['OBJNAME'] = parent_sample['OBJNAME'].apply(decode_if_bytes)

    #map predicted logM200 values from df_vol onto the parent sample
    parent_m200 = parent_sample.merge(df_vol[['OBJNAME','logM200_model']], on='OBJNAME', how='left')
    
    #return the new catalawg
    return parent_m200


#need if loading script as a module
def run_all(df, parent_sample, param_dict):
    '''
    Aim: run all of the functions from model generation to adding the logM200 column to the parent sample.
    '''
    
    #pull feature names from param_dict. might be [] (empty list), and if so default to reading from .npy files.
    feature_names = parse_force_features(param_dict)

    #generate the original model created with the Tempel+2017 galaxies in the mass complete WISESize volume
    model = make_m200_model(df, feature_names, param_dict)
    
    #now that we have the model, create the volume-limited dataframe
    df_vol = create_volume_sample(parent_sample, create_volume_flag(parent_sample))
    
    #isolate the X_vol data for model predictions
    X_vol = volume_sample_x(df_vol, feature_names)
    
    #grab the predicted logM200 for the volume-limited dataframe, using the original model...
    y_vol_pred = model.predict(X_vol)
    
    #add the predictions to the dataframe as a separate column
    df_vol['logM200_model'] = y_vol_pred
    
    parent_m200 = add_parent_m200(parent_sample, df_vol)
    
    return parent_m200


#need if running script
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ML model to predict log(M200) or environment class.")
    
    default_params_path = homedir+'/github/wisesize/ML_project/rf_regression_parameters.txt'
    
    parser.add_argument("-df", type=pd.core.frame.DataFrame, default=None, help="Input features/class pandas dataframe; defaults to looking for relevant file(s) on Desktop.")
    parser.add_argument("-features", type=list, default=None, help="Input list of feature column names; defaults to looking for relevant file(s) on Desktop.")
    parser.add_argument("-params", type=str, default=default_params_path, help="Input parameters for RF regression model and setup.")
    
    args = parser.parse_args()
    
    #create dictionary with keyword and values from param textfile...
    param_dict = read_params(args.params)
    
    df_path = param_dict['df_path']
    parent_sample_path = param_dict['df_path']
    
    parent_sample = Table.read(parent_sample_path).to_pandas()
    
    try:
        df = pd.read_csv(homedir+df_path)
    except:
        print('df not found. please generate the .csv file first before running.')
        sys.exit()
    
    parent_m200 = run_all(df, parent_sample, param_dict)
    
    save_path=homedir+'/Desktop/nedlvs_logm200.csv'
    parent_m200.to_csv(save_path, index=False)
    print(f'parent_m200 is now saved to {save_path}.')
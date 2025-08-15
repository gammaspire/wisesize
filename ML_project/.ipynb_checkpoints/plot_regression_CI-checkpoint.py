'''
AIM: Generate a plot of the 68% confidence interval outputs after running ML regression x times with different feature combinations. 

PROCEDURE:
* begin with full feature set
* run ML regression
* generate confidence intervals using the given bin_width
* save list of confidence interval width (high_ci - low_ci) and the corresponding number of features
* grab model feature importances using importances = model.feature_importances_
* isolate the highest int(N_full_features/2), or desired number of, features
* repeat beginning at step two
* once N_output_all is exhausted, plot medians and CI_widths (separately)

OUTPUT: figure showing the distribution of confidence intervals in each bin (of width bin_width), color-coded by the number of features
'''

from ML_M200_regression import *
import numpy as np
from rich import print
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
import pandas as pd
import sys
import os
homedir=os.getenv("HOME")


def read_df(param_dict):
        
    df_path = param_dict['df_path']
    try:
        df = pd.read_csv(homedir+df_path)
        return df
    except:
        print('df not found. please generate the .csv file first before running.')
        sys.exit()

        
#need to begin with the full list of features, which should already be saved in the .npy files on my Desktop.
def read_full_features():
    try:
        full_feature_list = read_features()
        return full_feature_list
    except:
        print('Bzzt. One or more feature files not found. Please generate the required .npy files first before running.')
        sys.exit()


def run_ML_regression(df, feature_names, param_dict):
    
    #will need this twice; may as well define a variable
    bin_width=float(param_dict['bin_width'])
    
    #just a basic little with no PCA features, no optimally-chosen features, no plots, and the parameters given in params.txt
    _, model, y_test, y_pred = RFR_model(df=df, feature_list=feature_names, use_pca=False, use_optimal_features=False, 
                                                  logM200_threshold=float(param_dict['logM200_threshold']), 
                                                  test_size=float(param_dict['test_size']), n_trees=int(param_dict['n_trees']), 
                                                  max_depth=int(param_dict['max_depth']), bin_width=bin_width, 
                                                  threshold_width=float(param_dict['threshold_width']),
                                                  regression_plot=False, importances_plot=False)
    
    
    #generate the confidence intervals in each bin...
    bin_centers, stats = confidence_intervals(y_test, y_pred, bin_width=bin_width)
    
    #isolate the CI_width for each bin
    CI_widths = stats['ci_width'].values
    
    #also isolate medians
    medians = stats['median'].values
    
    #return the lists
    return medians, CI_widths, model


def isolate_important_features(X_features, model, N_output):
    '''
    Find and create list of top feature importances for the input model
    
    Parameters:
    * X_features : features dataframe used to generate the model
    * model : the ML model used
    * N_output : integer indicating number of top features to output
    
    Returns:
    * list of N_output feature column names
    '''
    
    #create list of columns
    features = X_features.columns
    
    if len(features)<=int(N_output):
        print('The length of X_features is smaller than or the same as N_output. Defaulting to X_features column names.')
        return features
    
    #create list of row-matched importances to these columns
    importances = model.feature_importances_
    
    #generate indices to sort features and importances from least-to-most important
    sort_indices = np.argsort(importances)
    
    #organize the features
    features = features[sort_indices]
    
    #isolate the top N_output features
    features_trim = features[-int(N_output):]
    
    #and lastly, return the features list
    return features_trim


def plot_CI(median_list, CI_width_list, N_outputs_all):
    
    from matplotlib.ticker import MultipleLocator
        
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    plt.subplots_adjust(wspace=0.4)
    ax1, ax2 = axes.flat
    
    #I want the ticks to be integers with a step size of 1
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    
    for i, N in enumerate(N_outputs_all):
        
        bin_numbers = np.arange(len(median_list[i]))+1
        
        ax1.plot(bin_numbers, median_list[i], lw=3, alpha=0.5, label=f'{N} Important Feature(s)')
        ax2.plot(bin_numbers, CI_width_list[i], lw=3, alpha=0.5, label=f'{N} Important Feature(s)')
        
    ax1.set_title('Median "True" log(M200) per "Pred" log(M200) Bin', fontsize=12)
    ax2.set_title('68% CI Widths [High-Low] Per Bin', fontsize=12)
    
    ax1.set_xlabel(f'Predicted log(M200) Bin Number (Step Size = 0.3)',fontsize=12)
    ax2.set_xlabel(f'Predicted log(M200) Bin Number (Step Size = 0.3)',fontsize=12)
    
    ax1.set_ylabel("Median 'True' log(M200)")
    ax2.set_ylabel("68% CI Width")
    
    ax1.grid(True)
    ax2.grid(True)
    
    ax1.legend(fontsize=9)
    #ax2.legend(fontsize=9)
    
    fig.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    
    #initialize some crucial variables. so crucial, in fact, that this code is nonsensical without them.
    
    param_dict = read_params(homedir+'/github/wisesize/ML_project/rf_regression_parameters.txt')
    full_feature_list = read_full_features()
    df = read_df(param_dict)
    model = None    #need to set as None initially for the first pass of isolate_important_features()
    
    #will also need...lists of lists for the medians and widths!
    medians_list = []
    CI_widths_list = []
    
    #begin will the full list of features...
    X_features = df[full_feature_list]
    
    #list of N_output values to use when isolating the most important features for the ML model
    N_outputs_all = [len(full_feature_list), int(len(full_feature_list)/2), 6, 5, 3, 1]
    
    for i, N_output in enumerate(N_outputs_all):
        
        print(f'Using {N_output} "important" feature(s)...')
        
        feature_names = isolate_important_features(X_features, model, N_output)
        
        medians, CI_widths, model = run_ML_regression(df, feature_names, param_dict)
        
        medians_list.append(medians)
        
        CI_widths_list.append(CI_widths)
        
        X_features = df[feature_names]
        
    #and now, the plot:
    plot_CI(medians_list, CI_widths_list, N_outputs_all)
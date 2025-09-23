# GOAL: create cornerplot for each preselected "single" parameter in ML_features.csv vs. Tempel log(M200).
# The chosen parameters are dictated by the rf_regression_parameters.txt file in the ML_project directory

from ML_M200_regression import confidence_intervals, read_params, parse_force_features, RFR_model

import sys
import numpy as np
from rich import print
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

import os
homedir=os.getenv("HOME")


#read in dataframe
def read_df(param_dict):

    df_path = param_dict['df_path']
    try:
        df = pd.read_csv(homedir+df_path)
        return df
    except:
        print('df not found. please generate the .csv file first before running.')
        sys.exit()


def clean_the_df(df):
    
    #convert M200 to log(M200)
    #I figure that 1e5 is a "safe" number to evaluate whether the M200 values are already logscale :-)
    if np.max(df['group_M200'])>1e5:
        df['group_M200'] = np.log10(df['group_M200'])
    
    #replace -999 with NaNs, in case there are any lingering buggers
    df.replace(-999, np.nan, inplace=True)
        
    #REMOVE NaNs
    df_clean = df.dropna()
    
    return df_clean 


#generic running of RFR model to get y_test and y_pred log(M200)
def get_ytest_ypred(df, feature_names, param_dict, random_state=42):
    
    _, _, y_test, y_pred = RFR_model(df=df, 
                                     feature_list=feature_names, 
                                     use_pca=False, 
                                     use_optimal_features=False, 
                                     Ngal_threshold=float(param_dict['Ngal_threshold']), 
                                     test_size=float(param_dict['test_size']), 
                                     n_trees=int(param_dict['n_trees']), 
                                     max_depth=int(param_dict['max_depth']), 
                                     bin_width=0.3, 
                                     threshold_width=float(param_dict['threshold_width']),
                                     regression_plot=False, importances_plot=False, 
                                     random_state=random_state)

    return y_test, y_pred


def get_single_stats(df, xlabel, ylabel):
    
    #generate the confidence intervals in each bin...
    bin_centers, stats = confidence_intervals(df[xlabel], df[ylabel], bin_width=0.5)
    bin_centers = np.asarray(bin_centers)

    #isolate the CI_width for each bin
    CI_widths = stats['ci_width'].values

    #also isolate medians
    medians = stats['median'].values

    #filter out bins with very few galaxies...
    mask = (stats['count']>=5)
    
    return bin_centers, stats, mask


def get_RFR_stats(y_test, y_pred, bin_width=0.3):
    
    #generate the confidence intervals in each bin...
    bin_centers_model, stats_model = confidence_intervals(y_test, y_pred, bin_width=bin_width)
    bin_centers_model = np.asarray(bin_centers_model)

    #filter out bins with very few galaxies...
    mask_model = (stats_model['count']>=5)
    
    return bin_centers_model, stats_model, mask_model


#fit linear function to predict log(M200) from a single parameter
def get_feature_fit_predictions(df, feature, return_xy_array=False, invert_for_plot=False,
                               print_=False):
    """
    Fits a linear regression of log(M200) vs. feature using numpy and returns predicted log(M200)
    in the same format as RFR model predictions.
    
    return_xy_array : bool --> additionally outputs the x, y arrays for plotting
    invert_for_plot : bool --> if True, returns arrays in (logM200, feature) order
                               for use in cornerplot overlays. Need to invert axes. :)
    """
    df_clean = clean_the_df(df)
    
    X = df_clean[[feature]].values.flatten()
    y = df_clean['group_M200'].values   # log(M200)
    
    #fit a straight line y = slope * X + intercept
    slope, intercept = np.polyfit(X, y, 1)
    
    #generate predictions
    y_true_feature = y.copy()
    y_pred_feature = slope * X + intercept
    
    if print_:
        print(f'{feature}: slope={slope:.3f}, intercept={intercept:.3f}')
    
    if return_xy_array:
        x_vals = np.linspace(X.min(), X.max(), 200)
        y_vals = slope * x_vals + intercept
        
        if invert_for_plot:
            #solve for feature as a function of log(M200) for plotting
            xx = np.linspace(y.min(), y.max(), 200)
            yy = (xx - intercept) / slope
            return y_true_feature, y_pred_feature, xx, yy
        
        return y_true_feature, y_pred_feature, x_vals, y_vals
    
    return y_true_feature, y_pred_feature


#generate cornerplot
def create_cornerplot(df, feature_names):
    import warnings
    warnings.filterwarnings("ignore")
    
    #add log(M200) to list of feature names...
    feature_names = ['group_M200'] + feature_names
    
    #remove NaNs, convert M200 to logM200 if not already done.
    df_clean = clean_the_df(df)
        
    #create the cornerplot! 
    #filter out galaxies with isophote=0
    p = sns.pairplot(df_clean[feature_names], corner=True, height=2, kind='hist', diag_kind='kde')

    #pull the axes from the corner plot
    axes_flat = [ax for ax in p.axes.flat if ax is not None]
    
    #each diagonal (axis 0, 2, 5, 9, ...) shows the variable ellipse_tab[colname]
    #that colname is index 0, 1, 2, 3, ... as you go down the diagonal
    #this dictionary converts the axis number of the diagonal with the index of the corresponding colname
    ax_dict={0:'0', 2:'1', 5:'2', 9:'3', 14:'4', 20:'5', 27:'6'}

    for i, ax in enumerate(axes_flat):
        
        if i in [0,2,5,9,14,20,27]:   #omit histograms; note that hidden axes are None but also have an index
            continue
        
        #grab labels (which are, in fact, the column names!)
        x_label = ax.get_xlabel()
        y_label = ax.get_ylabel()

        #overlay straight-line fit with 1-sigma spread ONLY when one axis is log(M200)
        if x_label == 'group_M200':
            
            #get true and predicted values from single-parameter linear fit
            y_true_feature, y_pred_feature, line_x, line_y = get_feature_fit_predictions(df_clean, y_label, return_xy_array=True, invert_for_plot=True, print_=True)
            
            #compute 1-sigma scatter around the fit
            sigma = np.std(y_true_feature - y_pred_feature)
            
            #plot linear fit line
            ax.plot(line_x, line_y, color='black', lw=2, label=f'Linear Fit')
            
            #plot 1-sigma shaded region
            ax.fill_between(line_x, line_y - sigma, line_y + sigma, color='crimson', alpha=0.3, 
                            label=f'$\pm1\sigma$ {sigma:.2f}')

            ax.set_xlabel('log(M200)')
            ax.set_xlim(7.9,15.1)
            ax.set_ylim(7.9,12.7)
            ax.set_xticks(np.arange(8,15.1,1))
            
            ax.legend(fontsize=6.5)
            
        ax.grid(alpha=0.3)


def plot_CI_widths(df, feature_names, y_test, y_pred):
    
    df_clean = clean_the_df(df)
    
    plt.figure(figsize=(6, 5))

    #SINGLE-PARAMETER FUNCTION FITS
    names=[r'log$\Sigma_{M0}$', r'log$\Sigma_{M2}$', r'log$\Sigma_{M5}$', r'log$\Sigma_{M7}$', r'log$\Sigma_{M15}$', r'$\Sigma_2$', r'$\Sigma_5$', r'$\Sigma_{10}$', r'M$_*$', r'Ngal$_0$', r'Ngal$_2$', r'Ngal$_5$', r'Ngal$_7$']
    
    for i, feature in enumerate(['log_Sigma_M0','log_Sigma_M2','log_Sigma_M5','log_Sigma_M7','log_Sigma_M15','Sigma_2','Sigma_5','Sigma_10','Mstar','Sigma_ngal_0','Sigma_ngal_2','Sigma_ngal_5','Sigma_ngal_7']):
        
        y_true_feature, y_pred_feature = get_feature_fit_predictions(df_clean, feature)
        
        bin_centers, stats = confidence_intervals(y_true_feature, y_pred_feature, bin_width=0.5)
        bin_centers = np.asarray(bin_centers)
        mask = (stats['count']>=5)
        
        CI_width = stats['high'] - stats['low']
        
        plt.plot(bin_centers[mask], CI_width[mask], alpha=0.5, ls='--', label=names[i])
        
    #RFR MODEL
    bin_centers_model, stats_model, mask_model = get_RFR_stats(y_test, y_pred, bin_width=0.5)
    
    CI_width_model = stats_model['high'] - stats_model['low']
    
    plt.xlabel('Predicted log(M200) [dex]',fontsize=14)
    plt.ylabel('68% Confidence Interval Width [dex]',fontsize=14)
    
    plt.plot(bin_centers_model[mask_model], CI_width_model[mask_model], label='RFR Model', color='purple', lw=2)
    
    plt.legend(fontsize=8)
    
    plt.show()
                

if __name__ == "__main__":
    
    #initialize some crucial variables. so crucial, in fact, that this code is nonsensical without them.
    param_dict = read_params('rf_regression_parameters.txt')
    
    feature_names = parse_force_features(param_dict)
    
    df = read_df(param_dict)
    
    y_test, y_pred = get_ytest_ypred(df, feature_names, param_dict)
    
    create_cornerplot(df, feature_names)
    plot_CI_widths(df, feature_names, y_test, y_pred)
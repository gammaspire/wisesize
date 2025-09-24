'''
RandomForestRegression script for generating ML models to predict M200
'''

#this will not be code to import, so I am free to use name...main, etc.

from ML_M200_functions import *

import argparse
from rich import print
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from astropy.table import Table
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.stats import binned_statistic
import sys

import os
homedir=os.getenv('HOME')


def confidence_intervals(y_test, y_pred, ngal=None, bin_width=0.3, method='fixed'):
    '''
    INPUTS:
        y_test: true values
        y_pred: predicted values
        ngal: optional mask array for number of galaxies (filters data if provided)
        bin_width: width of bins for fixed method
        method: 'fixed' for equal-width bins (pd.cut), 
                'window' for bins centered on each galaxy (sliding window)
    '''
    
    #define new dataframe using y_test and y_pred
    df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    
    if method == 'fixed':
        #define bins - these bins will pool the data for which we calculate confidence intervals
        bins = np.arange(y_pred.min(), y_pred.max() + bin_width, bin_width)
        
        #cut the predicted log(M200) into the bins (intervals of width bin_width)
        df['bin'] = pd.cut(df['y_pred'], bins=bins)
        
        #group by bin, compute the median and 95% confidence interval of y_test (the true log(M200))
        #how to calculate percentiles: begin at 50; split CI in half (47.5), 50 +/- 47.5 are the percentiles 
        #(97.5 for high and 2.5 for low)
        #count is the number of samples in the bin
        #note: observed=False is needed to retain intended functionality in case pandas is updated
        stats = df.groupby('bin', observed=False)['y_true'].agg([
            ('median', 'median'),
            ('low', lambda x: np.percentile(x, 16) if len(x) > 0 else np.nan),   #for 95th percentile, 2.5; ignores bins with 0
            ('high', lambda x: np.percentile(x, 84) if len(x) > 0 else np.nan),  #for 95th percentile, 97.5; ignores bins with 0
            ('count', 'count')])    

        #get bin centers, just in case
        bin_centers = [interval.mid for interval in stats.index]

        #include the width of each CI bin. helpful for diagnosing where the width is small vs. large vs. whatever
        stats['ci_width'] = stats['high'] - stats['low']

        #drop empty/NaN rows
        stats_clean = stats.dropna(subset=['median'])    
        bin_centers = [interval.mid for interval in stats_clean.index]   #middle of distribution...

        #stats contains the median 
        return bin_centers, stats_clean
    
    elif method == 'window':

        #initialize some empty lists
        #'sliding window bins'
        bin_centers, medians, lows, highs, counts = [], [], [], [], []
        
        sorted_df = df.sort_values(by='y_pred')
        
        #for every y_pred in our sample...
        #select all y_true values within the window around this bin center (+/- bin_width/2)
        for yp in sorted_df['y_pred']:
            #select all y_true values for galaxies within ±bin_width of this y_pred
            mask = (sorted_df['y_pred'] >= yp - bin_width) & (sorted_df['y_pred'] <= yp + bin_width)
            y_window = sorted_df.loc[mask, 'y_true']

            if len(y_window) > 0:
                bin_centers.append(yp)
                medians.append(np.median(y_window))
                lows.append(np.percentile(y_window, 16))
                highs.append(np.percentile(y_window, 84))
                counts.append(len(y_window))
        
        #assemble DataFrame in same format as fixed. stats are automatically clean since we filter out 
        #bins with no galaxies
        stats_clean = pd.DataFrame({
            'median': medians,
            'low': lows,
            'high': highs,
            'count': counts})
        
        stats_clean['ci_width'] = stats_clean['high'] - stats_clean['low']
        
        return bin_centers, stats_clean
    
    else:
        raise ValueError("The method arg must be 'fixed' or 'window'.")
    

def plot_regression(y_test, y_pred, ngal, bin_centers=None, stats=None, threshold_width=0.5, min_bin_count=10, print_=False):
    '''
    INPUT: 
    y_test : true log(M200)
    y_pred : model-predicted log(M200)
    bin_centers of distribution of y_test at binned y_pred 
    stats : stats of distribution of y_test at binned y_pred values to determine confidence interval
    threshold_width : at or above which the code will flag said bin(s) as where the model is an unreliable log(M200) predictor
    min_bin_count : minimum number of galaxies that must exist in a bin for it to be plotted
    print_ : Whether user would like printed outputs of counts per bin, (bins, CI_widths), and the fraction of bins with CI_widths above the a range of threshold values. Not recommended for 'running' CIs.
    OUTPUT: figure of y_pred vs. y_true, with a 1-to-1 line plotted for ease of comparison
    Note: BE SURE NGAL IS ROW-MATCHED WITH Y_TEST AND Y_PRED!
    '''
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    #y_pred=y_pred[ngal>3]
    #y_test=y_test[ngal>3]
    #ngal=ngal[ngal>3]
    
    cmap = ListedColormap(['#440154', '#31688e', '#35b779', '#fde725', '#ff7f0e'])
    
    #create a cute little quantized colorbar
    bins = [2, 4, 7, 12, 20, np.max(ngal) + 1]  #added 1 to ensure the last value is included
    labels = ['2–3', '4–6', '7–11', '12–19', '20+']
    norm = BoundaryNorm(bins, cmap.N)   #maps ngal values to colors according to which bin they fall into
    
    #ensures tick marks of colorbar are at the centers of each color bin!
    bin_centers_cb = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    plot=ax.scatter(y_test, y_pred, s=10, c=ngal, cmap=cmap, norm=norm)
    ax.plot([y_pred.min(), y_pred.max()],
             [y_pred.min(), y_pred.max()], 'k--', lw=2)
    
    ax.set_xlabel(r"Tempel+2017 log(M$_{200T}$) [dex]",fontsize=14)
    ax.set_ylabel(r"Predicted log(M$_{200}$) [dex]",fontsize=14)
    ax.grid(True, alpha=0.5)
    
    cbar = fig.colorbar(plot, boundaries=bins, ticks=bin_centers_cb)
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Ngal in Group/Cluster',fontsize=14)
    
    #confidence intervals
    if bin_centers is not None:
        
        stats_filtered = stats[stats['count'] >= min_bin_count]
        bin_centers_filtered = np.array(bin_centers)[stats['count'] >= min_bin_count]
        
        #assumes bin_centers_filtered and stats_filtered are ordered by increasing y_pred
        mask = stats_filtered['ci_width'].values <= threshold_width
        
        if print_:
            print("Counts per bin:", stats['count'].values)        
            print("Bins and CI widths:", list(zip(np.round(bin_centers_filtered,2), 
                                                  np.round(stats_filtered['ci_width'].values,2))))
            print()

            for w in [0.5, 1.0, 1.50]:
                frac = np.mean(stats_filtered['ci_width'].values <= w)
                print(f"Fraction of bins with CI width ≤ {w:.2f}: {frac:.2%}")
            print()
        
        ax.fill_betweenx(y=bin_centers_filtered, x1=stats_filtered['low'], x2=stats_filtered['high'], 
                         color='gray', alpha=0.3, label=r'1$\sigma$ confidence intervals')
        
        ax.plot(stats_filtered['median'], bin_centers_filtered, color='black', lw=2, 
                label = r'Median logM$_{200T}$ per' + '\n' + r'predicted log(M$_{200}$) bin')
       
    ax.set_xlim(7.9,15)
    ax.set_ylim(10.2,15)
    
    ax.legend(loc='upper left',fontsize=10)
    
    fig.tight_layout()
    plt.show()


def RFR_model(df=None, feature_list=None, use_pca=True, use_optimal_features=False,
              pca_threshold=0.90, Ngal_threshold=0, regression_plot=True, importances_plot=True,
              test_size=0.3, n_trees=200, random_state=42, max_depth=10, threshold_width=0.5, 
              bin_width=0.1, method='fixed', min_bin_count=10):
    '''
    Train and evaluate a Random Forest Regressor to predict halo mass (log(M200)) for galaxy groups.

    Parameters:
    -----------
    df : pandas.DataFrame, optional
        Input dataframe. If None, attempts to load from a default CSV file.
    feature_list : list of str, optional
        List of feature column names. If None, attempts to load from .npy files on disk.
    use_pca : bool, default=True
        Whether to reduce features using PCA with specified correlation threshold.
    use_optimal_features : bool, default=False
        If True, selects optimal feature subset via repeated cross-validation.
    pca_threshold : float, default=0.90
        Correlation threshold used for PCA feature reduction (if use_pca is True).
    Ngal_threshold : int
        Excludes galaxies in groups with Ngal below this value from R^2, MSE calculations; set to 0 for "no threshold"
    regression_plot : bool, default=True
        Whether to display a scatter plot of predicted vs. true log(M200).
    importances_plot : bool, default=True
        Whether to show a bar chart of feature importances.
    test_size : float, default=0.3
        Fraction of data used for testing the model.
    n_trees : int, default=200
        Number of trees in the Random Forest.
    max_depth : int, default=10
        Maximum depth of each tree in the Random Forest.
    random_state : int, default=42
        The 'random state' for the train/test sample. Keep same to reproduce same sample or change to gauge model consistency
    width_threshold : float, default=0.5
        95% confidence interval width threshold for y_true distribution for every y_pred bin, 
        at or above which the ML's predictive power is unreliable.
    bin_width : float, default=0.1
        predicted log(M200) bin size within which distribution of y_true is evaluated
    method : str, default = 'fixed'
        indicates whether confidence intervals are calculated using fixed ('fixed') or adaptive ('adaptive') binning
    min_bin_count : int, default = 10
        the minumum number of items per bin under which the bin will be discarded  

    Returns:
    --------
    None
        Prints MSE and R² score, and optionally displays plots.
    '''
    if df is None:
        try:
            df = pd.read_csv(homedir+'/Desktop/ML_features.csv')
        except:
            print('No input df and ML_features.csv file not found. Exiting.')
            sys.exit()
            
    if feature_list is None:
        try:
            feature_list = read_features()
            
        except:
            print('No input feature list and .npy files not found. Exiting.')
            sys.exit()
        
    df_out = df.copy()
    if use_pca:
        df_out, feature_list = get_pca_features(df, feature_list, threshold)
        
    #add stellar mass to feature list; remove whatever 0.1 Mpc malarkey I thought was appropriate
    #feature_list=feature_list+['Mstar']
    
    #note: using halo mass, so must isolate Tempel+2017 group galaxies (which actually have a halo mass)
    df_group = df_out.copy()[df_out['tempel2017_groupIDs']>0]
    
    #convert M200 to log(M200)
    #I figure that 1e5 is a "safe" number to evaluate whether the M200 values are logscale :-)
    if np.max(df_group['group_M200'])>1e5:
        df_group['group_M200'] = np.log10(df_group['group_M200'])
    
    #replace -999 with NaNs, in case there are any lingering buggers
    df_group.replace(-999, np.nan, inplace=True)

    #REMOVE NaNs
    df_group = df_group.dropna()
    
    #define your model of choice
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=random_state)    
    
    #features...some hodgepodge of potentially relevant properties.
    X=df_group[feature_list]

    #target variable...log(M200)
    y=df_group['group_M200']  
    

    if use_optimal_features:
        
        if optimal_features_list != '[]':
        
            #find optimal feature set for the model
            selected_features = average_model_score(X, y, model=model, scoring='r2',regression=True,
                         n_splits=5, n_repeats=10, random_state=1)
        
        else:
            selected_features = optimal_features_list

        #predictors...some hodgepodge of potentially relevant properties.
        X=df_group[selected_features]
        
    
    #split dataset; include X_ngal for when regression_plot=True
    X_ngal = df_group['tempel2017_Ngal']
    X_train, X_test, y_train, y_test, ngal_train, ngal_test = train_test_split(X, y, X_ngal, 
                                                                               test_size=test_size, 
                                                                               random_state=63)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    #performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('_________________________')
    print(f'Model diagnostics for all group galaxies:')
    print(f"R²: {r2:.3f} (R: {np.sqrt(r2):.3f})")
    print(f"MSE: {mse:.3f}")
    print('_________________________')
    
    if Ngal_threshold>0:
        
        #apply flag [y_test>Ngal_threshold] to only evaluate how well the model performs for these galaxies!
        flag=(ngal_test>Ngal_threshold)
        mse_threshold = mean_squared_error(y_test[flag], y_pred[flag])
        r2_threshold = r2_score(y_test[flag], y_pred[flag])
        
        print(f'Model diagnostics for Ngal [> {Ngal_threshold} group galaxies only]:')
        print(f"R²: {r2_threshold:.3f} (R: {np.sqrt(r2_threshold):.3f})")
        print(f"MSE: {mse_threshold:.3f}")
        print('_________________________')

    #ploots
    print_=False
    if method=='fixed':
        print_=True
    
    if regression_plot:
        
        bin_centers, stats = confidence_intervals(y_test, y_pred, ngal_test, bin_width=bin_width,
                                                  method=method)
        plot_regression(y_test, y_pred, ngal_test, bin_centers, stats, threshold_width=threshold_width, min_bin_count=float(param_dict['min_bin_count']),
                        print_=print_)
    
    if importances_plot:
        plot_importances(X, model) 
        
    return X, model, y_test, y_pred
        

def read_features(param_dict_features=None):
    
    SigmaM_names = np.load(homedir+'/Desktop/SigmaMnames.npy', allow_pickle=True).tolist()
    Sigmak_names = np.load(homedir+'/Desktop/Sigmaknames.npy', allow_pickle=True).tolist()
            
    #prepare the FEATURE NAMES LIST!
    feature_list = SigmaM_names+Sigmak_names
    
    return feature_list
        
def read_params(params_path):
    #create dictionary with keyword and values from param textfile...
    param_dict = {}
    with open(params_path) as f:
        for line in f:
            try:
                key = line.split()[0]
                val = line.split()[1]
                param_dict[key] = val
            except:
                continue
    return param_dict

def parse_force_features(param_dict):
    #pull feature names from param_dict. might be [] (empty list), and if so default to reading from .npy files.
    feature_names = param_dict['force_features_list']
        
    if len(feature_names)<3:   #if empty bracket, pull full set of features names from .npy files
        print('No force_features_list parameter found in rf_regression_parameters.txt or the list is empty!')
        print('Searching for SigmaM and Sigmak .npy files...')
        
        try:
            feature_names = read_features()
            print('Success!')
        except:
            print('Bzzt. One or more feature files not found. Please generate the .npy files first before running.')
            sys.exit()  
    
    else:
        #okay...parse string to isolate the features
        #remove brackets, convert to list using ',' delimiter
        feature_names=feature_names.replace('[','')
        feature_names=feature_names.replace(']','')
        feature_names=feature_names.replace(' ','')   #also remove any stray spaces
        feature_names=feature_names.replace("'","")   #...and any single quotation marks
        feature_names=feature_names.split(',')
    
    return feature_names


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create ML model to predict log(M200) or environment class.")
    
    default_params_path = homedir+'/github/wisesize/ML_project/rf_regression_parameters.txt'
    
    parser.add_argument("-df", type=pd.core.frame.DataFrame, default=None, help="Input features/class pandas dataframe; defaults to looking for relevant file(s) on Desktop.")
    parser.add_argument("-features", type=list, default=None, help="Input list of feature column names; defaults to looking for relevant file(s) on Desktop.")
    parser.add_argument("-params", type=str, default=default_params_path, help="Input parameters for RF regression model and setup.")
    
    args = parser.parse_args()
    
    #create dictionary with keyword and values from param textfile...
    param_dict = read_params(args.params)
    
    df_path = param_dict['df_path']
    try:
        df = pd.read_csv(homedir+df_path)
    except:
        print('df not found. please generate the .csv file first before running.')
        sys.exit()
    
    #pull feature names from param_dict. might be [] (empty list), and if so default to reading from .npy files.
    feature_names = parse_force_features(param_dict)

    
    _ = RFR_model(df=df, feature_list=feature_names, use_pca=bool(int(param_dict['use_pca'])), 
              use_optimal_features=bool(int(param_dict['use_optimal_features'])), 
              pca_threshold=float(param_dict['correlation_threshold']), 
              Ngal_threshold=float(param_dict['Ngal_threshold']), test_size=float(param_dict['test_size']), 
              n_trees=int(param_dict['n_trees']), max_depth=int(param_dict['max_depth']), 
              random_state=int(param_dict['random_state']),
              bin_width=float(param_dict['bin_width']), threshold_width=float(param_dict['threshold_width']),
              min_bin_count=float(param_dict['min_bin_count']), method=str(param_dict['method']),
              regression_plot=True, importances_plot=True)

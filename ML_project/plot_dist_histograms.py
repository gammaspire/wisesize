'''
AIM: Generate a set of histograms which show both the distribution of y_true within a slice of y_pred log(M200), with a Gaussian model fit to the distribution.

PROCEDURE:
* use rf_regression_parameters.txt to govern the RFR model parameters
* create a bin 11.5 < logM200 [pred] < 12.5. 
* calculate the distance of true log(M200) values from the 1-to-1 line (i.e., true-pred)
* plot histogram of these distances 
* fit Gaussian curve to the histogram, maybe print mean/median/STDEV
* check for tails, whether there are more >3sigma galaxies than expected with the curve fit, etc.

**Procedure repeats for 12.5 < logM200 [pred] < 13.5 and 13.5 < logM200 [pred]< 14.5**

OUTPUT: 3-panel figure showing the distribution of the true log(M200) values for each pred log(M200) bin
'''

from ML_M200_regression import *

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from scipy.stats import norm


def read_df(param_dict):
        
    df_path = param_dict['df_path']
    try:
        df = pd.read_csv(homedir+df_path)
        return df
    except:
        print('df not found. please generate the .csv file first before running.')
        sys.exit()


def run_ML_regression(df, feature_names, param_dict):
    
    #just a basic little with no PCA features, no optimally-chosen features, no plots, and the parameters given in params.txt
    _, _, y_test, y_pred = RFR_model(df=df, 
                                         feature_list=feature_names, 
                                         use_pca=False, 
                                         use_optimal_features=False, 
                                         Ngal_threshold=float(param_dict['Ngal_threshold']), 
                                         test_size=float(param_dict['test_size']), 
                                         n_trees=int(param_dict['n_trees']), 
                                         max_depth=int(param_dict['max_depth']), 
                                         bin_width=float(param_dict['bin_width']), 
                                         threshold_width=float(param_dict['threshold_width']),
                                         regression_plot=False, importances_plot=False, 
                                         random_state=42) 
    
    #return the test (true) and predicted log(M200) arrays
    return y_test, y_pred


#create the three y_pred bins, calculate distance from 1-to-1 line
def create_ypred_bins(y_test, y_pred):
    
    #create the three y_pred bins -- will use the flags to isolate y_test galaxies within the defined ranges
    bin1_flag = (y_pred>=11.5) & (y_pred<12.5)
    bin2_flag = (y_pred>=12.5) & (y_pred<13.5)
    bin3_flag = (y_pred>=13.5) & (y_pred<14.5)

    #calculate distances between y_test in each bin and their corresponding y_pred (i.e., distance from 1-to-1 line)
    y1_dist = y_test[bin1_flag] - y_pred[bin1_flag]
    y2_dist = y_test[bin2_flag] - y_pred[bin2_flag]
    y3_dist = y_test[bin3_flag] - y_pred[bin3_flag]
    
    
    return [y1_dist, y2_dist, y3_dist]

def plot_hist_dist(dist_data):
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

    axes = axes.ravel()

    colors=['skyblue', 'orange', 'magenta']
    titles=['11.5 < y_pred < 12.5',
            '12.5 < y_pred < 13.5',
            '13.5 < y_pred < 14.5']
    
    for i, ax in enumerate(axes):
        
        #plot histogram
        counts, bins, _ = ax.hist(dist_data[i], bins=30, color=colors[i], edgecolor='black',
                                  density=False)  # use raw counts
        
        #fit Gaussian curve to the data
        mu, sigma = norm.fit(dist_data[i])
        
        #plot the curve...
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_width = bins[1] - bins[0]
        
        # scale PDF by number of points and bin width to match histogram height
        pdf = norm.pdf(bin_centers, mu, sigma) * len(dist_data[i]) * bin_width
        ax.plot(bin_centers, pdf, 'r--', linewidth=2,
                label=f'Fit: μ={mu:.2f}, σ={sigma:.2f}')
        
        ax.set_title(titles[i], fontsize=15)
        ax.set_xlabel('Distance between y_pred and y_test')
        ax.set_ylabel('Counts')
        
        ax.legend()
        ax.grid(alpha=0.3)

    plt.show()


    

if __name__ == "__main__":
    
    #initialize some crucial variables. so crucial, in fact, that this code is nonsensical without them.
    
    param_dict = read_params(homedir+'/github/wisesize/ML_project/rf_regression_parameters.txt')
    feature_list = read_features()
    
    df = read_df(param_dict)
    
    y_test, y_pred = run_ML_regression(df, feature_list, param_dict)
    dist_data = create_ypred_bins(y_test, y_pred)

    plot_hist_dist(dist_data)
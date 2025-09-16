'''
AIM: Generate a set of histograms which show both the distribution of y_true within a slice of y_pred log(M200), with a Gaussian model fit to the distribution.

PROCEDURE:
* use rf_regression_parameters.txt to govern the RFR model parameters
* create a bin 11.5 < logM200 [pred] < 12.5. 
* calculate the distance of true log(M200) values from the 1-to-1 line (i.e., true-pred)
* plot histogram of these distances 
* fit Gaussian curve to the histogram
* again find distances > 3-sigma and trim from distribution; save first 3-sigma value
* fit Gaussian curve again
* again find distances > 3-sigma and trim from distribution; save second 3-sigma value
* fit Gaussian curve AGAIN
* plot the Gaussian, print mean/median/STDEV
* color-code the region(s) of the histogram that were > 3-sigma of the first, second curve fits


**Procedure repeats for 12.5 < logM200 [pred] < 13.5 and 13.5 < logM200 [pred]< 14.5**

OUTPUT: 3-panel figure showing the distribution of the true log(M200) values for each pred log(M200) bin
'''

from ML_M200_regression import *

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from scipy.stats import norm
from rich import print


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

def plot_hist_dist(dist_data, plot_kde=False):
    
    '''
    plot_kde : bool -- set True if to add a smooth KDE fit to the data, which is more directly comparable to the Gaussian curve.
    '''
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True)

    axes = axes.ravel()

    colors=['skyblue', 'bisque', 'pink']
    colors_0 = ['blue', 'orangered', 'deeppink']
    titles=['11.5 < Pred log(M200) < 12.5',
            '12.5 < Pred log(M200) < 13.5',
            '13.5 < Pred log(M200) < 14.5']
    
    for i, ax in enumerate(axes):
        
        data = dist_data[i]
        x_vals = np.linspace(min(data), max(data), 500)
        
        if plot_kde:
            #KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            kde_vals = kde(x_vals)
                    
        #plot histogram
        counts, bins, patches = ax.hist(data, bins=40, color=colors[i], edgecolor='gray',
                                  alpha=0.6, density=True)
        
        #now...run the Gaussian fitting routine thrice -- the first two times will remove 3-sigma outliers
        #the third will be an actual fit
        #create a data_trim variable
        data_trim = data
        
        for iteration in range(3):
            
            #fit Gaussian curve to the data
            mu, sigma = norm.fit(data_trim)
            
            #if final iteration, continue with the mu and sigma fit
            if iteration==2:
                continue
            
            #calculate 3-sigma
            sigma3 = sigma*3
            
            #now remove the 3-sigma outliers. create the flag.
            #will be row-matched to original data array
            lower_bound = mu-sigma3
            upper_bound = mu+sigma3
            
            data_flag = (data > lower_bound) & (data < upper_bound)
            
            if iteration==1:
                #define sigma3 and mu bounds to a list; will use for plotting purposes below
                #to highlight the tail(s)
                sigma3_bounds = [lower_bound, upper_bound]
            
            #define the new data_trim
            data_trim = data_trim[data_flag]
            
        #calc PDF by number of points and bin width to match histogram height
        #to get density=True distribution, each bin is divided by [number of observations * the bin width]
        pdf = norm.pdf(x_vals, mu, sigma) #* len(dist_data[i]) * bin_width
        pdf_scaled = pdf * (counts.max() / pdf.max())
        
        ax.plot(x_vals, pdf_scaled, color='black', ls='--', linewidth=2,
                label=f'Fit: $\mu$={mu:.2f}\n'
                      f'1$\sigma$={sigma:.2f}')
        
        if plot_kde:
            kde_vals_scaled = kde_vals * (counts.max() / kde_vals.max())
            ax.plot(x_vals, kde_vals_scaled, color=colors_0[i], lw=2, label='Scaled KDE fit')        
        
        #lastly...color-code the wing(s) (i.e., galaxies which were omitted from the two 3-sigma tests)
        # `bins[:-1]` is used because there is one more bin edge than there are bins

        for patch, bin_edge in zip(patches, bins[:-1]):
            if bin_edge < sigma3_bounds[0] or bin_edge >= sigma3_bounds[1]:
                patch.set_facecolor(colors_0[i])
            else:
                patch.set_facecolor(colors[i])
        
        
        ax.set_title(titles[i], fontsize=15)
        ax.set_xlabel(r'log(M200)$_{Tempel}$ - log(M200)$_{Pred}$')
        ax.set_ylabel('Frequency')
        ax.set_xlim(-3,3)
        
        ax.legend(loc='upper left', fontsize=11.5)
        ax.grid(alpha=0.3)
    
    print('Note: Curve fits are scaled vertically for plotting purposes; does not affect mean or 1-sigma of the fit!')
    plt.show()


    

if __name__ == "__main__":
    
    #initialize some crucial variables. so crucial, in fact, that this code is nonsensical without them.
    
    param_dict = read_params(homedir+'/github/wisesize/ML_project/rf_regression_parameters.txt')
    feature_list = read_features()
    
    df = read_df(param_dict)
    
    y_test, y_pred = run_ML_regression(df, feature_list, param_dict)
    dist_data = create_ypred_bins(y_test, y_pred)

    plot_hist_dist(dist_data, plot_kde=True)
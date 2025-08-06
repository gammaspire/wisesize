'''
RandomForestRegression script for generating ML models to predict M200
'''

#this will not be code to import, so I am free to use name...main, etc.

#load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from astropy.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns

from ML_M200_functions import *


def plot_misclassifications(y_test, y_pred):
    '''
    Input: the true y-values, the predicted y-values from the ML model
    Output: histogram plot of True Classes, the total number of true galaxies per class (gray background), and the number of galaxies in the True Class which were misclassified as the other two classes (color-coded according to the legend). 
    FOR EXAMPLE, if Class 0 had 77 misclassified galaxies, the plot will show how many of those were misclassified as what classes...and since the gray shows the total number of galaxies in the True Class, the difference between the total height of the "misclassifications" and the "total" yields the number of galaxies correctly classified.
    '''
    
    #extract the total number of test galaxies in each class
    cm = confusion_matrix(y_test, y_pred)
    true_class_totals = cm.sum(axis=1)

    errors = (y_test != y_pred)
    error_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred, 'Error': errors})
    error_counts = error_df[error_df['Error']].groupby(['True', 'Predicted']).size().unstack(fill_value=0)

    error_counts.plot(kind='bar', stacked=True, alpha=0.7)

    plt.bar(error_counts.index, true_class_totals, color='lightgray', label='Total true count', zorder=0, width=0.5)

    plt.title("Misclassifications by True Class")
    plt.ylabel("Count")
    plt.xlabel("True Class")
    plt.show()


def RFC_model(df=None, feature_list=None, use_pca=True, use_optimal_features=False, threshold=0.90,
             logM200_threshold=None, regression_plot=True, correlation_plot=False, importances_plot=True,
             test_size=0.3, n_trees=200, max_depth=10):
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
    threshold : float, default=0.90
        Correlation threshold used for PCA feature reduction (if use_pca is True).
    logM200_threshold : float or None
        If set, excludes galaxies with log(M200) below this value from training/testing.
    regression_plot : bool, default=True
        Whether to display a scatter plot of predicted vs. true log(M200).
    correlation_plot : bool, default=False
        Whether to display a correlation heatmap of the selected features.
    importances_plot : bool, default=True
        Whether to show a bar chart of feature importances.
    test_size : float, default=0.3
        Fraction of data used for testing the model.
    n_trees : int, default=200
        Number of trees in the Random Forest.
    max_depth : int, default=10
        Maximum depth of each tree in the Random Forest.

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
            SigmaM_names = np.load(homedir+'/Desktop/SigmaMnames.npy', allow_pickle=True).tolist()
            Sigmak_names = np.load(homedir+'/Desktop/Sigmaknames.npy', allow_pickle=True).tolist()
            
            #prepare the FEATURE NAMES LIST!
            feature_list = SigmaM_names+Sigmak_names
        except:
            print('No input feature list and .npy files not found. Exiting.')
            sys.exit()
            
    df_out = df.copy()
    if use_pca:
        df_out, feature_list = get_pca_features(df, feature_list, threshold)
    
    
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
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    
    X=df_group[feature_list]
    y=df_group['group_M200']
    
    if logM200_threshold != None:
        #features...some hodgepodge of potentially relevant properties.
        X=df_group[feature_list][df_group['group_M200']>logM200_threshold]

        #target variable...log(M200)
        y=df_group['group_M200'][df_group['group_M200']>logM200_threshold]    
    

    if use_optimal_features:
        #find optimal feature set for the model
        selected_features = average_model_score(X, y, model=model, scoring='r2',regression=True,
                     n_splits=5, n_repeats=10, random_state=1)

        #predictors...some hodgepodge of potentially relevant properties.
        X=df_group[selected_features]
    
    #split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=63)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    #performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.3f}")
    print(f"R²: {r2:.3f}")
    
    
    #plerts
    
    if regression_plot:
        plot_regression(y_test, y_pred)
    
    if importances_plot:
        plot_importances(X, model)
    
    if correlation_plot:
        plot_correlations(X.corr())
    
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Create ML model to predict log(M200) or environment class.")
    
    parser.add_argument("-df", type=pd.core.frame.DataFrame, default=None, help="Input features/class pandas dataframe; defaults to looking for relevant file(s) on Desktop.")
    parser.add_argument("-features", type=list, default=None, help="Input list of feature column names; defaults to looking for relevant file(s) on Desktop.")
    parser.add_argument("-use_pca", action="store_true", help="Use PCA to reduce feature dimensionality.")
    parser.add_argument("-use_optimal_features", action="store_true", help="Iterate over model to find optimal features set.")
    parser.add_argument("-threshold", type=float, default=0.8, help="Correlation threshold (for PCA).")
    parser.add_argument("-logM200_threshold", type=float, default=None, help="log(M200) regression 'threshold', determines value above which the model will flag galaxies for test/train data.")
    parser.add_argument("-test_size", type=float, default=0.3, help="Defines the fraction of the sample user wants the model to use as test data.")
    parser.add_argument("-n_trees", type=int, default=200, help="Number of decision trees for creating the random forest model.")
    parser.add_argument("-max_depth", type=int, default=10, help="Hyperparameter for random trees model.")
    
    args = parser.parse_args()
    
    RFR_model(df=args.df, feature_list=args.features, use_pca=args.use_pca, 
              use_optimal_features=args.use_optimal_features, threshold=args.threshold, 
              logM200_threshold=args.logM200_threshold, test_size=args.test_size, 
              n_trees=args.n_trees, max_depth=args.max_depth, 
              regression_plot=True, correlation_plot=True, importances_plot=True)
    
    
    
    
    
    
    
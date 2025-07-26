'''
A collections of functions which are used for the predict_M200.ipynb Jupyter Notebook.
'''

def plot_misclassifications(y_test, y_pred):
    '''
    Input: the true y-values, the predicted y-values from the ML model
    Output: histogram plot of True Classes, the total number of true galaxies per class (gray background), and the number of galaxies in the True Class which were misclassified as the other two classes (color-coded according to the legend). 
    FOR EXAMPLE, if Class 0 had 77 misclassified galaxies, the plot will show how many of those were misclassified as what classes...and since the gray shows the total number of galaxies in the True Class, the difference between the total height of the "misclassifications" and the "total" yields the number of galaxies correctly classified.
    '''
    #load modules
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    
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
       
    
def perform_pca(df, correlated_groups, n_components):
    '''
    INPUT: pandas dataframe with row-matched features; correlated groups dictionary (see example below); n_components (integer) which will tell the function how many components to compress each group into
    OUTPUT: modified dataframe with the combined correlated features added and the individual components removed
    Note: if unsure about how to find correlated features, run find_correlations.py
    
    CORRELATED GROUPS EXAMPLE:
    correlated_groups = {
    'SigmaM1M2': ['Sigma_M1', 'Sigma_M2'],
    'SigmaM3M7': ['Sigma_M3', 'Sigma_M7'],
    'SigmaM9M10': ['Sigma_M9', 'Sigma_M10'],
    'Sigma10_20': ['Sigma_10', 'Sigma_20'],
    'SigmaNgal': ['Sigma_ngal_6', 'Sigma_ngal_11']
    }
    
    '''
    
    #load modules
    import pandas as pd  #just in case
    from sklearn.decomposition import PCA
    
    n_components=n_components
    suffix='_pca'   #each name in correlated_groups will have a _pca suffix attached

    df_out = df.copy()

    #for each row in correlated_groups dictionary, perform PCA
    for name, cols in correlated_groups.items():
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(df_out[cols])
        
        #add the new column(s) to df_out
        for i in range(n_components):
            new_col = f"{name}{suffix}{i+1}" if n_components > 1 else f"{name}{suffix}"
            df_out[new_col] = pca_data[:, i]
        
        #remove the component columns
        df_out.drop(columns=cols, inplace=True)
    
    return df_out
    

def find_correlations(X):
    '''
    INPUT: dataframe with features ONLY
    OUTPUT: seaborn grid plot which maps the strength of correlations between features, can use to diagnose which features may be removed or combined via PCA
    '''
    
    #import module...
    import seaborn as sns
    import pandas as pd #just in case
    
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')

    
    
    
    
#creating the feature CATALOGgGgGgGgGgGggggGGGGGggg

def generate_SigmaM_variants(vr_limits, radius_limits):
    '''
    INPUT: vr_limits and radius_limits,row-matched! 
        That is, if you want 1000 km/s and 1.0 Mpc, then 500 km/s and 2.0 Mpc, the lists should be [1000, 500] and [1, 2]
    OUTPUT: two lists -- one of the resultant features and one of names (strings)!
        In this case, the features are the logs of the 2D projected mass density of each applicable galaxy for each set of parameters
    '''
    
    import os
    import sys
    import numpy as np
    homedir=os.getenv("HOME")
    
    sys.path.append(homedir+'/github/wisesize/environment_classifications/')
    from mstar_local_density import Sigma_Mstar_Ngal
        
    if len(vr_limits) != len(radius_limits):
        print('vr_limits and radius_limits must have the same length! Exiting.')
        sys.exit()
    
    #initiate the lists!
    features_SigmaM = []
    names_SigmaM = []
    features_Ngal = []
    names_Ngal = []
    
    for n in range(len(vr_limits)):
        all_Sigma_Mstar, all_ngal = Sigma_Mstar_Ngal(vr_limit=vr_limits[n], radius_limit=radius_limits[n])
        
        features_SigmaM.append(np.log10(all_Sigma_Mstar))
        features_Ngal.append(all_ngal)
        
        names_SigmaM.append(f'Sigma_M{n}')
        names_Ngal.append(f'Sigma_ngal_{n}')

        
    #CALCULATING A FEW RATIOS...
    vr_limits = np.asarray(vr_limits)
    radius_limits = np.asarray(radius_limits)
    
    #find index at which 1000 km/s AND 1.0 Mpc
    ind1 = (vr_limits == 1000.) & (radius_limits == 1.0)
    
    #find index at which 300 km/s AND 1.5 Mpc
    ind2 = (vr_limits == 300.) & (radius_limits == 1.5)
    
    #find index at which 1000 km/s AND 2.0 Mpc
    ind3 = (vr_limits == 1000.) & (radius_limits == 2.0)

    
    # 1 Mpc vs 2 Mpc
    ratio_ngal = np.asarray(features_Ngal)[ind1]/np.asarray(features_Ngal)[ind3] 
    # 1000 km/s vs 300 km/s @ 1 Mpc
    ratio_Sigma_1 = np.asarray(features_SigmaM)[ind1]/np.asarray(features_SigmaM)[ind2] 
    # 1 Mpc vs 2 Mpc
    ratio_Sigma_2 = np.asarray(features_SigmaM)[ind1]/np.asarray(features_SigmaM)[ind3]   
    
    features = features_SigmaM + features_Ngal
    names = names_SigmaM + names_Ngal
    
    features.extend([ratio_ngal,ratio_Sigma_1,ratio_Sigma_2])
    names.extend(['ratio_ngal_1','ratio_SigmaM_1','ratio_SigmaM_2'])
    
    return features, names

def calculate_Sigmak_variants(vr_limits, k_list):
    '''
    INPUT: vr_limits and k nearest neighbor parameters, row-matched! 
        That is, if you want 1000 km/s and k=5, then 500 km/s and k=3, the lists should be [1000, 500] and [5, 3]
    OUTPUT: two lists -- one of the resultant features and one of names (strings)!
        In this case, the features are the logs of 2D kth nearest neighbor density of each applicable galaxy for each set of parameters
    '''
    
    import os
    import sys
    import numpy as np
    homedir=os.getenv("HOME")
    
    sys.path.append(homedir+'/github/wisesize/environment_classifications/')
    from kNN_local_density import compute_kNN_densities
    
    if len(vr_limits) != len(k_list):
        print('vr_limits and k_list must have the same length! Exiting.')
        sys.exit()
    
    #initiate the lists!
    features = []
    names = []
    
    for n in range(len(vr_limits)):
        all_kNN = compute_kNN_densities(vr_limit=vr_limits[n], k=k_list[n], use_radec=True)
        features.append(np.log10(all_kNN))  #all_kNN is the output of the above function :-)
        
        names.append(f'Sigma_{k_list[n]}')
    
    return features, names


def rowmatch_to_catalog(full_catalog, SigmaM_features, SigmaM_names, Sigmak_features, Sigmak_names):
    '''
    INPUT: full parent galaxy catalog of sample, SigmaM features, SigmaM names, Sigmak features, Sigmak names
        See calculate_SigmaM_variants and calculate_Sigmak_variants for clarification if unsure
    OUTPUT: full parent galaxy catalog with row-matched features!
    
    '''
    #import modules
    import os
    homedir=os.getenv("HOME")
    from astropy.table import Table
    import numpy as np
    
    #redefining to a variable I used below. can't be bothered switching it out.
    cat_full = full_catalog
    
    #combine the features...
    features = SigmaM_features + Sigmak_features   #joining two lists
    names = SigmaM_names + Sigmak_names  #joining two lists
    
    #create flags and row match the above arrays to the full catalog
    raflag = (cat_full['RA']>87) & (cat_full['RA']<300)
    decflag = (cat_full['DEC']>-10) & (cat_full['DEC']<85)
    mstarflag = cat_full['Mstar_all_flag']
    zflag = (cat_full['Z']>0.002) & (cat_full['Z']<0.025)

    #these are ALL flags applied to the input cat table for Sigma_*
    flags = (mstarflag) & (zflag) & (raflag) & (decflag)
    
    
    #create placeholder for row-matched features (helps avoid memory troubles!!)
    matched_features = []
    
    for n in range(len(features)):
                
        template = np.full(len(cat_full), -999.0)
        
        try:
            template[flags] = features[n]
        except:
            #in case the dimensions are wonky...e.g., (1,len(features[n]), as is the case with the ratio arrays
            template[flags] = np.asarray(features[n]).ravel()

        matched_features.append(template)
        
    #add these Sigma_* columns to the full catalog
    cat_full.add_columns(matched_features,
                          names=names)
    
    #heee go.
    return cat_full
    
    
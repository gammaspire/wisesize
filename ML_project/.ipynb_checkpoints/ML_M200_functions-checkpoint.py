'''
A collections of functions which are used for the predict_M200.ipynb Jupyter Notebook.
'''

    
def plot_importances(X, model):
    '''
    INPUT: X features, ML model
    OUTPUT: histogram of "importances" that the model assigns to each feature (i.e., how integrated each feature is into the formation of the model)
    '''
    from matplotlib import pyplot as plt
    
    importances = model.feature_importances_
    features = X.columns

    plt.barh(features, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.show()
    

def plot_correlations(df_out, feature_list):
    '''
    INPUT: dataframe with features ONLY
    OUTPUT: seaborn grid plot which maps the strength of correlations between features, can use to diagnose which features may be removed or combined via PCA
    '''
    
    #import module...
    import seaborn as sns
    import pandas as pd #just in case
    
    corr = df_out[feature_list].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)    
    
    
def perform_pca(df, correlated_groups, n_components):
    '''
    INPUT: pandas dataframe with row-matched FEATURES; correlated groups dictionary (see example below); n_components (integer) which will tell the function how many components to compress each group into
    OUTPUT: modified dataframe with the combined correlated features added and the individual components removed, list of pca columns
    Note: if unsure about how to find correlated features, run get_correlated_feature_groups.py
    
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
    pca_feature_names = []
    
    df_out = df.copy()

    #for each row in correlated_groups dictionary, perform PCA
    for name, cols in correlated_groups.items():
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(df_out[cols])
        
        #add the new column(s) to df_out
        for i in range(n_components):
            new_col = f"{name}{suffix}{i+1}" if n_components > 1 else f"{name}{suffix}"
            df_out[new_col] = pca_data[:, i]
            pca_feature_names.append(new_col)
        
        #remove the component columns
        df_out.drop(columns=cols, inplace=True)
    
    return df_out, pca_feature_names


def get_correlated_feature_groups(df, feature_names, threshold=0.90):
    '''
    PURPOSE: identify groups of correlated features at or above the input threshold, which the user can then input into perform_pca()
    
    INPUT: pandas dataframe, list of feature name columns, desired correlation threshold at or above which features will be grouped
    OUTPUT: correlated groups dictionary
    
    NOTE: This code evaluates the STRENGTH of the correlation, meaning that if the threshold is 0.90 then correlated groups will be those with |correlation| >= 0.90
          Yay dimensionality reduction!
    '''
    import pandas as pd
    import numpy as np
    import networkx as nx
    
    #compute the correlation matrix
    corr_matrix = df[feature_names].corr().abs()   
    
    #initialize an empty undirected graph. nodes are features, edges are connections between the nodes (connections are created only if |correlation|>threshold
    G = nx.Graph()   
    
    #this loop compares every pair of features without repeating pairs or checking diagonals
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j and corr_matrix.iloc[i, j] > threshold:   
                G.add_edge(col1, col2)   #add edge when two columns are highly correlated
    
    # Find connected components (i.e., groups of mutually correlated features)
    correlated_groups = {}
    for i, component in enumerate(nx.connected_components(G)):
        group_name = f"group_{i+1}"
        correlated_groups[group_name] = sorted(list(component))

    return correlated_groups    
    

def get_pca_features(df, feature_names, threshold):
    """
    Identifies groups of highly correlated features and applies PCA (Principal Component Analysis)
    to each group to reduce dimensionality, replacing them with a single principal component.

    Parameters:
        df (pd.DataFrame): The input dataframe containing features.
        feature_names (list): List of feature column names to consider for PCA.
        threshold (float): Correlation threshold above which features are considered highly correlated 
                           and candidates for dimensionality reduction via PCA.

    Returns:
        df_out (pd.DataFrame): Modified dataframe with principal component features added and
                               original correlated features removed.
        full_feature_list (list): Updated list of feature names, replacing correlated feature groups
                                  with their PCA-derived components.
    """

    #find the correlated groups
    correlated_groups = get_correlated_feature_groups(df, feature_names, threshold=threshold)
    
    #modify df, generate list of the pca feature names
    df_out, new_feature_names = perform_pca(df, correlated_groups, n_components=1)
    
    #combine list of old and new feature names
    full_feature_list = feature_names+new_feature_names
    
    #flattened_pca_input_features = [f for group in correlated_groups.values() for f in group]
    #full_feature_list = [f for f in feature_names if f not in flattened_pca_input_features] + new_feature_names
    
    #filter out the old features which were combined during PCA; no longer need the individual components
    full_feature_list = [col for col in full_feature_list if col in df_out.columns]
    
    plot_correlations(df_out, full_feature_list)
    
    return df_out, full_feature_list    
     
    
def optimal_features(X, y, model, scoring='f1_macro', regression=False):
    '''
    Select optimal features for a model using recursive feature elimination with cross-validation.

    Parameters:
        X: pd.DataFrame — Feature matrix. FEATURES ONLY.
        y: pd.Series or np.array — Labels/targets
        model: sklearn estimator — The model to use
        scoring: str — Scoring metric (e.g., 'f1_macro', 'accuracy', 'r2'). DEPENDS ON MODEL -- Regression or Classification
        regression: bool — Whether task is regression or classification
        n_splits: int — Number of folds in cross-validation

    Returns:
        List of selected feature names
    '''
    #import relevant modules
    from sklearn.feature_selection import RFECV
    
    if regression:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5)  #cross-validation strategy
    
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5)   #cross-validation strategy
    
    #fun fact...rfecv repeatedly splits data into training and validation/test folds internally
    rfecv = RFECV(estimator=model, step=1, cv=cv, scoring=scoring, n_jobs=1)
    rfecv.fit(X, y)

    selected_features = list(X.columns[rfecv.support_])
    
    print(f"Total number of features: {len(X.columns)}")
    print(f"Feature Names: {X.columns}")
    print()
    print(f"Optimal number of features: {rfecv.n_features_}")
    print("Selected features:", selected_features)
    print()
    
    return selected_features
    
    
def average_model_score(X, y, model=None, scoring='f1_macro', regression=False,
                     n_splits=5, n_repeats=10, random_state=1):
    '''
    INPUT: 
        X: DataFrame of input features
        y: Series or array of target values
        model: Estimator (defaults to balanced RandomForestClassifier)
        scoring: Scoring metric
        n_splits: Number of cross-validation folds
        n_repeats: Number of repeats
        random_state: Random seed for reproducibility
        regression: Whether the task is regression (default False)

    OUTPUT: 
        Prints mean and standard deviation of score; 
        Returns list of selected feature names
    
    NOTE: model will default to 
    model = RandomForestClassifier(n_estimators=80, random_state=0, max_depth=14,
                                 class_weight='balanced',max_features='sqrt')
    '''
    #import relevant modules
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    
    if regression:
        from sklearn.model_selection import RepeatedKFold
        cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=int(random_state))
    
    else:
        from sklearn.model_selection import RepeatedStratifiedKFold
        cv = RepeatedStratifiedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=int(random_state))
    
    if model is None:
        #define the model
        model = RandomForestClassifier(n_estimators=80, random_state=0, max_depth=14,
                                 class_weight='balanced',max_features='sqrt')
    
    selected_features = optimal_features(X, y, model, scoring=scoring, regression=regression)
    X_selected = X[selected_features]

    #evaluate model using cross-validation
    scores = cross_val_score(model, X_selected, y, scoring=scoring, cv=cv, n_jobs=1)

    #annnd, report performance
    print("Average performance of Random Forest Model using optimal features:")
    print(f"Mean {scoring}: {np.mean(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores)}")
    
    return selected_features
    
    
    
    
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
    
    features_logNgal = []
    names_logNgal = []
    
    for n in range(len(vr_limits)):
        all_Sigma_Mstar, all_ngal = Sigma_Mstar_Ngal(vr_limit=vr_limits[n], radius_limit=radius_limits[n])
        
        features_SigmaM.append(np.log10(all_Sigma_Mstar))
        features_logNgal.append(np.log10(all_ngal))
        features_Ngal.append(all_ngal)
        
        names_SigmaM.append(f'log_Sigma_M{n}')
        names_logNgal.append(f'log_Sigma_ngal_{n}')
        names_Ngal.append(f'Sigma_ngal_{n}')

        
    #CALCULATING A FEW RATIOS...
    vr_limits = np.asarray(vr_limits)
    radius_limits = np.asarray(radius_limits)
    
    #find index at which 1000 km/s AND 1.0 Mpc
    ind1 = (vr_limits == 1000.) & (radius_limits == 1.0)
    
    #find index at which 300 km/s AND 1.0 Mpc
    ind2 = (vr_limits == 300.) & (radius_limits == 1.0)
    
    #find index at which 1000 km/s AND 2.0 Mpc
    ind3 = (vr_limits == 1000.) & (radius_limits == 2.0)
    
    #find index at which 1000 km/s AND 0.5 Mpc
    ind4 = (vr_limits == 1000.) & (radius_limits == 0.5)
    
    #find index at which 1000 km/s AND 0.25 Mpc
    ind5 = (vr_limits == 1000.) & (radius_limits == 0.25)
    
    #find index at which 1000 km/s AND 3.0 Mpc
    ind6 = (vr_limits == 1000.) & (radius_limits == 3.0)
    
    # 1 Mpc vs 2 Mpc [1000 km/s]
    ratio_ngal_1 = np.asarray(features_logNgal)[ind1] - np.asarray(features_logNgal)[ind3] 
    
    #1.0 Mpc vs 0.5 Mpc [1000 km/s]
    ratio_ngal_2 = np.asarray(features_logNgal)[ind1] - np.asarray(features_logNgal)[ind4]
    
    # 1000 km/s vs 300 km/s [1 Mpc]
    ratio_Sigma_1 = np.asarray(features_SigmaM)[ind1] - np.asarray(features_SigmaM)[ind2] 
    
    # 1 Mpc vs 2 Mpc [1000 km/s]
    ratio_Sigma_2 = np.asarray(features_SigmaM)[ind1] - np.asarray(features_SigmaM)[ind3]   
    
    # 0.5 Mpc vs 0.25 Mpc [1000 km/s]
    ratio_Sigma_3 = np.asarray(features_SigmaM)[ind4] - np.asarray(features_SigmaM)[ind5]
    
    # 1.0 Mpc vs 0.5 Mpc [1000 km/s]
    ratio_Sigma_4 = np.asarray(features_SigmaM)[ind1] - np.asarray(features_SigmaM)[ind4]
    
    #0.5 Mpc vs. 3.0 Mpc [1000 km/s]
    ratio_Sigma_5 = np.asarray(features_SigmaM)[ind4] - np.asarray(features_SigmaM)[ind6]
    
    features = features_SigmaM + features_Ngal + features_logNgal
    names = names_SigmaM + names_Ngal + names_logNgal
    
    features.extend([ratio_ngal_1,ratio_Sigma_1,ratio_Sigma_2,ratio_Sigma_3,ratio_Sigma_4,ratio_ngal_2,ratio_Sigma_5])
    names.extend(['ratio_ngal_1','ratio_SigmaM_1','ratio_SigmaM_2','ratio_SigmaM_3','ratio_SigmaM_4','ratio_ngal_2',
                 'ratio_SigmaM_5'])
    
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
    
    #CALCULATING A FEW RATIOS...
    k_list = np.asarray(k_list)
    
    k2 = (k_list == 2)
    k5 = (k_list == 5)
    k10 = (k_list == 10)
    k20 = (k_list == 20)
   
    ratio_Sigmak_1 = np.asarray(features)[k10] - np.asarray(features)[k2] 
    
    features.extend([ratio_Sigmak_1])
    names.extend(['ratio_Sigmak_1'])
    
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
    
    
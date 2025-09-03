import pandas as pd
import numpy as np
import math
import os

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from numpy.random import default_rng
from scipy.spatial import KDTree
from scipy.stats import t
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

rng = np.random.RandomState(42)

# # set number of cores for parallel processing
# def configure_parallel_environment(verbose=True):
#     """
#     Configure parallel settings for joblib/loky and OpenMP.
#     - Limits OpenMP threads per process to 1.
#     - Sets LOKY_MAX_CPU_COUNT to physical cores if possible.

#     Call this BEFORE importing sklearn/joblib!
#     """
#     # 1) Limit OpenMP threads per process
#     os.environ['OMP_NUM_THREADS'] = '1'
#     if verbose:
#         print("OMP_NUM_THREADS set to 1")

#     # 2) Detect physical cores and set LOKY_MAX_CPU_COUNT
#     try:
#         import psutil
#         cores = psutil.cpu_count(logical=False)
#         if cores is None:
#             cores = psutil.cpu_count(logical=True)
#     except ImportError:
#         cores = os.cpu_count()

#     os.environ['LOKY_MAX_CPU_COUNT'] = str(cores)

#     if verbose:
#         print(f"LOKY_MAX_CPU_COUNT set to {cores} cores")

#     return cores



# function to synthesize data using Gaussian Mixture Model
def _synthesize_with_gmm(train_data, num_datasets, num_samples, num_components, covariance_type, n_init, random_state=None):
    """
    Helper function to synthesize data using Gaussian Mixture Model.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data to fit the GMM
    num_samples : int
        Number of samples to generate
    num_components : int
        Number of mixture components
    n_init : int, default=5
        Number of initializations to perform
    random_state : int, default=None
        Base random state for reproducibility
        
    Returns:
    --------
    list: sXs
        sXs: list of generated synthetic data frames
    """
    # Standardize the data for GMM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_data)
    
    # Fit GMM with a unique random state if provided
    gmm_random_state = random_state if random_state is not None else None
    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=gmm_random_state
    )
    gmm.fit(X_scaled)
    
    # Generate synthetic data with unique random seed for each dataset
    sXs = []
    for i in range(num_datasets):
        # Create a unique random state for each dataset
        dataset_random_state = None if random_state is None else random_state + i
        gmm.random_state = dataset_random_state
        synth_data, _ = gmm.sample(n_samples=num_samples)
        synth_data = scaler.inverse_transform(synth_data)
        sXs.append(pd.DataFrame(synth_data, columns=train_data.columns))
    
    return sXs
 
####################################################################################

# function to synthesize the joint distribution of categorical variables
def synthesize_with_joint_categorical(cat_data, num_datasets, num_samples=None, random_state=None):
    """
    Helper function to synthesize the joint distribution of a set of categorical variables.
    
    Steps:
    1) Compute all unique combinations and their empirical probabilities.
    2) Sample N new observations using these probabilities (N defaults to original row count).
    3) Return the newly sampled observations as DataFrames for each synthetic dataset.
    
    Parameters:
    -----------
    cat_data : pd.DataFrame
        DataFrame containing only the categorical variables to model jointly
    num_datasets : int
        Number of synthetic datasets to generate
    num_samples : int or None, default=None
        Number of rows to sample per dataset. If None, uses len(cat_data)
    random_state : int or None
        Base seed for reproducibility; each dataset will offset this seed by its index
    
    Returns:
    --------
    list[pd.DataFrame]
        List of synthetic DataFrames with the same columns and dtypes as cat_data
    """
    # Determine sample size
    if num_samples is None:
        num_samples = cat_data.shape[0]

    # Compute joint probabilities (include missing values explicitly)
    probs = cat_data.value_counts(dropna=False, normalize=True)
    # Convert the MultiIndex (or Index if single column) into a DataFrame of category combinations
    combos = probs.index.to_frame(index=False)
    p = probs.values

    # Prepare output datasets
    sXs = []

    for i in range(num_datasets):
        # Independent seed per dataset if provided
        rs = np.random.RandomState(None if random_state is None else random_state + i)
        # Sample combination indices with replacement according to empirical probabilities
        idx = rs.choice(len(p), size=num_samples, replace=True, p=p)
        sampled = combos.iloc[idx].reset_index(drop=True)

        # # Preserve categorical dtypes from input
        # for col in cat_data.columns:
        #     if pd.api.types.is_categorical_dtype(cat_data[col]):
        #         sampled[col] = pd.Categorical(
        #             sampled[col],
        #             categories=cat_data[col].cat.categories,
        #             ordered=cat_data[col].cat.ordered,
        #         )
        #     else:
        #         # Ensure dtype consistency for non-categorical columns (e.g., object/string)
        #         try:
        #             sampled[col] = sampled[col].astype(cat_data[col].dtype)
        #         except Exception:
        #             # Fallback to object if strict casting fails
        #             sampled[col] = sampled[col].astype(object)

        sXs.append(sampled)

    return sXs
 
####################################################################################

def _synthesize_with_multinomial(X, y, num_samples, synthetic_datasets, C, poly_degree, interaction_only, random_state=None):
    """
    Helper function to synthesize categorical data using multinomial logistic regression.
    
    Parameters:
    -----------
    X : pd.DataFrame or None
        Features for multinomial regression. If None, samples from prior.
    y : pd.Series
        Target variable to predict
    num_samples : int
        Number of samples to generate
    synthetic_datasets : list of pd.DataFrame or int
        Either a list of synthetic datasets or an integer specifying the number of synthetic datasets to generate
    C : float, default=1.0
        Inverse of regularization strength
    poly_degree : int, default=3
        Degree of polynomial features
    interaction_only : bool, default=False
        If true, only main effects and interactions are included
    random_state : int, default=None
        Random state for reproducibility
        
    Returns:
    --------
    list: sXs
        sXs: list of generated synthetic data frames
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Code to use if target variable is constant, i.e., it doesn't have more than once class
    if len(y.unique()) == 1:
        for i, sX in enumerate(synthetic_datasets):
            synth_data = np.repeat(y.iloc[0], num_samples)
            synthetic_datasets[i] = pd.concat([synthetic_datasets[i], pd.Series(synth_data, name=y.name)], axis=1)
        return synthetic_datasets
    
    # # Code used if this is the first variable being synthesized in the data set
    # if X is None or X.shape[1] == 0:
    #     # If no features, sample from prior distribution
    #     sXs = []
    #     for i in range(synthetic_datasets):
    #         counts = y.value_counts(normalize=True)
    #         synth_data = np.random.choice(
    #             counts.index,
    #             size=num_samples,
    #             p=counts.values
    #         )
    #         sXs.append(pd.Series(synth_data, name=y.name))
    #     return sXs
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    # Fit multinomial model
    model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='saga',
        max_iter=1000,
        random_state=random_state
    )
    model.fit(X_scaled, y)

    # loop over synthetic datasets
    for i, sX in enumerate(synthetic_datasets):
        sX_poly = poly.transform(sX)
        sX_scaled = scaler.transform(sX_poly)
        # Get predicted probabilities
        probs = model.predict_proba(sX_scaled)
        
        # Sample from multinomial distribution
        synth_data = [
            np.argmax(np.random.multinomial(n=1, pvals=p, size=1)) 
            for p in probs
            ]
        synth_data = model.classes_[synth_data]
        synthetic_datasets[i] = pd.concat([synthetic_datasets[i], pd.Series(synth_data, name=y.name)], axis=1)
    
    return synthetic_datasets

def ordered_vector_distances(x):

    # this function approximates the local sparsity of data using the median distance between
    # a given point and its four nearest neighbors based on a sorted index

    # number of elements
    size = len(x)

    # empty vector for storing median distances
    dists = np.empty(size, dtype=np.float32)

    # ---------- interior (i=2..size-3): neighbors are i-2, i-1, i+1, i-2 ----------
    # i_start and i_end are the indices of the first and last interior points
    i_start, i_end = 2, size-3
    # Build neighbor arrays aligned to x[i_start:i_end+1]
    L2 = x[(i_start-2):(i_end-2+1)]          # x[i-2]
    L1 = x[(i_start-1):(i_end-1+1)]          # x[i-1]
    R1 = x[(i_start+1):(i_end+1+1)]          # x[i+1]
    R2 = x[(i_start+2):(i_end+2+1)]          # x[i+2]
    center = x[i_start:i_end+1]

    # Stack the 4 absolute diffs as columns (shape: size x 4)
    diffs = np.stack([
        np.abs(L2 - center),
        np.abs(L1 - center),
        np.abs(R1 - center),
        np.abs(R2 - center)
    ], axis=1)

    # store median diffs for interior values
    dists[i_start:i_end+1] = np.median(diffs, axis=1)

    # ---------- edges ----------
    # i = 0: use x[1:5]
    dists[0] = np.median(np.abs(x[1:5] - x[0]))

    # i = 1: use [x[0], x[2:5]]
    dists[1] = np.median(np.abs(np.concatenate((x[0:1], x[2:5])) - x[1]))

    # i = n-2: use [x[n-5:n-2], x[n-1]]
    dists[size-2] = np.median(np.abs(np.concatenate((x[(size-5):(size-2)], x[(size-1):size])) - x[size-2]))

    # i = n-1: use x[n-5:n-1]
    dists[size-1] = np.median(np.abs(x[(size-5):(size-1)] - x[size-1]))

    # enforce minimum distance for smoothing
    dists =np.maximum(dists, 1e-6)

    return dists

#########################################################################################################

def synthesize_with_tree_regressor(
    X,
    y,
    num_samples,
    synthetic_datasets,
    min_samples_leaf,
    random_state=None,
):
    """
    Optimized version of tree-based synthesizer with vectorized operations.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 1) Transform y with Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    y_trans = pt.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()
    
    # 2) Fit DecisionTreeRegressor and get leaf assignments
    tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=random_state)
    tree.fit(X, y_trans)

    # Precompute training leaf assignments and values in each leaf
    # note that we are sorting the values in each leaf
    train_leaves = tree.decision_path(X).toarray()
    # store sorted values in each leaf    
    leaf_vals = {leaf: np.sort(y_trans[np.where(train_leaves[:, leaf] == 1)[0]]) for leaf in range(train_leaves.shape[1])}
    # compute the median distance to four nearest neighbors by index for each training value in each leaf
    leaf_dists = {leaf: ordered_vector_distances(leaf_vals[leaf]) for leaf in range(train_leaves.shape[1])}
    
    # Process each synthetic dataset
    results = []
    rs = np.random.RandomState(random_state)
    
    for i, sX in enumerate(synthetic_datasets):
        
        # Get leaf assignments for synthetic data
        s_leaves = tree.apply(sX[X.columns])
        
        # For each unique leaf, process all its samples at once
        unique_s_leaves, counts = np.unique(s_leaves, return_counts=True)
        synth_vals = np.empty(num_samples, dtype=float)
        
        for leaf, count in zip(unique_s_leaves, counts):
            
            # Get indices in synth_vals that belong to this leaf
            leaf_mask = (s_leaves == leaf)
            
            # Get training values for this leaf
            vals = leaf_vals[leaf]
                
            # Sample indexes of base values
            base_samples_indexes = rs.choice(range(len(vals)), size=count, replace=True)

            # Add noise
            synth_vals[leaf_mask] = vals[base_samples_indexes] + rs.normal(loc=0, scale=leaf_dists[leaf][base_samples_indexes])
        
        # Inverse transform and store results
        synth_orig = pt.inverse_transform(synth_vals.reshape(-1, 1)).ravel()
        results.append(pd.concat([sX, pd.Series(synth_orig, name=y.name)], axis=1))

        print('Synthetic dataset', i, 'generated.')
    
    return results

#########################################################################################################

# # function to compute the number of features/covariates that are a function of synthetic variables
# # this includes standalone features and their polynomials and interactions (even with non-synthesized variables)
# def num_terms_with_synthetic(p, s, d):
#     """
#     Calculate the number of transformed variables involving at least one synthetic variable
#     after polynomial expansion of degree d.

#     Parameters:
#     p (int): total number of original variables
#     s (int): number of synthetic variables
#     d (int): maximum degree of polynomial terms

#     Returns:
#     int: number of terms involving at least one synthetic variable
#     """
#     def binomial(n, k):
#         return math.comb(n, k) if n >= k else 0

#     total_terms = sum(binomial(p + i - 1, i) for i in range(1, d + 1))
#     nonsynthetic_terms = sum(binomial(p - s + i - 1, i) for i in range(1, d + 1))
#     return total_terms - nonsynthetic_terms

#########################################################################################################

# # function to compute the pMSE ratio from a given original and synthetic data set
# def pmse_ratio(original_data, synthetic_data, num_synthetic_vars, poly_degree, interaction_only):
    
#     # number of features/covariates that are a function of synthetic variables
#     k = num_terms_with_synthetic(p=synthetic_data.shape[1], s=num_synthetic_vars, d=poly_degree)

#     # observation counts
#     N_synth = synthetic_data.shape[0]
#     N_orig = original_data.shape[0]
    
#     # combine original and synthetic datasets
#     full_X = pd.concat([original_data, synthetic_data], axis=0).reset_index(drop=True)
    
#     # generate interactions and powers of variables
#     poly = PolynomialFeatures(poly_degree, interaction_only=interaction_only, include_bias=False)
    
#     full_X = poly.fit_transform(full_X)

#     # scale the combined dataset
#     full_X = preprocessing.StandardScaler().fit_transform(full_X)
    
#     c = N_synth/(N_synth+N_orig)

#     y = np.repeat([0, 1], repeats=[N_orig, N_synth])
    
#     pMSE_model = LogisticRegression(penalty=None, max_iter=1000, solver='saga').fit(full_X, y)
    
#     probs = pMSE_model.predict_proba(full_X)
    
#     pMSE = 1/(N_synth+N_orig) * np.sum((probs[:,1] - c)**2)
    
#     # the formula uses (k - 1) in the paper because k includes the intercept
#     # we compute k using the dimensionality of the predictor matrix excluding the
#     # intercept
#     e_pMSE = 2*(k)*(1-c)**2 * c/(N_synth+N_orig)
        
#     return pMSE/e_pMSE

#########################################################################################################

# def perform_synthesis(
#     train_data,
#     number_synthetic_datasets,
#     poly_degree_mnl,
#     poly_degree_pmse,
#     interaction_only,
#     covariance_type,
#     gmm_n_init,
#     perform_pca,
#     columns_to_round,
#     # New parameters with defaults for backward compatibility
#     synthesis_steps=None,
#     # New parameter for optimization bounds
#     param_values=None,
#     random_state=None
# ):
#     """
#     Generate synthetic datasets using specified synthesis methods for each variable.
    
#     Parameters:
#     -----------
#     train_data : pd.DataFrame
#         The training data to synthesize
#     number_synthetic_datasets : int
#         Number of synthetic datasets to generate
#     synthesis_steps : list of tuples
#         List of (variables, method) tuples specifying synthesis order.
#         Variables can be a single variable name (str) or list of variable names.
#         Method can be 'gmm', 'multinomial', or 'cart'.
#         Example:
#             [
#                 (['var1', 'var2'], 'gmm'),  # First synthesize these with GMM
#                 ('var3', 'multinomial'),     # Then var3 using previously synthesized vars
#                 ('var4', 'multinomial')      # Then var4 using all previous vars
#             ]
#     param_values : dict
#         Bounds for optimization parameters.
#         Example:
#             {
#                 'gmm': {'num_components': 25,
#                         'n_init': 5},
#                 'multinomial': {
#                     'var3': {'C_var3': 1.0}  # Specific for var3
#                 }
#             }
#     random_state : int, optional
#         Random state for reproducibility
        
#     Returns:
#     --------
#     tuple: (pmse_ratios, synthetic_datasets)
#         pmse_ratios: List of pMSE ratio for each synthetic dataset
#         synthetic_datasets: List of synthetic datasets
#     """
#     # Set random state
#     if random_state is not None:
#         np.random.seed(random_state)

#     # # this is standalone functionality for now - it ignores the code below
#     # if perform_pca:
#     #     # extract the desired number of components
#     #     num_pca_components = int(param_values.get('num_pca_components'))
#     #     # define PCA model
#     #     pca_model = PCA(n_components=num_pca_components)
#     #     # fit PCA model to the training data
#     #     pca_model.fit(train_data)
#     #     # transform the training data
#     #     pca_train = pd.DataFrame(pca_model.transform(train_data))

#     #     # Initialize data structures
#     #     num_samples = pca_train.shape[0]

#     #     # Get parameters for GMM
#     #     gmm_params = param_values.get('gmm', {})
#     #     num_components = gmm_params.get('num_components', {})
            
#     #     # Synthesize with GMM
#     #     # returns a list of synthetic data sets
#     #     synthetic_datasets = _synthesize_with_gmm(
#     #         pca_train,
#     #         num_datasets=number_synthetic_datasets,
#     #         num_samples=num_samples,
#     #         num_components=num_components,
#     #         covariance_type=covariance_type,
#     #         n_init=gmm_n_init,
#     #         random_state=random_state
#     #     )

#     #     # reverse the PCA transformation on the synthetic data
#     #     synthetic_datasets = [pd.DataFrame(pca_model.inverse_transform(synth_df)) for synth_df in synthetic_datasets]
        
#     #     # add correct feature names back to synthetic data
#     #     for synth_df in synthetic_datasets:
#     #         synth_df.columns = train_data.columns
#     #         # round any columns specified in columns_to_round
#     #         for col in columns_to_round:
#     #             synth_df[col] = synth_df[col].round(decimals=0)

#     #     # Calculate pMSE ratios
#     #     pmse_ratios = [
#     #         pmse_ratio(
#     #         train_data, 
#     #         synth_df, 
#     #         train_data.shape[1], 
#     #         poly_degree=poly_degree_pmse,
#     #         interaction_only=interaction_only
#     #         ) for synth_df in synthetic_datasets
#     #     ]

#     #     return pmse_ratios, synthetic_datasets

#     # Initialize data structures
#     num_samples = train_data.shape[0]

#     # Track which variables have been synthesized
#     synthesized_vars = []
    
#     # Process each synthesis step
#     for step_idx, (step_vars, method) in enumerate(synthesis_steps):
#         if isinstance(step_vars, str):
#             step_vars = [step_vars]
            
#         # Get dependencies (all previously synthesized variables)
#         dependencies = []
#         for prev_vars, _ in synthesis_steps[:step_idx]:
#             if isinstance(prev_vars, str):
#                 prev_vars = [prev_vars]
#             dependencies.extend(prev_vars)
        
#         # Remove duplicates and variables being synthesized in this step
#         dependencies = [v for v in dependencies if v not in step_vars]
        
#         if method == 'gmm':
#             if dependencies:
#                 raise ValueError("GMM synthesis cannot depend on previously synthesized variables. "
#                                "It should be the first synthesis step.")
            
#             # Get parameters for GMM
#             gmm_params = param_values.get('gmm', {})
#             num_components = gmm_params.get('num_components', {})
            
#             # Synthesize with GMM
#             # returns a list of synthetic data sets
#             synthetic_datasets = _synthesize_with_gmm(
#                 train_data[step_vars],
#                 num_datasets=number_synthetic_datasets,
#                 num_samples=num_samples,
#                 num_components=num_components,
#                 covariance_type=covariance_type,
#                 n_init=gmm_n_init,
#                 random_state=random_state
#             )
                
#         elif method == 'joint_categorical':
#             if dependencies:
#                 raise ValueError("joint_categorical synthesis cannot depend on previously synthesized variables. "
#                                  "It should be the first (or an independent) synthesis step.")

#             # Generate joint samples for the specified categorical variables
#             joint_sets = synthesize_with_joint_categorical(
#                 cat_data=train_data[step_vars],
#                 num_datasets=number_synthetic_datasets,
#                 num_samples=num_samples,
#                 random_state=random_state,
#             )

#             synthetic_datasets = [df.copy() for df in joint_sets]

#         elif method == 'multinomial':
#             # Process each variable in this step
#             for var in step_vars:
#                 # Get parameters for this variable
#                 var_params = {}
#                 if param_values and 'multinomial' in param_values:
#                     # Get variable-specific params
#                     if var in param_values['multinomial']:
#                         var_params.update(param_values['multinomial'][var])
                
#                 C = var_params.get('C', {})
                
#                 # Get features (dependencies)
#                 X = train_data[dependencies] if dependencies else None

#                 # Generate synthetic data for this variable
#                 synthetic_datasets = _synthesize_with_multinomial(
#                     X=X,
#                     y=train_data[var],
#                     num_samples=num_samples,
#                     poly_degree=poly_degree_mnl,
#                     interaction_only=interaction_only,
#                     synthetic_datasets=synthetic_datasets,
#                     C=C,
#                     random_state=random_state
#                 )
#         else:
#             raise ValueError(f"Unknown synthesis method: {method}")
        
#         # Update list of synthesized variables
#         synthesized_vars.extend(step_vars)
    
#     # Calculate pMSE ratios
#     pmse_ratios = [
#         pmse_ratio(
#             train_data, 
#             synth_df, 
#             len(synthesized_vars), 
#             poly_degree=poly_degree_pmse,
#             interaction_only=interaction_only
#         )
#         for synth_df in synthetic_datasets
#     ]
    
#     return pmse_ratios, synthetic_datasets

######################################################################################################

# def optimize_models(train_data,
#                     number_synthetic_datasets,
#                     synthesis_steps,
#                     param_bounds,
#                     poly_degree_mnl,
#                     poly_degree_pmse,
#                     interaction_only,
#                     covariance_type,
#                     gmm_n_init,
#                     perform_pca,
#                     columns_to_round,
#                     random_state=None,
#                     num_iter_optimization=25,
#                     num_init_optimization=5):
#     """
#     Optimize synthesis model parameters using Bayesian optimization.
    
#     Parameters:
#     -----------
#     train_data : pd.DataFrame
#         The training data to synthesize
#     number_synthetic_datasets : int
#         Number of synthetic datasets to generate
#     synthesis_steps : list of tuples
#         List of (variables, method) tuples specifying synthesis order.
#     param_bounds : dict
#         Dictionary specifying parameter bounds for optimization.
#         Example:
#             {
#                 'gmm': {
#                     'num_components': (10, 200),
#                     'n_init': (1, 10)
#                 },
#                 'multinomial': {
#                     'C': (0.001, 3),  # Global default
#                     'var1': {'C': (0.1, 5)}  # Specific for var1
#                 }
#             }
#     random_state : int, optional
#         Random state for reproducibility
#     """
#     # Process parameter bounds for Bayesian optimization
#     dimensions = []
#     param_mapping = []
    
#     # Track which parameters we need to optimize
#     for step_vars, method in synthesis_steps:
#         if isinstance(step_vars, str):
#             step_vars = [step_vars]

#         # Add PCA parameters if perform_pca is True
#         if perform_pca:
#             pca_bounds = param_bounds.get('pca', {})
#             if 'num_pca_components' in pca_bounds:
#                 bounds = pca_bounds['num_pca_components']
#                 dimensions.append((bounds[0], bounds[1], 'uniform', 'pca_components'))
#                 param_mapping.append(('pca', 'num_pca_components'))
            
#         if method == 'gmm' and 'gmm' in param_bounds:
#             for param, bounds in param_bounds['gmm'].items():
#                 if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
#                     # Add dimension for this parameter
#                     dim_name = f"gmm_{param}"
#                     dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
#                     param_mapping.append(('gmm', param))
                    
#         elif method in ['multinomial', 'tree']:
#             for var in step_vars:
#                 method_bounds = param_bounds.get(method, {}).get(var, {})
#                 for param, bounds in method_bounds.items():
#                     if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
#                         dim_name = f"{method}_{var}_{param}"
#                         dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
#                         param_mapping.append((method, var, param))
    
#     def evaluate_models(**kwargs):
#         # Reconstruct the param_values structure from the flat parameter space
#         param_values = {'gmm': {}, 'multinomial': {}, 'tree': {}}
        
#         for i, (method, *rest) in enumerate(param_mapping):
#             param_name = f"x{i}"
#             if param_name in kwargs:
#                 if method == 'gmm':
#                     param, = rest
#                     param_values['gmm'][param] = int(kwargs[param_name])
#                 else:  # 'multinomial' or 'tree'
#                     var, param = rest
#                     if var not in param_values[method]:
#                         param_values[method][var] = {}
#                     param_values[method][var][param] = kwargs[param_name]
        
#         # Run synthesis with current parameters
#         pmse_ratios, _ = perform_synthesis(
#             train_data=train_data,
#             number_synthetic_datasets=number_synthetic_datasets,
#             synthesis_steps=synthesis_steps,
#             poly_degree_mnl=poly_degree_mnl,
#             poly_degree_pmse=poly_degree_pmse,
#             interaction_only=interaction_only,
#             covariance_type=covariance_type,
#             gmm_n_init=gmm_n_init,
#             perform_pca=perform_pca,
#             columns_to_round=columns_to_round,
#             param_values=param_values,
#             random_state=random_state
#         )
        
#         # Return negative of mean squared pMSE ratio error (we want to maximize this)
#         return -1 * ((1 - np.mean(pmse_ratios)) ** 2)
    
#     # Create parameter bounds for Bayesian optimization
#     pbounds = {f"x{i}": (low, high) for i, (low, high, _, _) in enumerate(dimensions)}

#     optimizer = BayesianOptimization(
#         f=evaluate_models,
#         pbounds=pbounds,
#         verbose=2,
#         random_state=random_state,
#         acquisition_function=acquisition.ExpectedImprovement(xi=1e-02)
#     )
    
#     optimizer.maximize(init_points=num_init_optimization, n_iter=num_iter_optimization)
    
#     # Process results
#     best_params = {'gmm': {}, 'multinomial': {}, 'tree': {}}
#     for i, (method, *rest) in enumerate(param_mapping):
#         param_name = f"x{i}"
#         if method == 'gmm':
#             param, = rest
#             best_params['gmm'][param] = int(optimizer.max['params'][param_name])
#         else:  # 'multinomial' or 'tree'
#             var, param = rest
#             if var not in best_params[method]:
#                 best_params[method][var] = {}
#             best_params[method][var][param] = optimizer.max['params'][param_name]
    
#     return {
#         'best_params': best_params,
#         'best_score': -optimizer.max['target'],  # Convert back to positive pMSE
#         'optimizer': optimizer
#     }

#########################################################################################################

# Helper to compute OLS parameter vector for a given dataset and model spec
def _ols_params(X, y):
    """Helper to compute OLS parameters with constant term."""
    X = add_constant(X, has_constant='add')
    return OLS(y, X).fit().params

#########################################################################################################

def perform_synthesis_with_param_target(
    train_data,
    number_synthetic_datasets,
    synthesis_steps,
    target_params,
    param_values=None,
    random_state=None
):
    """
    Generate synthetic datasets and evaluate quality by comparing regression parameters.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        The training data to synthesize
    number_synthetic_datasets : int
        Number of synthetic datasets to generate
    synthesis_steps : list of tuples
        List of (variables, method) tuples specifying synthesis order
    target_params : array-like
        Target regression parameters to compare against (including intercept)
    param_values : dict, optional
        Dictionary of parameter values for synthesis methods
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    tuple: (ssd_scores, synthetic_datasets)
        ssd_scores: List of sum of squared deviations for each synthetic dataset
        synthetic_datasets: List of synthetic datasets
    """
    # Run the standard synthesis pipeline (without pMSE computation)
    # This duplicates the control flow but replaces the final metric.

    if random_state is not None:
        np.random.seed(random_state)

    if perform_pca:
        num_pca_components = int(param_values.get('num_pca_components'))
        pca_model = PCA(n_components=num_pca_components)
        pca_model.fit(train_data)
        pca_train = pd.DataFrame(pca_model.transform(train_data))

        num_samples = pca_train.shape[0]
        gmm_params = param_values.get('gmm', {})
        num_components = gmm_params.get('num_components', {})

        synthetic_datasets = _synthesize_with_gmm(
            pca_train,
            num_datasets=number_synthetic_datasets,
            num_samples=num_samples,
            num_components=num_components,
            covariance_type=covariance_type,
            n_init=gmm_n_init,
            random_state=random_state,
        )

        synthetic_datasets = [pd.DataFrame(pca_model.inverse_transform(synth_df)) for synth_df in synthetic_datasets]
        for synth_df in synthetic_datasets:
            synth_df.columns = train_data.columns
            for col in columns_to_round:
                synth_df[col] = synth_df[col].round(decimals=0)
    else:
        num_samples = train_data.shape[0]
        synthesized_vars = []
        for step_idx, (step_vars, method) in enumerate(synthesis_steps):
            if isinstance(step_vars, str):
                step_vars = [step_vars]

            dependencies = []
            for prev_vars, _ in synthesis_steps[:step_idx]:
                if isinstance(prev_vars, str):
                    prev_vars = [prev_vars]
                dependencies.extend(prev_vars)
            dependencies = [v for v in dependencies if v not in step_vars]

            if method == 'gmm':
                if dependencies:
                    raise ValueError("GMM synthesis cannot depend on previously synthesized variables. It should be the first synthesis step.")
                gmm_params = param_values.get('gmm', {}) if param_values else {}
                num_components = gmm_params.get('num_components', {})
                synthetic_datasets = _synthesize_with_gmm(
                    train_data[step_vars],
                    num_datasets=number_synthetic_datasets,
                    num_samples=num_samples,
                    num_components=num_components,
                    covariance_type=covariance_type,
                    n_init=gmm_n_init,
                    random_state=random_state,
                )

            elif method == 'joint_categorical':
                if dependencies:
                    raise ValueError("joint_categorical synthesis cannot depend on previously synthesized variables. It should be the first (or an independent) synthesis step.")
                joint_sets = synthesize_with_joint_categorical(
                    cat_data=train_data[step_vars],
                    num_datasets=number_synthetic_datasets,
                    num_samples=num_samples,
                    random_state=random_state,
                )
                synthetic_datasets = [df.copy() for df in joint_sets]

            elif method == 'multinomial':
                for var in step_vars:
                    var_params = {}
                    if param_values and 'multinomial' in param_values and var in param_values['multinomial']:
                        var_params.update(param_values['multinomial'][var])
                    C = var_params.get('C', {})
                    X = train_data[dependencies] if dependencies else None
                    synthetic_datasets = _synthesize_with_multinomial(
                        X=X,
                        y=train_data[var],
                        num_samples=num_samples,
                        poly_degree=poly_degree_mnl,
                        interaction_only=interaction_only,
                        synthetic_datasets=synthetic_datasets,
                        C=C,
                        random_state=random_state,
                    )
            elif method == 'tree':
                for var in step_vars:
                    var_params = {}
                    if param_values and 'tree' in param_values and var in param_values['tree']:
                        var_params.update(param_values['tree'][var])
                    max_depth = var_params.get('max_depth', {})
                    X = train_data[dependencies] if dependencies else None
                    synthetic_datasets = _synthesize_with_tree(
                        X=X,
                        y=train_data[var],
                        num_samples=num_samples,
                        synthetic_datasets=synthetic_datasets,
                        max_depth=max_depth,
                        random_state=random_state,
                    )
            else:
                raise ValueError(f"Unknown synthesis method: {method}")

            synthesized_vars.extend(step_vars)

    # Compute SSD per synthetic dataset
    # Ensure target_params is aligned to the params index order
    target_series = pd.Series(target_params)
    ssds = []
    for Y in synthetic_datasets:
        params = _ols_params(data=Y, y=target_variable, X=exog_variables)
        # Align indices
        aligned = params.reindex(target_series.index)
        ssds.append(float(np.sum((aligned.values - target_series.values) ** 2)))

    return ssds, synthetic_datasets

#########################################################################################################

def optimize_models_with_param_target(
    train_data,
    number_synthetic_datasets,
    synthesis_steps,
    target_params,
    param_bounds,
    random_state=None,
    n_iter=25,
    n_init=5
):
    """
    Optimize synthesis parameters to match target regression parameters.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        The training data to synthesize
    number_synthetic_datasets : int
        Number of synthetic datasets to generate per evaluation
    synthesis_steps : list of tuples
        List of (variables, method) tuples specifying synthesis order
    target_params : array-like
        Target regression parameters (including intercept)
    param_bounds : dict
        Dictionary of parameter bounds for optimization
    random_state : int, optional
        Random state for reproducibility
    n_iter : int
        Number of optimization iterations
    n_init : int
        Number of initial random evaluations
        
    Returns:
    --------
    dict
        Dictionary containing best parameters and optimization results
    """
    dimensions = []
    param_mapping = []

    for step_idx, (step_vars, method) in enumerate(synthesis_steps):
        if isinstance(step_vars, str):
            step_vars = [step_vars]

        if perform_pca:
            pca_bounds = param_bounds.get('pca', {})
            if 'num_pca_components' in pca_bounds:
                bounds = pca_bounds['num_pca_components']
                dimensions.append((bounds[0], bounds[1], 'uniform', 'pca_components'))
                param_mapping.append(('pca', 'num_pca_components'))

        if method == 'gmm' and 'gmm' in param_bounds:
            for param, bounds in param_bounds['gmm'].items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    dim_name = f"gmm_{param}"
                    dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
                    param_mapping.append(('gmm', param))
        elif method in ['multinomial', 'tree']:
            for var in step_vars:
                method_bounds = param_bounds.get(method, {}).get(var, {})
                for param, bounds in method_bounds.items():
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        dim_name = f"{method}_{var}_{param}"
                        dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
                        param_mapping.append((method, var, param))

    def evaluate_models(**kwargs):
        pv = {'gmm': {}, 'multinomial': {}, 'tree': {}}
        for i, (method, *rest) in enumerate(param_mapping):
            param_name = f"x{i}"
            if param_name in kwargs:
                if method == 'gmm':
                    param, = rest
                    pv['gmm'][param] = int(kwargs[param_name])
                elif method in ['multinomial', 'tree']:
                    var, param = rest
                    if var not in pv[method]:
                        pv[method][var] = {}
                    pv[method][var][param] = kwargs[param_name]

        ssds, _ = perform_synthesis_with_param_target(
            train_data=train_data,
            number_synthetic_datasets=number_synthetic_datasets,
            synthesis_steps=synthesis_steps,
            target_variable=target_variable,
            exog_variables=exog_variables,
            target_params=target_params,
            poly_degree_mnl=poly_degree_mnl,
            interaction_only=interaction_only,
            covariance_type=covariance_type,
            gmm_n_init=gmm_n_init,
            perform_pca=perform_pca,
            columns_to_round=columns_to_round,
            param_values=pv,
            random_state=random_state,
        )

        # Maximize negative average SSD
        return -np.mean(ssds)

    pbounds = {f"x{i}": (low, high) for i, (low, high, _, _) in enumerate(dimensions)}

    optimizer = BayesianOptimization(
        f=evaluate_models,
        pbounds=pbounds,
        verbose=2,
        random_state=random_state,
        acquisition_function=acquisition.ExpectedImprovement(xi=1e-02),
    )

    optimizer.maximize(init_points=n_init, n_iter=n_iter)

    best_params = {'gmm': {}, 'multinomial': {}, 'tree': {}}
    for i, (method, *rest) in enumerate(param_mapping):
        param_name = f"x{i}"
        if method == 'gmm':
            param, = rest
            best_params['gmm'][param] = int(optimizer.max['params'][param_name])
        elif method in ['multinomial', 'tree']:
            var, param = rest
            if var not in best_params[method]:
                best_params[method][var] = {}
            best_params[method][var][param] = optimizer.max['params'][param_name]

    return {
        'best_params': best_params,
        'best_score': -optimizer.max['target'],  # Convert back to positive average SSD
        'optimizer': optimizer,
    }

#########################################################################################################

# # function to generate polynomial features and standardize a given data set
# def polynomial_and_standardize(dataset, poly_degree=3, interaction_only=False):
    
#     poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False)
    
#     X = poly.fit_transform(dataset)
    
#     scaled_X = preprocessing.StandardScaler().fit_transform(X)
    
#     return scaled_X

#########################################################################################################

# # function to synthesize a variable using logistic regression model
# def multinomial_synthesizer(orig_data, synth_data_sets, target, penalty_param, poly_degree=3, interaction_only=False):
    
#     mn_model = LogisticRegression(penalty='l1', C=penalty_param, solver='saga', max_iter=1000, multi_class='multinomial', random_state=rng)
    
#     X = polynomial_and_standardize(dataset=orig_data, poly_degree=poly_degree, interaction_only=interaction_only)
    
#     sXs = [polynomial_and_standardize(dataset=Y, poly_degree=poly_degree, interaction_only=interaction_only) for Y in synth_data_sets]
    
#     vals = []
    
#     mn_model.fit(X, target)
    
#     rng_mn = default_rng()
    
#     for Y in sXs:
        
#         probs = mn_model.predict_proba(Y)
    
#         v = [np.argmax(rng_mn.multinomial(n=1, pvals=p, size=1)==1) for p in probs]
        
#         vals.append(pd.Series(v, name=target.name))
    
#     return vals

#########################################################################################################

# # function to calculate privacy metrics IMS and 5th percentiles of DCR and NNDR distributions
# def privacy_metrics(train_data, synthetic_datasets, type_of_synthetic, delta):

#     # create scaler
#     scaler = preprocessing.StandardScaler()

#     # scale training data
#     train_scaled = scaler.fit_transform(X=train_data)

#     # create tree for nearest neighbor searching
#     training_tree = KDTree(train_scaled)

#     # calculate nearest neighbor distances within training data
#     train_dists, train_neighbors = training_tree.query(x=train_scaled, k=6, p=2)

#     # calculate identical match share
#     # using the second column because we know there is at least one identical record (the record itself)
#     # so we care about the next most similar record (the second nearest neighbor)
#     IMS_train = np.mean(train_dists[:,1] <= delta)

#     # calculate 5th percentile of DCR distribution for synthetic and train data
#     DCR_train = np.percentile(train_dists[:,1], q=5)

#     # calculate nearest neighbor distance ratios
#     ratios_train = train_dists[:,1]/train_dists[:,-1]

#     # we can encounter division by zero in the above ratios. If this occurs, neighbors 1-5 have the same distance (0)
#     # and the ratio can be set to one.
#     ratios_train = np.nan_to_num(ratios_train, nan=1.0)

#     # calculate 5th percentile of nearest neighbor distance ratios
#     NNDR_train = np.percentile(ratios_train, q=5)

#     IMS_synthetic, DCR_synthetic, NNDR_synthetic = [], [], []
    
#     for Z in synthetic_datasets:
        
#         # create scaler
#         scaler = preprocessing.StandardScaler().fit(X=Z)

#         # scale synthetic data using means and standard deviations 
#         synthetic_scaled = scaler.transform(X=Z)
#         train_scaled = scaler.transform(X=train_data)
    
#         # create tree for nearest neighbor searching
#         training_tree = KDTree(train_scaled)

#         # calculate the nearest neighbor distances between synthetic and training data
#         synthetic_dists, synthetic_neighbors = training_tree.query(x=synthetic_scaled, k=5, p=2)

#         # calculate identical match share
#         IMS_synthetic.append(np.mean(synthetic_dists[:,0] <= delta))

#         # calculate 5th percentile of DCR distribution for synthetic and train data
#         DCR_synthetic.append(np.percentile(synthetic_dists[:,0], q=5))

#         # calculate nearest neighbor distance ratios
#         ratios_synthetic = synthetic_dists[:,0]/synthetic_dists[:,-1]

#         # we can encounter division by zero in the above ratios. If this occurs, neighbors 1-5 have the same distance (0)
#         # and the ratio can be set to one.
#         ratios_synthetic = np.nan_to_num(ratios_synthetic, nan=1.0)

#         # calculate 5th percentile of nearest neighbor distance ratios
#         NNDR_synthetic.append(np.percentile(ratios_synthetic, q=5))

#     return (pd.DataFrame({"Type" : np.concatenate([["Train"], np.repeat(type_of_synthetic, len(synthetic_datasets))]), 
#                           "IMS" : np.concatenate([[IMS_train], IMS_synthetic]), 
#                           "DCR" : np.concatenate([[DCR_train], DCR_synthetic]), 
#                           "NNDR" : np.concatenate([[NNDR_train], NNDR_synthetic])}))

#########################################################################################################
#########################################################################################################
#########################################################################################################

# # function to calculate IMS specifically
# def ims_calc(train_data, synthetic_data, delta, synthetic_is_train=False):
    
#     scaler = preprocessing.StandardScaler().fit(X=synthetic_data)
    
#     train_data_scaled = scaler.transform(X=train_data)
    
#     training_tree = KDTree(train_data_scaled)
    
#     synthetic_data_scaled = scaler.transform(X=synthetic_data)
    
#     synthetic_dists, synthetic_neighbors = training_tree.query(x=synthetic_data_scaled, k=2, p=2)

#     if synthetic_is_train:
#         IMS_synthetic = np.mean(synthetic_dists[:,1] <= delta)
#     else:
#         IMS_synthetic = np.mean(synthetic_dists[:,0] <= delta)
    
#     return IMS_synthetic

# # function that applies the above ims_calc function over a set of synthetic data sets for a range of delta values and returns the average for each delta value
# def ims_apply(train_data, synthetic_data_sets, delta_vals, synthetic_is_train=False):
#     ims = [[ims_calc(train_data=train_data, synthetic_data=y, delta=x, synthetic_is_train=synthetic_is_train) for x in delta_vals] for y in synthetic_data_sets]
#     if synthetic_is_train:
#         return ims
#     else:
#         avg_ims = np.mean(np.vstack(ims), axis=0)
#         return avg_ims

#########################################################################################################

# # return point estimates, estimated variance of coefficients, and confidence intervals from OLS
# def ols_param_fetcher(data, y, X):
#     predictors = data.loc[:, X]
#     predictors = add_constant(predictors)
#     state_ols = OLS(endog=data.loc[:, y], exog=predictors)
#     ols_results = state_ols.fit()
#     return {"params": ols_results.params, 
#             "l_var": np.diag(ols_results.cov_params()),
#             "CI": ols_results.conf_int().reset_index(drop=True)}

# # function to calculate the L1 distance, confidence interval ratio, and sign, significance, and overlap metrics
# def coef_L1_calc(original_data, synthetic_datasets, synthetic_data_type, target_variable, exog_variables, param_names):

#     # copy synthetic datasets so they don't get edited on a global scope
#     all_synth = synthetic_datasets.copy()

#     # train a logistic regression model with state as the target and lat, long, sex, age, and sex*age as predictors
#     # function returns all parameter estimates, standard errors, and confidence intervals for the training data
#     ols_train = ols_param_fetcher(data=original_data, y=target_variable, X=exog_variables)

#     # estimate the same logistic regression model for all synthetic data sets and save params, standard errors, and CIs
#     ols_synth = [ols_param_fetcher(data=Y, y=target_variable, X=exog_variables) for Y in synthetic_datasets]

#     # create a dataframe with the L1 distances for each coefficient in the columns, (rows are for each synthetic data set)
#     # and a column identifying the data type
#     l1_frame = pd.DataFrame()

#     # calculate L1 distance
#     for i in ols_synth:
#         l1_frame = pd.concat([l1_frame, np.abs(i['params'] - ols_train['params'])], axis=1)

#     l1_frame = l1_frame.T.reset_index(drop=True)
#     l1_frame.columns = param_names
#     l1_frame['Data Type'] = synthetic_data_type
#     l1_frame['Measure'] = 'L1 Distance'

#     # calculate CI ratio (width of synthetic / width of original)
#     # calculate confidence interval ratios
#     CI_ratio_frame = pd.DataFrame()
#     for i in ols_synth:
#         CI_ratio_frame = pd.concat([CI_ratio_frame, (i['CI'].iloc[:,1]-i['CI'].iloc[:,0]) / (ols_train['CI'].iloc[:,1]-ols_train['CI'].iloc[:,0])], axis=1)

#     CI_ratio_frame = CI_ratio_frame.T.reset_index(drop=True)
#     CI_ratio_frame.columns = param_names
#     CI_ratio_frame['Data Type'] = synthetic_data_type
#     CI_ratio_frame['Measure'] = 'CI Ratio'
    
#     # calculate whether the signs of coefficients match
#     sign_frame = pd.DataFrame()
#     for i in ols_synth:
#         sign_frame = pd.concat([sign_frame, abs(ols_train['params']) + abs(i['params']) == abs(ols_train['params'] + i['params'])], axis=1)

#     sign_frame = sign_frame.T.reset_index(drop=True)
#     sign_frame.columns = param_names
#     sign_frame['Data Type'] = synthetic_data_type
#     sign_frame['Measure'] = 'Sign Match'
    
#     # check whether the statistical significance of the coefficients matches
#     sig_frame = pd.DataFrame()
#     orig_sig = pd.concat([ols_train['CI'].iloc[:,0] <= 0, 0 <= ols_train['CI'].iloc[:,1]], axis=1).all(axis=1)
#     for i in ols_synth:
#         sig_frame = pd.concat([sig_frame, pd.concat([i['CI'].iloc[:,0] <= 0, 0 <= i['CI'].iloc[:,1]], axis=1).all(axis=1).eq(orig_sig, axis=0)], axis=1)

#     sig_frame = sig_frame.T.reset_index(drop=True)
#     sig_frame.columns = param_names
#     sig_frame['Data Type'] = synthetic_data_type
#     sig_frame['Measure'] = 'Significance Match'
    
#     # check whether confidence intervals overlap
#     overlap_frame = pd.DataFrame()
#     for synth in ols_synth:
#         overlaps = []
#         for i,j in synth['CI'].iterrows():
#             i1 = pd.Interval(ols_train['CI'].iloc[i,0], ols_train['CI'].iloc[i,1], closed='both')
#             i2 = pd.Interval(j[0], j[1], closed='both')
#             overlaps.append(i1.overlaps(i2))
#         overlap_frame = pd.concat([overlap_frame, pd.Series(overlaps)], axis=1)

#     overlap_frame = overlap_frame.T.reset_index(drop=True)
#     overlap_frame.columns = param_names
#     overlap_frame['Data Type'] = synthetic_data_type
#     overlap_frame['Measure'] = 'CI Overlap'

#     # create dataframe with the actual point estimates and confidence intervals
#     p_and_i_full = pd.DataFrame()
    
#     for i, Z in enumerate(ols_synth):
#         p_and_i = pd.concat([Z['params'].reset_index(), Z['CI']], axis=1)
#         p_and_i.columns = ['Parameter', 'Point Estimate', 'Lower Bound', 'Upper Bound']
#         p_and_i.loc[:,'Type'] = synthetic_data_type
#         p_and_i.loc[:,'index'] = i
#         p_and_i_full = pd.concat([p_and_i_full, p_and_i], axis=0)

#     p_and_i_full = p_and_i_full.reset_index(drop=True)

#     return pd.concat([l1_frame, CI_ratio_frame, sign_frame, sig_frame, overlap_frame], axis=0), p_and_i_full


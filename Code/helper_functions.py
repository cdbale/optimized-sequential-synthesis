import pandas as pd
import numpy as np
import math
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from statsmodels.tools.tools import add_constant
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

rng = np.random.RandomState(42)
 
####################################################################################

def define_synthesis_steps(train_data, current_param_bounds):
    
    # compute correlation matrix of the train data
    # this will be used to determine the synthesis order of the 'f' variables
    correlation_matrix = train_data.corr()

    # compute the synthesis order by rank-ordering the variables by the sum of absolute values of their cross-correlation coefficients
    # the stronger the correlations, the earlier we synthesize to avoid introducing noise to the more important variables
    tree_synthesis_order = list(np.abs(correlation_matrix).sum()[:12].sort_values(ascending=False).index)

    # synthesis steps
    # written as a list of tuples (features, model)
    synthesis_steps = [
        (['treatment', 'exposure', 'visit', 'conversion'], 'joint_categorical'),
        (tree_synthesis_order, 'tree')]

    # define a list to contain all column names in order of synthesis
    all_cols = synthesis_steps[0][0].copy()

    # add the remaining column names (the 'f' variables)
    [all_cols.append(synthesis_steps[1][0][i]) for i in range(len(synthesis_steps[1][0]))]
    
    all_leaf_sizes = []

    # loop over column indices from f0 to f11, with the categorical variables placed as the first four variables
    for id in range(4, len(all_cols)):
        
        # define X variables
        covariates = train_data.loc[:, all_cols[:id]]

        # define Y variable
        target = train_data.loc[:, all_cols[id]].to_numpy()

        # transform target with Yeo-johnson
        # pt = PowerTransformer(method='yeo-johnson', standardize=True)
        # target = pt.fit_transform(target.to_numpy().reshape(-1, 1)).ravel()

        # define and fit CART model
        # use min_samples_leaf = 1 for a tree unconstrained on leaf size
        tree = DecisionTreeRegressor(min_samples_leaf=1)
        tree.fit(covariates, target)

        # compute leaf assignments
        # obtain leaf assignments for training data
        train_leaves = tree.apply(covariates)
        # count the number of samples in each leaf
        all_leaf_sizes.append(np.unique(train_leaves, return_counts=True)[1])

    # compute the minimum leaf size from each tree
    min_leaf_sizes = pd.Series([np.min(x) for x in all_leaf_sizes], index=all_cols[4:])

    param_bounds = current_param_bounds.copy()

    # define the optimization parameter bounds using the minimum leaf sizes
    for i, j in zip(min_leaf_sizes.index, min_leaf_sizes):
        if j == 1:
            param_bounds['tree'][i]['min_samples_leaf'] = [5, int(train_data.shape[0]/2)]
        elif j == train_data.shape[0]:
            param_bounds['tree'][i]['min_samples_leaf'] = [5, 5]
        else:
            if j < int(train_data.shape[0]/2):
                param_bounds['tree'][i]['min_samples_leaf'] = [int(np.max([5, j])), int(train_data.shape[0]/2)]
            else:
                param_bounds['tree'][i]['min_samples_leaf'] = [j, train_data.shape[0]]

    return all_cols, synthesis_steps, param_bounds

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

        sXs.append(sampled)

    return sXs
 
####################################################################################

# def _synthesize_with_multinomial(X, y, num_samples, synthetic_datasets, C, poly_degree, interaction_only, random_state=None):
#     """
#     Helper function to synthesize categorical data using multinomial logistic regression.
    
#     Parameters:
#     -----------
#     X : pd.DataFrame or None
#         Features for multinomial regression. If None, samples from prior.
#     y : pd.Series
#         Target variable to predict
#     num_samples : int
#         Number of samples to generate
#     synthetic_datasets : list of pd.DataFrame or int
#         Either a list of synthetic datasets or an integer specifying the number of synthetic datasets to generate
#     C : float, default=1.0
#         Inverse of regularization strength
#     poly_degree : int, default=3
#         Degree of polynomial features
#     interaction_only : bool, default=False
#         If true, only main effects and interactions are included
#     random_state : int, default=None
#         Random state for reproducibility
        
#     Returns:
#     --------
#     list: sXs
#         sXs: list of generated synthetic data frames
#     """
#     if random_state is not None:
#         np.random.seed(random_state)

#     # Code to use if target variable is constant, i.e., it doesn't have more than once class
#     if len(y.unique()) == 1:
#         for i, sX in enumerate(synthetic_datasets):
#             synth_data = np.repeat(y.iloc[0], num_samples)
#             synthetic_datasets[i] = pd.concat([synthetic_datasets[i], pd.Series(synth_data, name=y.name)], axis=1)
#         return synthetic_datasets
    
#     # # Code used if this is the first variable being synthesized in the data set
#     # if X is None or X.shape[1] == 0:
#     #     # If no features, sample from prior distribution
#     #     sXs = []
#     #     for i in range(synthetic_datasets):
#     #         counts = y.value_counts(normalize=True)
#     #         synth_data = np.random.choice(
#     #             counts.index,
#     #             size=num_samples,
#     #             p=counts.values
#     #         )
#     #         sXs.append(pd.Series(synth_data, name=y.name))
#     #     return sXs
    
#     # Generate polynomial features
#     poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False)
#     X_poly = poly.fit_transform(X)

#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_poly)
    
#     # Fit multinomial model
#     model = LogisticRegression(
#         penalty='l1',
#         C=C,
#         solver='saga',
#         max_iter=1000,
#         random_state=random_state
#     )
#     model.fit(X_scaled, y)

#     # loop over synthetic datasets
#     for i, sX in enumerate(synthetic_datasets):
#         sX_poly = poly.transform(sX)
#         sX_scaled = scaler.transform(sX_poly)
#         # Get predicted probabilities
#         probs = model.predict_proba(sX_scaled)
        
#         # Sample from multinomial distribution
#         synth_data = [
#             np.argmax(np.random.multinomial(n=1, pvals=p, size=1)) 
#             for p in probs
#             ]
#         synth_data = model.classes_[synth_data]
#         synthetic_datasets[i] = pd.concat([synthetic_datasets[i], pd.Series(synth_data, name=y.name)], axis=1)
    
#     return synthetic_datasets

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
    center = x[i_start:(i_end+1)]

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
    dists =np.maximum(dists, 1e-3)

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
    # pt = PowerTransformer(method='yeo-johnson', standardize=True)
    # y_trans = pt.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

    y_trans = y.to_numpy().copy()
    
    # 2) Fit DecisionTreeRegressor and get leaf assignments
    tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=random_state)
    tree.fit(X, y_trans)

    # Precompute training leaf assignments and values in each leaf
    # note that we are sorting the values in each leaf
    train_leaves = tree.decision_path(X).toarray()
    # store indexes of values in each leaf    
    leaf_vals = {leaf: np.sort(y_trans[np.where(train_leaves[:, leaf] == 1)[0]]) for leaf in range(train_leaves.shape[1])}
    # compute the median distance to four nearest neighbors by index for each training value in each leaf
    leaf_dists = {leaf: ordered_vector_distances(leaf_vals[leaf]) for leaf in range(train_leaves.shape[1])}
    
    # Process each synthetic dataset
    results = []
    rs = np.random.RandomState(random_state)
    
    for i, sX in enumerate(synthetic_datasets):
        
        # Get leaf assignments for synthetic data
        s_leaves = tree.apply(sX[X.columns])
        
        # find unique leaf indexes and count of observations in each unique leaf
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

        # compute minimum and maximum values found in the real data
        # clip the new synthetic values by these values
        # results in slightly truncated distributions of synthetic values but improves outlier protection
        # and prevents NA values in the inverse Yeo-Johnson transform
        # min_val = np.min(y_trans)
        # max_val = np.max(y_trans)
        # min_synth = np.min(y_trans) + sys.float_info.epsilon # + np.abs(rs.normal(loc=0, scale=(max_val-min_val)/2))
        # max_synth = np.max(y_trans) - sys.float_info.epsilon # - np.abs(rs.normal(loc=0, scale=(max_val-min_val)/2))
        # synth_vals = np.clip(synth_vals, min_synth, max_synth)

        null_sum = pd.Series(synth_vals).isnull().sum()
        if null_sum > 0:
            raise ValueError(f"There are {null_sum} missing values in the transformed synthetic variable {y.name}.")

        # Inverse transform and store results
        # synth_orig = pt.inverse_transform(synth_vals.reshape(-1, 1)).ravel()

        synth_orig = synth_vals.copy()

        null_sum = pd.Series(synth_orig).isnull().sum()
        if null_sum > 0:
            raise ValueError(f"There are {null_sum} missing values in the original scale synthetic variable {y.name}.")

        # if we get infinity values in the synthetic variable, replace them with the corresponding min or max value
        # of that same synthetic variable

        # synth_orig[synth_orig == np.inf] = np.max(synth_orig[synth_orig != np.inf])
        # synth_orig[synth_orig == -np.inf] = np.min(synth_orig[synth_orig != -np.inf])

        results.append(pd.concat([sX, pd.Series(synth_orig, name=y.name)], axis=1))
    
    return results

#########################################################################################################

# Helper to compute OLS parameter vector for a given dataset and model spec
def logit_params(X, y):
    """Helper to compute logistic regression parameters with constant term.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Feature matrix (n_samples, n_features)
    y : array-like or Series
        Target variable (binary or multi-class)
        
    Returns:
    --------
    pd.Series
        Series of parameter estimates with feature names as index
    """

    # Create a pipeline that first standardizes the features and then fits logistic regression
    model = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')

    # Fit the model
    model.fit(X, y)
    
    # Get the feature names (excluding the intercept)
    # feature_names = [f'x{i}' for i in range(X.shape[1]+1)]
    feature_names = ['intercept']
    feature_names.extend(X.columns)
    
    # Return intercept and coefficients as a series
    return pd.Series(
        np.concatenate([[model.intercept_[0]], model.coef_.flatten()]),
        index=feature_names
    )

#########################################################################################################

def perform_synthesis_with_param_target(
    train_data,
    number_synthetic_datasets,
    synthesis_steps,
    target_params,
    target_variable,
    exog_variables,
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

    num_samples = train_data.shape[0]
    synthesized_vars = []
    for step_idx, (step_vars, method) in enumerate(synthesis_steps):

        dependencies = []
        for prev_vars, _ in synthesis_steps[:step_idx]:
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

        # elif method == 'multinomial':
        #     for var in step_vars:
        #         var_params = {}
        #         if param_values and 'multinomial' in param_values and var in param_values['multinomial']:
        #             var_params.update(param_values['multinomial'][var])
        #         C = var_params.get('C', {})
        #         X = train_data[dependencies] if dependencies else None
        #         synthetic_datasets = _synthesize_with_multinomial(
        #             X=X,
        #             y=train_data[var],
        #             num_samples=num_samples,
        #             poly_degree=poly_degree_mnl,
        #             interaction_only=interaction_only,
        #             synthetic_datasets=synthetic_datasets,
        #             C=C,
        #             random_state=random_state,
        #         )
        elif method == 'tree':
            for var in step_vars:
                var_params = {}
                if param_values and 'tree' in param_values and var in param_values['tree']:
                    var_params.update(param_values['tree'][var])
                min_samples_leaf = int(var_params.get('min_samples_leaf'))
                X = train_data[dependencies] if dependencies else None
                synthetic_datasets = synthesize_with_tree_regressor(
                    X=X,
                    y=train_data[var],
                    num_samples=num_samples,
                    synthetic_datasets=synthetic_datasets,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
                null_sums = [X.isnull().sum() for X in synthetic_datasets]
                if np.sum(null_sums) > 0:
                    raise ValueError(f"There are {np.sum(null_sums)} missing values in the synthetic datasets produced for variable {var}.")
                dependencies.append(var)
        else:
            raise ValueError(f"Unknown synthesis method: {method}")

        synthesized_vars.extend(step_vars)

    # Compute SSD per synthetic dataset
    # Ensure target_params is aligned to the params index order
    target_series = pd.Series(target_params)
    # standardize target_series
    target_series_std = (target_series - target_series.mean()) / target_series.std()
    ssds = []
    for sX in synthetic_datasets:
        # Get features and target from the synthetic dataset
        X_synth = sX[exog_variables]
        y_synth = sX[target_variable]
        
        # Get logistic regression parameters
        params = logit_params(X_synth, y_synth)
        
        # Align indices with target parameters and compute SSD
        aligned = params.reindex(target_series.index)  # Fill missing with 0
        # standardize aligned series
        aligned = (aligned - target_series.mean()) / target_series.std()
        ssds.append(float(np.sum((aligned.values - target_series_std.values) ** 2)))

    return ssds, synthetic_datasets

#########################################################################################################

def optimize_models_with_param_target(
    train_data,
    number_synthetic_datasets,
    synthesis_steps,
    target_params,
    target_variable,
    exog_variables,
    param_bounds,
    random_state=None,
    n_iter=25,
    n_init=5
):
    """
    Optimize synthesis parameters to minimize average SSD between synthetic OLS params and target_params.
    
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

    # # Flatten param_bounds into dimensions and param_mapping
    # if 'gmm' in param_bounds:
    #     for param, bounds in param_bounds['gmm'].items():
    #         if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
    #             dim_name = f"gmm_{param}"
    #             dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
    #             param_mapping.append(('gmm', param))
    
    # Check for tree parameters
    if 'tree' in param_bounds:
        # # Handle global tree parameters (applied to all variables)
        # if 'global' in param_bounds['tree']:
        #     for param, bounds in param_bounds['tree']['global'].items():
        #         if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        #             dim_name = f"tree_global_{param}"
        #             dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
        #             param_mapping.append(('tree', 'global', param))
        
        # Handle per-variable tree parameters (overrides global if both exist)
        for var, var_bounds in param_bounds['tree'].items():
            if var != 'global' and isinstance(var_bounds, dict):
                for param, bounds in var_bounds.items():
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        dim_name = f"tree_{var}_{param}"
                        dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
                        param_mapping.append(('tree', var, param))
    
    # # Check for multinomial parameters (if needed)
    # if 'multinomial' in param_bounds:
    #     for var, var_bounds in param_bounds['multinomial'].items():
    #         if isinstance(var_bounds, dict):
    #             for param, bounds in var_bounds.items():
    #                 if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
    #                     dim_name = f"multinomial_{var}_{param}"
    #                     dimensions.append((bounds[0], bounds[1], 'uniform', dim_name))
    #                     param_mapping.append(('multinomial', var, param))

    def evaluate_models(**kwargs):
        pv = {'gmm': {}, 'multinomial': {}, 'tree': {}}
        for i, (method, *rest) in enumerate(param_mapping):
            param_name = f"x{i}"
            if param_name in kwargs:
                if method == 'gmm':
                    param, = rest
                    pv['gmm'][param] = int(kwargs[param_name])
                elif method in ['multinomial', 'tree']:
                    _, param = rest
                    if synthesis_steps[1][0][i] not in pv[method]:
                        pv[method][synthesis_steps[1][0][i]] = {}
                    pv[method][synthesis_steps[1][0][i]][param] = kwargs[param_name]

        ssds, _ = perform_synthesis_with_param_target(
            train_data=train_data,
            number_synthetic_datasets=number_synthetic_datasets,
            synthesis_steps=synthesis_steps,
            target_params=target_params,
            target_variable=target_variable,
            exog_variables=exog_variables,
            param_values=pv,
            random_state=random_state,
        )

        # Maximize negative average SSD
        return -np.mean(ssds)

    pbounds = {f"x{i}": (low, high) for i, (low, high, _, _) in enumerate(dimensions)}

    if not pbounds:
        raise ValueError("No valid parameter bounds found. Please check your param_bounds structure. "
                         "It should be a dictionary with 'gmm', 'tree', or 'multinomial' keys, "
                         "each containing parameter bounds for the respective method.")
    
    print(f'Optimizing {len(pbounds)} parameters: {list(pbounds.keys())}')

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
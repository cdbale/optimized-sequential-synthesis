import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.random import default_rng
from scipy.spatial import KDTree
from scipy.stats import t
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS

rng = np.random.RandomState(42)

#########################################################################################################
#########################################################################################################
#########################################################################################################

# function to compute the pMSE ratio from a given original and synthetic data set
def pmse_ratio(original_data, synthetic_data):
    
    N_synth = synthetic_data.shape[0]
    N_orig = original_data.shape[0]
    
    # combine original and synthetic datasets
    full_X = pd.concat([original_data, synthetic_data], axis=0).reset_index(drop=True)
    
    # generate interactions and powers of variables
    poly = PolynomialFeatures(3, interaction_only=False, include_bias=False)
    
    full_X = poly.fit_transform(full_X)

    # scale the combined dataset
    full_X = preprocessing.StandardScaler().fit_transform(full_X)
    
    c = N_synth/(N_synth+N_orig)

    y = np.repeat([0, 1], repeats=[N_orig, N_synth])
    
    pMSE_model = LogisticRegression(penalty=None, max_iter=1000).fit(full_X, y)
    
    probs = pMSE_model.predict_proba(full_X)
    
    pMSE = 1/(N_synth+N_orig) * np.sum((probs[:,1] - c)**2)

    # didn't subtract one from degrees of freedom to account for the intercept
    e_pMSE = 2*(full_X.shape[1])*(1-c)**2 * c/(N_synth+N_orig)
        
    return pMSE/e_pMSE

#########################################################################################################
#########################################################################################################
#########################################################################################################

# function to generate polynomial features and standardize a given data set
def polynomial_and_standardize(dataset, poly_degree=3, interaction_only=False):
    
    poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False)
    
    X = poly.fit_transform(dataset)
    
    scaled_X = preprocessing.StandardScaler().fit_transform(X)
    
    return scaled_X

#########################################################################################################
#########################################################################################################
#########################################################################################################

# function to synthesize a variable using logistic regression model
def multinomial_synthesizer(orig_data, synth_data_sets, target, penalty_param, poly_degree=3, interaction_only=False):
    
    mn_model = LogisticRegression(penalty='l1', C=penalty_param, solver='saga', max_iter=1000, multi_class='multinomial', random_state=rng)
    
    X = polynomial_and_standardize(dataset=orig_data, poly_degree=poly_degree, interaction_only=interaction_only)
    
    sXs = [polynomial_and_standardize(dataset=Y, poly_degree=poly_degree, interaction_only=interaction_only) for Y in synth_data_sets]
    
    vals = []
    
    mn_model.fit(X, target)
    
    rng_mn = default_rng()
    
    for Y in sXs:
        
        probs = mn_model.predict_proba(Y)
    
        v = [np.argmax(rng_mn.multinomial(n=1, pvals=p, size=1)==1) for p in probs]
    
        vals.append(pd.Series(v, name=target.name))
    
    return vals

#########################################################################################################
#########################################################################################################
#########################################################################################################

# function to calculate privacy metrics IMS and 5th percentiles of DCR and NNDR distributions
def privacy_metrics(train_data, synthetic_datasets, type_of_synthetic, delta):

    # create scaler
    scaler = preprocessing.StandardScaler()

    # scale training data
    train_scaled = scaler.fit_transform(X=train_data)

    # create tree for nearest neighbor searching
    training_tree = KDTree(train_scaled)

    # calculate nearest neighbor distances within training data
    train_dists, train_neighbors = training_tree.query(x=train_scaled, k=6, p=2)

    # calculate identical match share
    # using the second column because we know there is at least one identical record (the record itself)
    # so we care about the next most similar record (the second nearest neighbor)
    IMS_train = np.mean(train_dists[:,1] <= delta)

    # calculate 5th percentile of DCR distribution for synthetic and train data
    DCR_train = np.percentile(train_dists[:,1], q=5)

    # calculate nearest neighbor distance ratios
    ratios_train = train_dists[:,1]/train_dists[:,-1]

    # we can encounter division by zero in the above ratios. If this occurs, neighbors 1-5 have the same distance (0)
    # and the ratio can be set to one.
    ratios_train = np.nan_to_num(ratios_train, nan=1.0)

    # calculate 5th percentile of nearest neighbor distance ratios
    NNDR_train = np.percentile(ratios_train, q=5)

    IMS_synthetic, DCR_synthetic, NNDR_synthetic = [], [], []
    
    for Z in synthetic_datasets:
        
        # create scaler
        scaler = preprocessing.StandardScaler().fit(X=Z)

        # scale synthetic data using means and standard deviations 
        synthetic_scaled = scaler.transform(X=Z)
        train_scaled = scaler.transform(X=train_data)
    
        # create tree for nearest neighbor searching
        training_tree = KDTree(train_scaled)

        # calculate the nearest neighbor distances between synthetic and training data
        synthetic_dists, synthetic_neighbors = training_tree.query(x=synthetic_scaled, k=5, p=2)

        # calculate identical match share
        IMS_synthetic.append(np.mean(synthetic_dists[:,0] <= delta))

        # calculate 5th percentile of DCR distribution for synthetic and train data
        DCR_synthetic.append(np.percentile(synthetic_dists[:,0], q=5))

        # calculate nearest neighbor distance ratios
        ratios_synthetic = synthetic_dists[:,0]/synthetic_dists[:,-1]

        # we can encounter division by zero in the above ratios. If this occurs, neighbors 1-5 have the same distance (0)
        # and the ratio can be set to one.
        ratios_synthetic = np.nan_to_num(ratios_synthetic, nan=1.0)

        # calculate 5th percentile of nearest neighbor distance ratios
        NNDR_synthetic.append(np.percentile(ratios_synthetic, q=5))

    return (pd.DataFrame({"Type" : np.concatenate([["Train"], np.repeat(type_of_synthetic, len(synthetic_datasets))]), 
                          "IMS" : np.concatenate([[IMS_train], IMS_synthetic]), 
                          "DCR" : np.concatenate([[DCR_train], DCR_synthetic]), 
                          "NNDR" : np.concatenate([[NNDR_train], NNDR_synthetic])}))


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
#         scaler = preprocessing.StandardScaler()
        
#         # scale synthetic data using means and standard deviations 
#         synthetic_scaled = scaler.fit(X=train_data).transform(X=Z)

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

# function to calculate IMS specifically
def ims_calc(train_data, synthetic_data, delta, synthetic_is_train=False):
    
    scaler = preprocessing.StandardScaler().fit(X=synthetic_data)
    
    train_data_scaled = scaler.transform(X=train_data)
    
    training_tree = KDTree(train_data_scaled)
    
    synthetic_data_scaled = scaler.transform(X=synthetic_data)
    
    synthetic_dists, synthetic_neighbors = training_tree.query(x=synthetic_data_scaled, k=2, p=2)

    if synthetic_is_train:
        IMS_synthetic = np.mean(synthetic_dists[:,1] <= delta)
    else:
        IMS_synthetic = np.mean(synthetic_dists[:,0] <= delta)
    
    return IMS_synthetic

# function that applies the above ims_calc function over a set of synthetic data sets for a range of delta values and returns the average for each delta value
def ims_apply(train_data, synthetic_data_sets, delta_vals, synthetic_is_train=False):
    ims = [[ims_calc(train_data=train_data, synthetic_data=y, delta=x, synthetic_is_train=synthetic_is_train) for x in delta_vals] for y in synthetic_data_sets]
    if synthetic_is_train:
        return ims
    else:
        avg_ims = np.mean(np.vstack(ims), axis=0)
        return avg_ims

#########################################################################################################
#########################################################################################################
#########################################################################################################

def attribute_disclosure_reduction(original_data, synthetic_data, continuous_vars, categorical_vars, sensitive_var, num_mixture_components, deltas, c, prior_prob):

    # assume the original and synthetic data are passed on their original scales

    # number of original records
    num_records = original_data.shape[0]

    # random number generator
    rng = default_rng()

    # copy the synthetic dataset
    new_sX = synthetic_data.copy()

    # normalizer
    # normalize synthetic data
    scaler = preprocessing.StandardScaler().fit(X=new_sX.loc[:, continuous_vars])
    new_sX.loc[:, continuous_vars] = scaler.transform(X=new_sX.loc[:, continuous_vars])

    # normalize original data
    original_scaled = original_data.copy()
    original_scaled.loc[:, continuous_vars] = scaler.transform(X=original_scaled.loc[:, continuous_vars])

    # fit a new GMM to the original_scaled data (use optimal parameters from synthesis)
    mixture_model = GaussianMixture(num_mixture_components, n_init=2, covariance_type='full', init_params="k-means++").fit(original_scaled.loc[:, continuous_vars])

    # store mixture component parameters
    mus = mixture_model.means_
    sigmas = mixture_model.covariances_

    # tree of synthetic records (continuous values only)
    sX_tree = KDTree(new_sX[continuous_vars])

    # temporary count of the number of rows that violate one or more conditions
    violator_count = num_records

    # number of anonymization loops required
    num_loops = 1

    # list all categorical variables including sensitive variable
    all_categorical = categorical_vars + [sensitive_var]

    # as long as the last loop had a violator, we need to recheck all records
    while violator_count > 0:

        # reset violator count
        violator_count = 0

        # loop over delta values
        for delta in deltas:

            # shuffle the original data
            original_scaled = original_scaled.sample(frac=1.0, ignore_index=True)
            # # order the non-normalized original data the same way
            # original_data = original_data.loc[original_scaled.index,:]

            # reset the indexes
            # original_scaled = original_scaled.reset_index(drop=True)
            # original_data = original_data.reset_index(drop=True)

            # split original records to preserve memory when doing nearest neighbor searches - do computations on each portion
            for subset_id, subset in enumerate(np.array_split(original_scaled, 5)):

                # compute the neighbors within radius delta based on continuous variables only
                neighbor_indices = sX_tree.query_ball_point(subset[continuous_vars], r=delta, p=2.0)

                # store the neighbors for each original record
                neighbor_records = [new_sX.loc[y, all_categorical] for y in neighbor_indices]

                # calculate the proportion of non-white observations amongst the synthetic neighbors
                # replace NA values (there were no neighbors) with the prior probability
                prop_pos_sensitive = np.nan_to_num([np.mean(y.loc[y.loc[:, categorical_vars].eq(subset.loc[:, categorical_vars].iloc[x,:]).all(axis=1), sensitive_var]) for x,y in enumerate(neighbor_records)], nan=prior_prob)

                # compute the updated probability based on the proportion of neighborhood records that match the sensitive value
                updated_probs = np.where(subset.loc[:, sensitive_var] == 1, np.array(prop_pos_sensitive), 1-np.array(prop_pos_sensitive))

                # vector of prior probability for each record
                prior_probs = np.where(subset.loc[:, sensitive_var] == 1, prior_prob, 1-prior_prob)

                # compute the attribute disclosure prevention condition
                condition = updated_probs/prior_probs

                # indexes of records that violate the condition
                violator_indices = list(np.where(condition > c)[0])

                # if we have violating records
                if len(violator_indices) > 0:

                    # increment the count of violating records
                    violator_count += len(violator_indices)

                    # for each violating record, how many neighbors match on the non-sensitive variables
                    num_matching_non_sensitive = np.array([np.sum(neighbor_records[x].loc[:, categorical_vars].eq(subset.loc[:, categorical_vars].iloc[x,:]).all(axis=1)) for x in violator_indices])

                    # for each violating record, how many neighbors match on all variables
                    num_matching_all = np.array([np.sum(neighbor_records[x].loc[:, all_categorical].eq(subset.loc[:, all_categorical].iloc[x,:]).all(axis=1)) for x in violator_indices])

                    # how many new records in the neighborhood are needed to meet the condition
                    num_needed = np.ceil(num_matching_all/(prior_prob*c) - num_matching_non_sensitive).astype(int)

                    # loop over the indices of violating records
                    for i,j in enumerate(violator_indices):

                        # variables names
                        original_record = subset.iloc[j,:]
                        original_continuous = pd.DataFrame(original_record[continuous_vars]).T
                        original_cat = original_record[all_categorical]

                        # find the component with the highest responsibility for the violating record
                        component_index = np.argmax(mixture_model.predict_proba(pd.DataFrame(original_continuous, columns=continuous_vars)), axis = 1)[0]
                        current_mu = mus[component_index,:]
                        current_sigma = sigmas[component_index,:,:]

                        # numpy array for storing new records
                        valid_candidates = np.zeros((0,len(continuous_vars)))
                        # track number of while loops needed to generate enough new records
                        num_candidate_loops = 0
                        # as long as we have fewer new records than needed
                        while valid_candidates.shape[0] < num_needed[i]:
                            # generate a bunch of candidate points
                            candidate_points = pd.DataFrame(rng.multivariate_normal(current_mu, current_sigma, size=100000), columns=continuous_vars)
                            # check for IPUMS years_of_educ variable - round if exists
                            if 'years_of_educ' in continuous_vars:
                                candidate_points = pd.DataFrame(scaler.inverse_transform(X=candidate_points), columns=continuous_vars)
                                candidate_points.loc[:, ['years_of_educ']] = np.round(candidate_points.loc[:, ['years_of_educ']], 0)
                                candidate_points = pd.DataFrame(scaler.transform(X=candidate_points), columns=continuous_vars)
                            candidate_tree = KDTree(candidate_points)
                            valid_indices = candidate_tree.query_ball_point(original_continuous, delta, p=2.0, return_sorted=True)[0]
                            valid_candidates = np.vstack([valid_candidates, candidate_points.iloc[valid_indices,:]])
                            num_candidate_loops += 1
                            if num_candidate_loops > 100:
                                print('Stuck in inference loop.')

                        # select the number of needed candidates and create the new records
                        new_locations = valid_candidates[:num_needed[i],:]
                        new_categorical = np.vstack([np.array(original_cat).reshape(1,-1) for k in range(num_needed[i])])
                        new_records = pd.DataFrame(np.hstack([new_locations, new_categorical]), columns=new_sX.columns)

                        # edit sensitive variable value to meet the condition
                        new_records[sensitive_var] = 1.0 - original_cat[sensitive_var]

                        # add new records to synthetic data
                        new_sX = pd.concat([new_sX, new_records], axis=0).reset_index(drop=True)
                    
                        # rebuild the tree for synthetic locations
                        sX_tree = KDTree(new_sX[continuous_vars])

                print('Subset ' + str(subset_id) + ' done.')

            print('Completed for delta = ' + str(delta))

        print("Full anonymization loop " + str(num_loops) + " completed.")
        
        num_loops += 1

    print('Completed AD reduction.')

    new_sX.loc[:, continuous_vars] = scaler.inverse_transform(X=new_sX.loc[:, continuous_vars])

    return new_sX

# # function to apply attribute disclosure prevention algorithm
# def attribute_disclosure_reduction(original_data, synthetic_data, continuous_vars, categorical_vars, sensitive_var, mixture_model, deltas, c, prior_prob):
    
#     # # number of original records
#     # num_records = original_data.shape[0]
    
#     # # record percentages
#     # print_nums = [int(np.ceil(i*num_records)) for i in [0.25, 0.5, 0.75]]
    
#     # # random number generator
#     # rng = default_rng()
    
#     # # copy the synthetic dataset
#     # new_sX = synthetic_data
    
#     # # tree for synthetic locations
#     # sX_tree = KDTree(new_sX[continuous_vars])
    
#     # # store mixture component parameters
#     # mus = mixture_model.means_
#     # sigmas = mixture_model.covariances_
    
#     # temporary count of the number of rows that violate one or more conditions
#     violator_count = num_records
    
#     # # number of anonymization loops required
#     # num_loops = 1

#     # all_categorical = categorical_vars.copy()
#     # all_categorical.append(sensitive_var)
    
#     # while we have any violator rows
#     while violator_count > 0:
        
#         # # reset violator count
#         # violator_count = 0
        
#         # for each original record
#         # we shuffle the records each time so that the violating records are fixed in a random order
#         for i, original_record in original_data.sample(frac=1.0).reset_index(drop=True).iterrows():
            
#             original_location = original_record.loc[continuous_vars]
#             original_categorical = original_record.loc[all_categorical]
            
#             # for each delta
#             for delta in deltas:
                    
#                 ##### Test the Inference Criterion
                
#                 # # find synthetic neighbors based on location
#                 # location_neighbors = sX_tree.query_ball_point(original_location, r=delta, p=2.0)
                
#                 # # matches on categorical attributes from location neighbors
#                 # categorical_matches = (new_sX.loc[location_neighbors, categorical_vars] == original_categorical[categorical_vars]).all(1)
                
#                 # matching_rows = new_sX.loc[location_neighbors,:].loc[categorical_matches.values,:]
                
#                 # if there are any records in the location neighborhood that match on sex and age
                
#                 # if matching_rows.shape[0] > 0:
                
#                 #     if original_categorical[sensitive_var] == 1.0:
#                 #         prior = prior_prob
#                 #     else:
#                 #         prior = 1-prior_prob
                        
#                 #     num_matching = np.sum(matching_rows[sensitive_var] == original_categorical[sensitive_var])
            
#                 #     cond = num_matching/matching_rows.shape[0] * 1/prior
                
#                     # if cond > c:
                        
#                     #     # add one to number of violators
#                     #     violator_count += 1
                        
#                     #     # number of records with non-matching sensitive variable needed to meet inference
#                     #     num_needed = int(np.ceil(num_matching/(prior*c) - matching_rows.shape[0]))
                        
#                         # find the component with the highest responsibility for the confidential record
#                         component_index = np.argmax(mixture_model.predict_proba(pd.DataFrame(original_location).T), axis = 1)[0]
#                         current_mu = mus[component_index,:]
#                         current_sigma = sigmas[component_index,:,:]
            
#                         valid_candidates = np.zeros((0,len(continuous_vars)))
#                         num_candidate_loops = 0
#                         while valid_candidates.shape[0] < num_needed:
                        
#                             # generate a bunch of candidate points
#                             candidate_points = rng.multivariate_normal(current_mu, current_sigma, size=100000)
#                             candidate_tree = KDTree(candidate_points)
#                             valid_indices = candidate_tree.query_ball_point(original_location, delta, p=2.0, return_sorted=True)
#                             valid_candidates = np.vstack([valid_candidates, candidate_points[valid_indices,:]])
#                             num_candidate_loops += 1
#                             if num_candidate_loops > 100:
#                                 print('Stuck in inference loop.')
                    
#                         # select the number of needed candidates
#                         new_locations = valid_candidates[:num_needed,:]
                    
#                         new_categorical = np.vstack([np.array(original_categorical).reshape(1,-1) for k in range(num_needed)])
                    
#                         new_records = pd.DataFrame(np.hstack([new_locations, new_categorical]))
                    
#                         new_records.columns = new_sX.columns
                        
#                         new_records[sensitive_var] = 1.0 - original_categorical[sensitive_var]
                    
#                         new_sX = pd.concat([new_sX, new_records], axis=0).reset_index(drop=True)
                    
#                         # rebuild the tree for synthetic locations
#                         sX_tree = KDTree(new_sX[continuous_vars])
    
#             if int(i) in print_nums:
#                 print("Record " + str(i) + " completed.")
                
#         print("Full anonymization loop " + str(num_loops) + " completed.")
        
#         num_loops += 1
                    
#     return new_sX

#########################################################################################################
#########################################################################################################
#########################################################################################################

# function to assess risk of attribute disclosure for IPUMS data
def attribute_disclosure_evaluation(original_data, synthetic_data, continuous_vars, categorical_vars, sensitive_var, prior_prob, deltas):
    
    # list for all attribute disclosure conditions
    full_ad_conds = []

    # fit scaler to synthetic data
    # scale synthetic data
    scaler = preprocessing.StandardScaler().fit(X=synthetic_data.loc[:, continuous_vars])
    synthetic_scaled = synthetic_data.copy() 
    synthetic_scaled.loc[:, continuous_vars] = scaler.transform(X=synthetic_data.loc[:, continuous_vars])

    # scale original data using statistics from synthetic data
    original_scaled = original_data.copy() 
    original_scaled.loc[:, continuous_vars] = scaler.transform(X=original_data.loc[:, continuous_vars])
    
    # tree for synthetic locations
    sX_tree = KDTree(synthetic_scaled[continuous_vars])
    
    # for each value of delta
    for d in deltas:

        # lists to store the inference condition for each original row in the subset
        ad_conds = []

        # split original records to preserve memory when doing nearest neighbor searches - do computations on each portion
        for subset_id, subset in enumerate(np.array_split(original_scaled, 7)):

            # reset row indexes for interation
            subset = subset.reset_index(drop=True)

            # tree for original locations
            orig_subset_tree = KDTree(subset[continuous_vars])
    
            # find synthetic neighbors of each original point
            location_neighbors = orig_subset_tree.query_ball_tree(sX_tree, r=d, p=2.0)
    
            # for each original record
            for i, row in subset.iterrows():
        
                # matches on categorical attributes from location neighbors
                categorical_matches = (synthetic_scaled.loc[location_neighbors[i], categorical_vars] == row[categorical_vars]).all(1)
            
                matching_rows = synthetic_scaled.loc[location_neighbors[i],:].loc[categorical_matches.values,:]
            
                if matching_rows.shape[0] > 0:
                
                    if row[sensitive_var] == 1.0:
                        prior = prior_prob
                    else:
                        prior = 1 - prior_prob
            
                    cond = np.mean(matching_rows[sensitive_var] == row[sensitive_var])/prior
                
                else:
                
                    cond = 1
        
                # store condition
                ad_conds.append(cond)
        
        ad_conds = pd.Series(ad_conds)
        
        full_ad_conds.append(ad_conds)
        
    print("Dataset completed.")
        
    return full_ad_conds

# # function to assess risk of attribute disclosure for IPUMS data
# def attribute_disclosure_evaluation(original_data, synthetic_data, continuous_vars, categorical_vars, sensitive_var, prior_prob, deltas):
    
#     full_ad_conds = []
#     full_indices = []

#     scaler = preprocessing.StandardScaler().fit(X=synthetic_data.loc[:, continuous_vars])
#     synthetic_scaled = synthetic_data.copy() 
#     synthetic_scaled.loc[:, continuous_vars] = scaler.transform(X=synthetic_data.loc[:, continuous_vars])

#     original_scaled = original_data.copy() 
#     original_scaled.loc[:, continuous_vars] = scaler.transform(X=original_data.loc[:, continuous_vars])
    
#     # tree for original locations
#     orig_tree = KDTree(original_scaled[continuous_vars])
    
#     # tree for synthetic locations
#     sX_tree = KDTree(synthetic_scaled[continuous_vars])
    
#     for d in deltas:
        
#         # lists to store the inference condition for each original row and the indices of those rows that violate
#         ad_conds = []
    
#         # find synthetic neighbors of each original point
#         location_neighbors = orig_tree.query_ball_tree(sX_tree, r=d, p=2.0)
    
#         # for each original record
#         for i, row in original_scaled.iterrows():
        
#             # matches on categorical attributes from location neighbors
#             categorical_matches = (synthetic_scaled.loc[location_neighbors[i], categorical_vars] == row[categorical_vars]).all(1)
            
#             matching_rows = synthetic_scaled.loc[location_neighbors[i],:].loc[categorical_matches.values,:]
            
#             if matching_rows.shape[0] > 0:
                
#                 if row[sensitive_var] == 1.0:
#                     prior = prior_prob
#                 else:
#                     prior = 1 - prior_prob
            
#                 cond = np.mean(matching_rows[sensitive_var] == row[sensitive_var])/prior
                
#             else:
                
#                 cond = 1
        
#             # store number of matches and their indices
#             ad_conds.append(cond)
        
#         ad_conds = pd.Series(ad_conds)
        
#         full_ad_conds.append(ad_conds)
        
#     print("Dataset completed.")
        
#     return full_ad_conds

# return point estimates, estimated variance of coefficients, and confidence intervals from OLS
def ols_param_fetcher(data, y, X):
    predictors = data.loc[:, X]
    predictors = add_constant(predictors)
    state_ols = OLS(endog=data.loc[:, y], exog=predictors)
    ols_results = state_ols.fit()
    return {"params": ols_results.params, 
            "l_var": np.diag(ols_results.cov_params()),
            "CI": ols_results.conf_int().reset_index(drop=True)}

# function to calculate the L1 distance, confidence interval ratio, and sign, significance, and overlap metrics
def coef_L1_calc(original_data, synthetic_datasets, synthetic_data_type, target_variable, exog_variables, param_names):

    # copy synthetic datasets so they don't get edited on a global scope
    all_synth = synthetic_datasets.copy()

    # train a logistic regression model with state as the target and lat, long, sex, age, and sex*age as predictors
    # function returns all parameter estimates, standard errors, and confidence intervals for the training data
    ols_train = ols_param_fetcher(data=original_data, y=target_variable, X=exog_variables)

    # estimate the same logistic regression model for all synthetic data sets and save params, standard errors, and CIs
    ols_synth = [ols_param_fetcher(data=Y, y=target_variable, X=exog_variables) for Y in synthetic_datasets]

    # create a dataframe with the L1 distances for each coefficient in the columns, (rows are for each synthetic data set)
    # and a column identifying the data type
    l1_frame = pd.DataFrame()

    # calculate L1 distance
    for i in ols_synth:
        l1_frame = pd.concat([l1_frame, np.abs(i['params'] - ols_train['params'])], axis=1)

    l1_frame = l1_frame.T.reset_index(drop=True)
    l1_frame.columns = param_names
    l1_frame['Data Type'] = synthetic_data_type
    l1_frame['Measure'] = 'L1 Distance'

    # calculate CI ratio (width of synthetic / width of original)
    # calculate confidence interval ratios
    CI_ratio_frame = pd.DataFrame()
    for i in ols_synth:
        CI_ratio_frame = pd.concat([CI_ratio_frame, (i['CI'].iloc[:,1]-i['CI'].iloc[:,0]) / (ols_train['CI'].iloc[:,1]-ols_train['CI'].iloc[:,0])], axis=1)

    CI_ratio_frame = CI_ratio_frame.T.reset_index(drop=True)
    CI_ratio_frame.columns = param_names
    CI_ratio_frame['Data Type'] = synthetic_data_type
    CI_ratio_frame['Measure'] = 'CI Ratio'
    
    # calculate whether the signs of coefficients match
    sign_frame = pd.DataFrame()
    for i in ols_synth:
        sign_frame = pd.concat([sign_frame, abs(ols_train['params']) + abs(i['params']) == abs(ols_train['params'] + i['params'])], axis=1)

    sign_frame = sign_frame.T.reset_index(drop=True)
    sign_frame.columns = param_names
    sign_frame['Data Type'] = synthetic_data_type
    sign_frame['Measure'] = 'Sign Match'
    
    # check whether the statistical significance of the coefficients matches
    sig_frame = pd.DataFrame()
    orig_sig = pd.concat([ols_train['CI'].iloc[:,0] <= 0, 0 <= ols_train['CI'].iloc[:,1]], axis=1).all(axis=1)
    for i in ols_synth:
        sig_frame = pd.concat([sig_frame, pd.concat([i['CI'].iloc[:,0] <= 0, 0 <= i['CI'].iloc[:,1]], axis=1).all(axis=1).eq(orig_sig, axis=0)], axis=1)

    sig_frame = sig_frame.T.reset_index(drop=True)
    sig_frame.columns = param_names
    sig_frame['Data Type'] = synthetic_data_type
    sig_frame['Measure'] = 'Significance Match'
    
    # check whether confidence intervals overlap
    overlap_frame = pd.DataFrame()
    for synth in ols_synth:
        overlaps = []
        for i,j in synth['CI'].iterrows():
            i1 = pd.Interval(ols_train['CI'].iloc[i,0], ols_train['CI'].iloc[i,1], closed='both')
            i2 = pd.Interval(j[0], j[1], closed='both')
            overlaps.append(i1.overlaps(i2))
        overlap_frame = pd.concat([overlap_frame, pd.Series(overlaps)], axis=1)

    overlap_frame = overlap_frame.T.reset_index(drop=True)
    overlap_frame.columns = param_names
    overlap_frame['Data Type'] = synthetic_data_type
    overlap_frame['Measure'] = 'CI Overlap'

    # create dataframe with the actual point estimates and confidence intervals
    p_and_i_full = pd.DataFrame()
    
    for i, Z in enumerate(ols_synth):
        p_and_i = pd.concat([Z['params'].reset_index(), Z['CI']], axis=1)
        p_and_i.columns = ['Parameter', 'Point Estimate', 'Lower Bound', 'Upper Bound']
        p_and_i.loc[:,'Type'] = synthetic_data_type
        p_and_i.loc[:,'index'] = i
        p_and_i_full = pd.concat([p_and_i_full, p_and_i], axis=0)

    p_and_i_full = p_and_i_full.reset_index(drop=True)

    return pd.concat([l1_frame, CI_ratio_frame, sign_frame, sig_frame, overlap_frame], axis=0), p_and_i_full

# # function to combine point estimates and confidence intervals across synthetic data sets and 
# # compare them to those from the original data
# def combined_estimates(synthetic_estimates, original_estimates, type, num_synthetic_datasets, n, nsynth):
    
#     # number of synthetic datasets
#     m = num_synthetic_datasets

#     # the combined point estimate (q-bar_m)
#     qms = []
#     for i in ['const', 'latitude', 'longitude', 'sex', 'age']:
#         qms.append(np.mean([j['params'][i] for j in synthetic_estimates]))

#     # the values for b_m
#     bms = []
#     for i, j in enumerate(['const', 'latitude', 'longitude', 'sex', 'age']):
#         bms.append(np.sum([(1/(m-1))*(k['params'][j]-qms[i])**2 for k in synthetic_estimates]))

#     # the values for u-bar_m
#     ums = []
#     for i, j in enumerate(['const', 'latitude', 'longitude', 'sex', 'age']):
#         ums.append(np.mean([j['l_var'][i] for j in synthetic_estimates]))

#     # calculate variance of qms estimates
#     Tf = (1 + 1/m) * np.array(bms) - np.array(ums)
#     delta = 1*(Tf<0)
#     Tfstar = (Tf>0) * Tf + delta*((nsynth/n) * np.array(ums))

#     # calculate degrees of freedom
#     vf = (m - 1) * (1 - np.array(ums)/((1 + 1/m) * np.array(bms)))**2

#     # upper and lower CI bounds
#     upper = pd.Series(np.array(qms) + t.ppf(q=0.975, df=vf)*np.sqrt(Tfstar))
#     lower = pd.Series(np.array(qms) - t.ppf(q=0.975, df=vf)*np.sqrt(Tfstar))

#     cis = pd.concat([pd.Series(qms), lower, upper], axis=1)

#     cis.columns = ['Point Estimate', 'Lower', 'Upper']

#     cis['Type'] = type

#     # calculate the L1 distance for the point estimates
#     cis['L1 Point Estimate'] = np.abs(cis['Point Estimate'] - original_estimates['params'].reset_index(drop=True))

#     # calculate the confidence interval ratio (synthetic width / original width)
#     cis['Synthetic CI Width'] = cis['Upper'] - cis['Lower']
#     cis['Original CI Width'] = original_estimates['CI'].iloc[:,1] - original_estimates['CI'].iloc[:,0]
#     cis['CI Ratio'] = cis['Synthetic CI Width'] / cis['Original CI Width']

#     return cis
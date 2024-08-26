import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.random import default_rng
from scipy.spatial import KDTree
from scipy.stats import t

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
        scaler = preprocessing.StandardScaler()
        
        # scale synthetic data using means and standard deviations 
        synthetic_scaled = scaler.fit(X=train_data).transform(X=Z)

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

#########################################################################################################
#########################################################################################################
#########################################################################################################

# function to calculate IMS specifically
def ims_calc(train_data, synthetic_data, delta, synthetic_is_train=False):
    
    scaler = preprocessing.StandardScaler()
    
    train_data_scaled = scaler.fit_transform(X=train_data)
    
    training_tree = KDTree(train_data_scaled)
    
    synthetic_data_scaled = scaler.fit(X=train_data).transform(X=synthetic_data)
    
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

def attribute_disclosure_reduction(original_data, synthetic_data, continuous_vars, categorical_vars, sensitive_var, mixture_model, deltas, c, prior_prob):

    # number of original records
    num_records = original_data.shape[0]

    # record percentages
    print_nums = [int(np.ceil(i*num_records)) for i in [0.25, 0.5, 0.75]]

    # random number generator
    rng = default_rng()

    # copy the synthetic dataset
    new_sX = synthetic_data.copy()

    # tree of synthetic records (continuous values only)
    sX_tree = KDTree(new_sX[continuous_vars])

    # store mixture component parameters
    mus = mixture_model.means_
    sigmas = mixture_model.covariances_

    # temporary count of the number of rows that violate one or more conditions
    violator_count = num_records

    # number of anonymization loops required
    num_loops = 1

    # list all categorical variables including sensitive variable
    all_categorical = categorical_vars.copy()
    all_categorical.append(sensitive_var)

    # as long as the last loop had a violator, we need to recheck all records
    while violator_count > 0:

        # reset violator count
        violator_count = 0

        # loop over delta values
        for delta in deltas:

            # split original records in thirds to preserve memory - do computations on each portion
            original_data = original_data.sample(frac=1.0).reset_index(drop=True)
            # orig_1, orig_2, orig_3 = np.array_split(original_data, 3)

            for subset_id, subset in enumerate(np.array_split(original_data, 3)):

                # compute the neighbors within radius delta based on continuous variables only
                neighbor_indices = sX_tree.query_ball_point(subset[continuous_vars], r=delta, p=2.0)

                # store the neighbors for each original record
                neighbor_records = [new_sX.loc[y, all_categorical] for y in neighbor_indices]

                # calculate the proportion of non-white observations amongst the synthetic neighbors
                prop_pos_sensitive = [np.mean(y.loc[y.loc[:, categorical_vars].eq(subset.loc[:, categorical_vars].iloc[x,:]).all(axis=1), sensitive_var]) for x,y in enumerate(neighbor_records)]

                # replace NA values (there were no neighbors) with the prior probability
                prop_pos_sensitive = np.nan_to_num(prop_pos_sensitive, nan=prior_prob)

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
                        original_continuous = original_record[continuous_vars]
                        original_cat = original_record[all_categorical]
                        original_sens = original_record[sensitive_var]

                        # find the component with the highest responsibility for the violating record
                        component_index = np.argmax(mixture_model.predict_proba(pd.DataFrame(original_continuous).T), axis = 1)[0]
                        current_mu = mus[component_index,:]
                        current_sigma = sigmas[component_index,:,:]

                        # numpy array for storing new records
                        valid_candidates = np.zeros((0,len(continuous_vars)))
                        # track number of while loops needed to generate enough new records
                        num_candidate_loops = 0
                        # as long as we have fewer new records than needed
                        while valid_candidates.shape[0] < num_needed[i]:
                            # generate a bunch of candidate points
                            candidate_points = rng.multivariate_normal(current_mu, current_sigma, size=100000)
                            candidate_tree = KDTree(candidate_points)
                            valid_indices = candidate_tree.query_ball_point(original_continuous, delta, p=2.0, return_sorted=True)
                            valid_candidates = np.vstack([valid_candidates, candidate_points[valid_indices,:]])
                            num_candidate_loops += 1
                            if num_candidate_loops > 100:
                                print('Stuck in inference loop.')

                        # select the number of needed candidates and create the new records
                        new_locations = valid_candidates[:num_needed[i],:]
                        new_categorical = np.vstack([np.array(original_cat).reshape(1,-1) for k in range(num_needed[i])])
                        new_records = pd.DataFrame(np.hstack([new_locations, new_categorical]))

                        # edit column names
                        new_records.columns = new_sX.columns

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
    
    full_ad_conds = []
    full_indices = []
    
    # tree for original locations
    orig_tree = KDTree(original_data[continuous_vars])
    
    # tree for synthetic locations
    sX_tree = KDTree(synthetic_data[continuous_vars])
    
    for d in deltas:
        
        # lists to store the inference condition for each original row and the indices of those rows that violate
        ad_conds = []
    
        # find synthetic neighbors of each original point
        location_neighbors = orig_tree.query_ball_tree(sX_tree, r=d, p=2.0)
    
        # for each original record
        for i, row in original_data.iterrows():
        
            # matches on categorical attributes from location neighbors
            categorical_matches = (synthetic_data.loc[location_neighbors[i], categorical_vars] == row[categorical_vars]).all(1)
            
            matching_rows = synthetic_data.loc[location_neighbors[i],:].loc[categorical_matches.values,:]
            
            if matching_rows.shape[0] > 0:
                
                if row[sensitive_var] == 1.0:
                    prior = prior_prob
                else:
                    prior = 1 - prior_prob
            
                cond = np.mean(matching_rows[sensitive_var] == row[sensitive_var])/prior
                
            else:
                
                cond = 1
        
            # store number of matches and their indices
            ad_conds.append(cond)
        
        ad_conds = pd.Series(ad_conds)
        
        full_ad_conds.append(ad_conds)
        
    print("Dataset completed.")
        
    return full_ad_conds

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
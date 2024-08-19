import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.random import default_rng
from scipy.spatial import KDTree
from scipy.stats import t

rng = np.random.RandomState(42)

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

# function to generate polynomial features and standardize a given data set
def polynomial_and_standardize(dataset, poly_degree=3, interaction_only=False):
    
    poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False)
    
    X = poly.fit_transform(dataset)
    
    scaled_X = preprocessing.StandardScaler().fit_transform(X)
    
    return scaled_X

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

    return (pd.DataFrame({"Type" : np.concatenate([["Train"], np.repeat(type_of_synthetic, 20)]), 
                         "IMS" : np.concatenate([[IMS_train], IMS_synthetic]), 
                         "DCR" : np.concatenate([[DCR_train], DCR_synthetic]), 
                         "NNDR" : np.concatenate([[NNDR_train], NNDR_synthetic])}))

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

# function to apply attribute disclosure prevention algorithm
def attribute_disclosure_reduction_sk(original_data, synthetic_data, mixture_model, deltas, c, prior_prob):
    
    # number of original records
    num_records = original_data.shape[0]
    
    # record percentages
    print_nums = [int(np.ceil(i*num_records)) for i in [0.25, 0.5, 0.75]]
    
    # random number generator
    rng = default_rng()
    
    # copy the synthetic dataset
    new_sX = synthetic_data
    
    # tree for synthetic locations
    sX_tree = KDTree(new_sX[["latitude", "longitude"]])
    
    # store mixture component parameters
    mus = mixture_model.means_
    sigmas = mixture_model.covariances_
    
    # temporary count of the number of rows that violate one or more conditions
    violator_count = num_records
    
    # number of anonymization loops required
    num_loops = 1
    
    # while we have any violator rows
    while violator_count > 0:
        
        # reset violator count
        violator_count = 0
        
        # for each original record
        # we shuffle the records each time so that the violating records are fixed in a random order
        for i, original_record in original_data.sample(frac=1.0).reset_index(drop=True).iterrows():
            
            original_location = original_record.loc[["latitude", "longitude"]]
            original_categorical = original_record.loc[["sex", "age", "state"]]
            
            # for each delta
            for delta in deltas:
                    
                ##### Test the Inference Criterion
                
                # find synthetic neighbors based on location
                location_neighbors = sX_tree.query_ball_point(original_location, r=delta, p=2.0)
                
                # matches on categorical attributes from location neighbors
                categorical_matches = (new_sX.loc[location_neighbors,['sex', 'age']] == original_categorical[["sex", "age"]]).all(1)
                
                matching_rows = new_sX.loc[location_neighbors,:].loc[categorical_matches.values,:]
                
                # if there are any records in the location neighborhood that match on sex and age
                
                if matching_rows.shape[0] > 0:
                
                    if original_categorical['state'] == 1.0:
                        prior = prior_prob
                    else:
                        prior = 1-prior_prob
                        
                    num_matching = np.sum(matching_rows['state'] == original_categorical['state'])
            
                    cond = num_matching/matching_rows.shape[0] * 1/prior
                
                    if cond > c:
                        
                        # add one to number of violators
                        violator_count += 1
                        
                        # number of records with non-matching sensitive variable needed to meet inference
                        num_needed = int(np.ceil(num_matching/(prior*c) - matching_rows.shape[0]))
                        
                        # find the component with the highest responsibility for the confidential record
                        component_index = np.argmax(GMM.predict_proba(pd.DataFrame(original_location).T), axis = 1)[0]
                        current_mu = mus[component_index,:]
                        current_sigma = sigmas[component_index,:,:]
            
                        valid_candidates = np.zeros((0,2))
                        num_candidate_loops = 0
                        while valid_candidates.shape[0] < num_needed:
                        
                            # generate a bunch of candidate points
                            candidate_points = rng.multivariate_normal(current_mu, current_sigma, size=100000)
                            candidate_tree = KDTree(candidate_points)
                            valid_indices = candidate_tree.query_ball_point(original_location, delta, p=2.0, return_sorted=True)
                            valid_candidates = np.vstack([valid_candidates, candidate_points[valid_indices,:]])
                            num_candidate_loops += 1
                            if num_candidate_loops > 100:
                                print('Stuck in inference loop.')
                    
                        # select the number of needed candidates
                        new_locations = valid_candidates[:num_needed,:]
                    
                        new_categorical = np.vstack([np.array(original_categorical).reshape(1,-1) for k in range(num_needed)])
                    
                        new_records = pd.DataFrame(np.hstack([new_locations, new_categorical]))
                    
                        new_records.columns = new_sX.columns
                        
                        new_records['state'] = 1.0 - original_categorical['state']
                    
                        new_sX = pd.concat([new_sX, new_records], axis=0).reset_index(drop=True)
                    
                        # rebuild the tree for synthetic locations
                        sX_tree = KDTree(new_sX[["latitude", "longitude"]])
    
            if int(i) in print_nums:
                print("Record " + str(i) + " completed.")
                
        print("Full anonymization loop " + str(num_loops) + " completed.")
        
        num_loops += 1
                    
    return new_sX

# function to assess risk of attribute disclosure for South Korea data
def attribute_disclosure_evaluation_sk(original_data, synthetic_data, prior_prob, deltas):
    
    full_ad_conds = []
    full_indices = []
    
    # tree for original locations
    orig_tree = KDTree(original_data[["latitude", "longitude"]])
    
    # tree for synthetic locations
    sX_tree = KDTree(synthetic_data[["latitude", "longitude"]])
    
    for d in deltas:
        
        # lists to store the inference condition for each original row and the indices of those rows that violate
        ad_conds = []
    
        # find synthetic neighbors of each original point
        location_neighbors = orig_tree.query_ball_tree(sX_tree, r=d, p=2.0)
    
        # for each original record
        for i, row in original_data.iterrows():
        
            # matches on categorical attributes from location neighbors
            categorical_matches = (synthetic_data.loc[location_neighbors[i],['sex', 'age']] == row[['sex', 'age']]).all(1)
            
            matching_rows = synthetic_data.loc[location_neighbors[i],:].loc[categorical_matches.values,:]
            
            if matching_rows.shape[0] > 0:
                
                if row['state'] == 1.0:
                    prior = prior_prob
                else:
                    prior = 1 - prior_prob
            
                cond = np.mean(matching_rows['state'] == row['state'])/prior
                
            else:
                
                cond = 1
        
            # store number of matches and their indices
            ad_conds.append(cond)
        
        ad_conds = pd.Series(ad_conds)
        
        full_ad_conds.append(ad_conds)
        
    print("Dataset completed.")
        
    return full_ad_conds

# function to combine point estimates and confidence intervals across synthetic data sets and 
# compare them to those from the original data
def combined_estimates(synthetic_estimates, original_estimates, type, num_synthetic_datasets, n, nsynth):
    
    # number of synthetic datasets
    m = num_synthetic_datasets

    # the combined point estimate (q-bar_m)
    qms = []
    for i in ['const', 'latitude', 'longitude', 'sex', 'age']:
        qms.append(np.mean([j['params'][i] for j in synthetic_estimates]))

    # the values for b_m
    bms = []
    for i, j in enumerate(['const', 'latitude', 'longitude', 'sex', 'age']):
        bms.append(np.sum([(1/(m-1))*(k['params'][j]-qms[i])**2 for k in synthetic_estimates]))

    # the values for u-bar_m
    ums = []
    for i, j in enumerate(['const', 'latitude', 'longitude', 'sex', 'age']):
        ums.append(np.mean([j['l_var'][i] for j in synthetic_estimates]))

    # calculate variance of qms estimates
    Tf = (1 + 1/m) * np.array(bms) - np.array(ums)
    delta = 1*(Tf<0)
    Tfstar = (Tf>0) * Tf + delta*((nsynth/n) * np.array(ums))

    # calculate degrees of freedom
    vf = (m - 1) * (1 - np.array(ums)/((1 + 1/m) * np.array(bms)))**2

    # upper and lower CI bounds
    upper = pd.Series(np.array(qms) + t.ppf(q=0.975, df=vf)*np.sqrt(Tfstar))
    lower = pd.Series(np.array(qms) - t.ppf(q=0.975, df=vf)*np.sqrt(Tfstar))

    cis = pd.concat([pd.Series(qms), lower, upper], axis=1)

    cis.columns = ['Point Estimate', 'Lower', 'Upper']

    cis['Type'] = type

    # calculate the L1 distance for the point estimates
    cis['L1 Point Estimate'] = np.abs(cis['Point Estimate'] - original_estimates['params'].reset_index(drop=True))

    # calculate the confidence interval ratio (synthetic width / original width)
    cis['Synthetic CI Width'] = cis['Upper'] - cis['Lower']
    cis['Original CI Width'] = original_estimates['CI'].iloc[:,1] - original_estimates['CI'].iloc[:,0]
    cis['CI Ratio'] = cis['Synthetic CI Width'] / cis['Original CI Width']

    return cis
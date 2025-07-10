This example uses the Criteo Uplift dataset, described as follows:

This dataset is constructed by assembling data resulting from several incrementality tests, a particular randomized trial procedure where a random part of the population is prevented from being targeted by advertising. it consists of 14M rows, each one representing a user with 11 features, a treatment indicator and 2 labels (visits and conversions).

Here is a detailed description of the fields (they are comma-separated in the file):
f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11: feature values (dense, float)
treatment: treatment group (1 = treated, 0 = control)
conversion: whether a conversion occured for this user (binary, label)
visit: whether a visit occured for this user (binary, label)
exposure: treatment effect, whether the user has been effectively exposed (binary)

The dataset is constructed and cleaned using the Criteo Data Cleaning python script. Following this, synthetic data can be created using the GMM and MNL Synthesis Criteo python script. The performance of models with the simulated and synthetic data can be performed using the Criteo Uplift Model Comparison R script.

# Load required libraries
library(tidyverse)
library(magrittr)
library(KernSmooth)
library(ks)

# Define function for cross-validation to find optimal bandwidth
find_optimal_bandwidth <- function(data) {
  cv_result <- numeric()
  for (bw in seq(0.2, 1, by = 0.1)) {
    cv_result <- c(cv_result, sum(bkde(data, bandwidth = bw)$y))
  }
  optimal_bandwidth <- (which.max(cv_result) - 1) * 0.1 + 0.1
  return(optimal_bandwidth)
}

# Define privacy attack
privacy_attack <- function(seed, simulations, train, adversary, outside_training, protected_training, protected_adversary) {
  set.seed(seed)
  epsilons <- c() # To store results
  
  for (iter in 1:simulations) {
    set.seed(iter) # Again for reproducibility
    cat("iteration is ", iter, "\n")
    
    # To prevent a naive model
    N <- nrow(train) / 10
    
    # Step 1, 2, and 3 from paper
    bandwidths <- 10^seq(-1, 1, length.out = 20) # Vary the bandwidth
    
    density_train <- numeric(nrow(train)) # Initialize vector to store densities for train data
    density_adversary <- numeric(nrow(train)) # Initialize vector to store densities for adversary data
    
    # Loop over each column of the dataset
    for (i in 1:ncol(protected_training)) {
      # Perform cross-validation to find optimal bandwidth
      optimal_bandwidth <- find_optimal_bandwidth(protected_training[, i])
      
      # Estimate pdf from train data using optimal bandwidth
      kde_train <- bkde(protected_training[, i], bandwidth = optimal_bandwidth)
      kde_adversary <- bkde(protected_adversary[, i], bandwidth = optimal_bandwidth)
      
      density_train <- density_train + kde_train$y # Accumulate densities for train data
      density_adversary <- density_adversary + kde_adversary$y # Accumulate densities for adversary data
    }
    
    # Calculate average densities
    density_train <- density_train / ncol(protected_training)
    density_adversary <- density_adversary / ncol(protected_adversary)
    
    # Calculate TPR
    TPR <- sum(density_train > density_adversary) / length(density_train)
    
    # Step 5
    density_train_new <- numeric(nrow(outside_training)) # Initialize vector to store densities for outside training data
    density_adversary_new <- numeric(nrow(outside_training)) # Initialize vector to store densities for outside adversary data
    
    # Loop over each column of the dataset
    for (i in 1:ncol(protected_training)) {
      # Estimate pdf from outside training data using optimal bandwidth
      kde_train_new <- bkde(outside_training[, i], bandwidth = optimal_bandwidth, gridsize = 1000)
      kde_adversary_new <- bkde(outside_training[, i], bandwidth = optimal_bandwidth, gridsize = 1000)
      
      density_train_new <- density_train_new + kde_train_new$y # Accumulate densities for outside training data
      density_adversary_new <- density_adversary_new + kde_adversary_new$y # Accumulate densities for outside adversary data
    }
    
    # Calculate average densities
    density_train_new <- density_train_new / ncol(outside_training)
    density_adversary_new <- density_adversary_new / ncol(outside_training)
    
    # Calculate FPR
    FPR <- sum(density_train_new > density_adversary_new) / length(density_train_new)
    
    TNR <- 1 - FPR
    FNR <- 1 - TPR
    epsilons <- c(epsilons, max(log((1 - (1/N) - FPR) / FNR), log((1 - (1/N) - FNR) / FPR))) # Append resulting epsilon to epsilons
    cat("FPR is ", FPR, "\n")
    cat("FNR is ", FNR, "\n")
    cat("TPR is ", TPR, "\n")
    cat("TNR is ", TNR, "\n")
    cat("empirical epsilon = ", max(log((1 - (1/N) - FPR) / FNR), log((1 - (1/N) - FNR) / FPR)), "\n")
  }
  return(list(epsilons = epsilons, FPR = FPR, TNR = TNR, FNR = FNR, TPR = TPR))
}

# here we import the external data, train data and adversary training data (unprotected)
evaluation_outside_training <- read.csv("external_data.csv")
train <- read.csv("train_data.csv")
adversary_training <- read.csv("adv_data.csv")

# Apply privacy attack
privacy_attack(seed = 1, 
               simulations = 10, 
               train = train, 
               adversary = adversary_training, 
               outside_training = evaluation_outside_training, 
               protected_training = train, 
               protected_adversary = adversary_training)

##################################################
##################################################

# apply privacy attack to synthpop data sets
# Initialize an empty vector to store results
results_synthpop <- c()

for (i in 0:2) {  # R uses 1-based indexing, but since the files are 0-based, we use 0:2
    # Read the synthetic data files
    synth_train <- read.csv(paste0("synthetic_train/synthpop_train_data_", i, ".csv"))
    synth_adversary_training <- read.csv(paste0("synthetic_adv/synthpop_adv_data_", i, ".csv"))
    
    # Assuming attacks is an R object with a privacy_attack function
    result <- privacy_attack(
        seed = 1,
        simulations = 10,
        train = train,
        adversary = adversary_training,
        outside_training = evaluation_outside_training,
        protected_training = synth_train,
        protected_adversary = synth_adversary_training
    )
    
    # Append the mean of the first element of result to results_synthpop
    results_synthpop <- c(results_synthpop, result[[1]])
}

# apply privacy attack to synthpop data sets
# Initialize an empty vector to store results
results_bayes <- c()

for (i in 0:2) {  # R uses 1-based indexing, but since the files are 0-based, we use 0:2
    # Read the synthetic data files
    synth_train <- read.csv(paste0("synthetic_train/sd_", i, ".csv"))
    synth_adversary_training <- read.csv(paste0("synthetic_adv/sd_", i, ".csv"))
    
    # Assuming attacks is an R object with a privacy_attack function
    result <- privacy_attack(
        seed = 1,
        simulations = 10,
        train = train,
        adversary = adversary_training,
        outside_training = evaluation_outside_training,
        protected_training = synth_train,
        protected_adversary = synth_adversary_training
    )
    
    # Append the mean of the first element of result to results_synthpop
    results_bayes <- c(results_bayes, result[[1]])
}
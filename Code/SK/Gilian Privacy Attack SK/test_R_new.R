# Load required libraries
library(tidyverse)
library(magrittr)
#install.packages("ks")
#install.packages("KernSmooth")
library(KernSmooth)
library(ks)

# Read public churn data
url <- 'https://raw.githubusercontent.com/albayraktaroglu/Datasets/master/churn.csv'
churn <- read.csv(url)

# Select some variables
churn <- churn[, 7:15]
samples <- 300 # Select the number of observations
churn <- churn[!duplicated(churn), ] # Drop duplicates

# Create the train, adversary, and outside_training set
set.seed(42)
train_indices <- sample(1:nrow(churn), size = samples*2/3)
train <- churn[train_indices, ]
adversary_training <- churn[-train_indices, ]

# Define the data protected method: Swapping 25% of the observations
swapping <- function(percent, data) {
  set.seed(42)
  idx <- sample(1:ncol(data), 1) # Pick a random variable
  variable <- data[, idx] # Select variable from data
  ix_size <- percent * length(variable) * 0.5 # Select proportion to shuffle
  ix_1 <- sample(seq_along(variable), size = ix_size, replace = FALSE) # Select rows to shuffle
  ix_2 <- sample(seq_along(variable), size = ix_size, replace = FALSE) # Select rows to shuffle
  b1 <- variable[ix_1] # Take rows from variable and create b
  b2 <- variable[ix_2] # Take rows from variable and create b
  variable[ix_2] <- b1 # Swap 1
  variable[ix_1] <- b2 # Swap 2
  data[, idx] <- variable  # Place variable back in original data
  return(data)
}

# Apply protection to train and adversary
swap25_train <- swapping(percent = 0.25, data = train) # Apply swapping 25% to train
swap25_adversary_training <- swapping(percent = 0.25, data = adversary_training)  # Apply swapping 25% to adv

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
privacy_attack <- function(seed,
                           simulations,
                           train,
                           adversary,
                           outside_training,
                           protected_training,
                           protected_adversary) {

  set.seed(seed)
  epsilons <- c() # To store results

  for (iter in 1:simulations) {
    set.seed(iter) # Again for reproducibility
    cat("iteration is ", iter, "\n")

    # To prevent a naive model
    N <- nrow(train) / 10

    # Step 1, 2, and 3 from paper
    # bandwidths <- 10^seq(-1, 1, length.out = 20) # Vary the bandwidth
    
    density_train <- numeric(nrow(train)) # Initialize vector to store densities for train data
    density_adversary <- numeric(nrow(train)) # Initialize vector to store densities for adversary data
    
    # Loop over each column of the dataset
    for (i in 1:ncol(protected_training)) {
      # Perform cross-validation to find optimal bandwidth
      optimal_bandwidth <- find_optimal_bandwidth(protected_training[, i])
      
      # Estimate pdf from train data using optimal bandwidth
      kde_train <- bkde(protected_training[, i], bandwidth = optimal_bandwidth)
      kde_adversary <- bkde(protected_adversary[, i], bandwidth = optimal_bandwidth)

      # Evaluate train data points using the densities
      univ_density_train <- approxfun(kde_train$x, kde_train$y, rule = 2)(protected_training[, i])
      univ_density_adversary <- approxfun(kde_adversary$x, kde_adversary$y, rule = 2)(protected_training[, i])
      
      # Accumulate densities
      density_train <- density_train + univ_density_train # Accumulate densities for train data
      density_adversary <- density_adversary + univ_density_adversary # Accumulate densities for adversary data
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

      # Perform cross-validation to find optimal bandwidth
      optimal_bandwidth <- find_optimal_bandwidth(protected_training[, i])

      # Estimate pdf from outside training data using optimal bandwidth
      kde_train_new <- bkde(protected_training[, i], bandwidth = optimal_bandwidth, gridsize = 1000)
      kde_adversary_new <- bkde(protected_adversary[, i], bandwidth = optimal_bandwidth, gridsize = 1000)

      # Use approxfun to evaluate the density of outside training points
      univ_density_train_new <- approxfun(kde_train_new$x, kde_train_new$y, rule = 2)(outside_training[, i])
      univ_density_adversary_new <- approxfun(kde_adversary_new$x, kde_adversary_new$y, rule = 2)(outside_training[, i])
      
      density_train_new <- density_train_new + univ_density_train_new # Accumulate densities for outside training data
      density_adversary_new <- density_adversary_new + univ_density_adversary_new # Accumulate densities for outside adversary data
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

# Apply privacy attack
privacy_attack(seed = 1, simulations = 10, train = train, adversary = adversary_training, 
               outside_training = adversary_training, protected_training = swap25_train, 
               protected_adversary = swap25_adversary_training)
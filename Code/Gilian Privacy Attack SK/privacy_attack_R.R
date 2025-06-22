# Load required libraries
library(tidyverse)
library(magrittr)
library(KernSmooth)
library(ks)

# Function to compute multivariate KDE
compute_mvkde <- function(data, method = "hpi") {
  # Remove any rows with NA or infinite values
  if (!is.matrix(data)) data <- as.matrix(data)
  valid_rows <- apply(is.finite(data), 1, all)
  
  if (!all(valid_rows)) {
    warning(sum(!valid_rows), " rows with non-finite values removed from data")
    data <- data[valid_rows, , drop = FALSE]
  }

  tryCatch({
    # Compute bandwidth matrix
    if (method == "hpi") {
      # Note: this function is valid for 1 to 6 dimensional data
      H <- Hpi(data, binned = FALSE, bgridsize = min(20, nrow(data)))
    } else if (method == "lscv") {
      # Note: this function is valid for 1 to 6 dimensional data
      H <- Hscv(data, binned = FALSE, bgridsize = min(20, nrow(data)))
    } else {
      stop("Invalid method. Use 'hpi' or 'lscv'")
    }
    
    # Compute KDE
    de <- kde(x = data, H = H, binned = FALSE, eval.points = data, density=TRUE)
    
    # Return a function that evaluates the KDE at new points
    function(newdata) {
      if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
      destimates <- predict(de, x = newdata)
      # replace NA values in destimates with 0
      destimates[is.na(destimates)] <- 0
      return(destimates)
    }
  }, error = function(e) {
    warning("Error in multivariate KDE: ", e$message)
  })
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
  bw_estimation_method <- "hpi"

  # Pre-compute constants
  N <- nrow(train) / 10  # To prevent a naive model
  one_minus_1_over_N <- 1 - (1/N)
  
  for (iter in 1:simulations) {
    set.seed(iter)  # For reproducibility
    cat("\n--- Iteration", iter, "---\n")
    
    # Step 1-3: Train KDE models and compute densities
    kde_train_fn <- compute_mvkde(protected_training, method = bw_estimation_method)
    kde_adv_fn <- compute_mvkde(protected_adversary, method = bw_estimation_method)
    
    # Compute densities for training data
    density_train <- kde_train_fn(train)
    density_adversary <- kde_adv_fn(train)
    
    # Calculate True Positive Rate (TPR)
    TPR <- mean(density_train > density_adversary)
    
    # Step 5: Evaluate on outside training data
    density_train_new <- kde_train_fn(outside_training)
    density_adversary_new <- kde_adv_fn(outside_training)
    
    # Calculate False Positive Rate (FPR)
    FPR <- mean(density_train_new > density_adversary_new)
    
    # Compute derived metrics
    TNR <- 1 - FPR
    FNR <- 1 - TPR
    
    # Compute epsilon (differential privacy parameter)
    epsilon <- max(
      log((one_minus_1_over_N - FPR) / FNR),
      log((one_minus_1_over_N - FNR) / FPR)
    )
    epsilons <- c(epsilons, epsilon)
    
    # Print metrics
    metrics <- data.frame(
      Metric = c("FPR", "FNR", "TPR", "TNR", "epsilon"),
      Value = c(FPR, FNR, TPR, TNR, epsilon)
    )
    print(metrics, row.names = FALSE)
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

for (i in 0:4) {  # R uses 1-based indexing, but since the files are 0-based, we use 0:4
    # Read the synthetic data files
    synth_train <- read.csv(paste0("synthetic_train/synthpop_", i, ".csv"))
    synth_adversary_training <- read.csv(paste0("synthetic_adv/synthpop_", i, ".csv"))
    
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
  
for (i in 0:4) {  # R uses 1-based indexing, but since the files are 0-based, we use 0:4
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

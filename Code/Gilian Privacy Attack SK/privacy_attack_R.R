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
  
  if (nrow(data) < 2) {
    stop("Insufficient valid data points for KDE estimation")
  }
  
  tryCatch({
    # Compute bandwidth matrix
    if (method == "hpi") {
      H <- Hpi(data, binned = FALSE, bgridsize = min(100, nrow(data)))
    } else if (method == "lscv") {
      H <- Hscv(data, binned = FALSE, bgridsize = min(50, nrow(data)))
    } else {
      stop("Invalid method. Use 'hpi' or 'lscv'")
    }
    
    # Compute KDE
    kde <- kde(x = data, H = H, eval.points = data)
    
    # Return a function that evaluates the KDE at new points
    function(newdata) {
      if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
      predict(kde, x = newdata)
    }
  }, error = function(e) {
    warning("Error in multivariate KDE: ", e$message)
    # Fallback to product of univariate KDEs
    warning("Falling back to product of univariate KDEs")
    
    # Compute univariate KDEs for each dimension
    univ_kdes <- lapply(1:ncol(data), function(i) {
      kde_1d <- bkde(data[, i], bandwidth = dpik(data[, i]))
      approxfun(kde_1d$x, kde_1d$y, yleft = 0, yright = 0)
    })
    
    # Return product of marginals
    function(newdata) {
      if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
      densities <- matrix(1, nrow = nrow(newdata))
      for (i in 1:ncol(newdata)) {
        densities <- densities * pmax(0, univ_kdes[[i]](newdata[, i]))
      }
      as.numeric(densities)
    }
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

  for (iter in 1:simulations) {
    set.seed(iter) # Again for reproducibility
    cat("iteration is ", iter, "\n")

    # To prevent a naive model
    N <- nrow(train) / 10

    # Step 1, 2, and 3 from paper
    
    density_train <- numeric(nrow(train)) # Initialize vector to store densities for train data
    density_adversary <- numeric(nrow(train)) # Initialize vector to store densities for adversary data
    
    # Create multivariate KDE functions
    kde_train_fn <- compute_mvkde(protected_training, method = bw_estimation_method)
    kde_adv_fn <- compute_mvkde(protected_adversary, method = bw_estimation_method)
    
    # Calculate densities using multivariate KDE
    density_train <- kde_train_fn(train)
    density_adversary <- kde_adv_fn(train)
    
    # No need to average for multivariate KDE as it already handles the full space
    
    # Calculate TPR
    TPR <- sum(density_train > density_adversary) / length(density_train)
    
    # Step 5
    density_train_new <- numeric(nrow(outside_training)) # Initialize vector to store densities for outside training data
    density_adversary_new <- numeric(nrow(outside_training)) # Initialize vector to store densities for outside adversary data
    
    # Evaluate outside training data using multivariate KDE
    density_train_new <- kde_train_fn(outside_training)
    density_adversary_new <- kde_adv_fn(outside_training)
    
    # No need to average for multivariate KDE as it already handles the full space
    
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

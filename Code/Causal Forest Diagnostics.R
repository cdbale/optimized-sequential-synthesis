library(tidyverse)
library(grf)
library(cli)

# Runs a series of checks on causal forests run on syntheric and original data
# Needs 3 dataframes containing outcome, covariates, and explanatory variables
# Dataframes must have named columns, and outcome_position asks for a numeric
# value indicated which column is the outcome.
# Function will iterate through causal forests with each different variable as
# explanatory and the others as covariates
cf_comparison_checks <- function(
    train_data,
    synthesized_data,
    test_data,
    outcome_position
) {
  # List to be returned
  outcome <- list(
    means = NULL,
    vars = NULL,
    corrs = NULL,
    var_imp = NULL,
    prediction_values = NULL
  )
  # num of columns
  ncols <- ncol(train_data)
  # quick function to return pure vector
  ulun <- function(vector) {
    vector <- vector |> unlist() |> unname()
    return(vector)
  }
  # Progress bar
  cli_progress_bar("Running Causal Forests", total = ncols - 1)
  # Run the forests, get comparison checks
  outcome <- list(
    means = NULL,
    vars = NULL,
    corrs = NULL,
    var_imp = NULL,
    prediction_values = NULL
  )
  ncols <- ncol(train_data)
  ulun <- function(vector) {
    vector <- vector |> unlist() |> unname()
    return(vector)
  }
  cli_progress_bar("Running Causal Forests", total = ncols - 1)
  for (i in 1:ncols) {
    explanatory_cols <- setdiff(1:ncols, c(outcome_position, i))
    if (outcome_position != i) {
      # Run Forests -------------------------------------------------------------
      # Run the causal forest on the original data
      original_forest <- causal_forest(train_data[,explanatory_cols],
                                       ulun(train_data[,outcome_position]),
                                       ulun(train_data[,i]))
      predictions <- predict(original_forest, test_data[,explanatory_cols])
      pred_values <- predictions$predictions
      # Run the causal forest on the synthetic data
      synthetic_forest <- causal_forest(synthesized_data[,explanatory_cols],
                                        ulun(synthesized_data[,outcome_position]),
                                        ulun(synthesized_data[,i]))
      predictions_synth <- predict(synthetic_forest, test_data[,explanatory_cols])
      pred_values_synth <- predictions_synth$predictions
      # Get Diagnostics ---------------------------------------------------------
      # Overall Mean and Variance, Correlation
      ## Means
      means <- bind_cols(
        orig_mean = pred_values |> mean(),
        synthetic_mean = pred_values_synth |> mean(),
        treatment_variable = colnames(train_data)[i]
      )
      ## Variance
      vars <- bind_cols(
        orig_mean = pred_values |> var(),
        synthetic_mean = pred_values_synth |> var(),
        treatment_variable = colnames(train_data)[i]
      )
      ## Correlation
      corrs <- bind_cols(
        correlation = cor(pred_values, pred_values_synth),
        treatment_variable = colnames(train_data)[i]
      )
      # Variable Importance
      var_imp <- tibble(variable = colnames(train_data)[explanatory_cols],
                        orig_variable_imp = as.vector(variable_importance(original_forest)),
                        synth_variable_imp = as.vector(variable_importance(synthetic_forest)),
                        diff = orig_variable_imp - synth_variable_imp,
                        treatment_variable = colnames(train_data)[i],
      )
      # CATE Distributions - Density Curves
      prediction_values <- tibble(original_pred_vals = pred_values,
                                  synthetic_pred_vals = pred_values_synth,
                                  treatment_variable = colnames(train_data)[i])
      # Consolidate Values ------------------------------------------------------
      outcome$means <- bind_rows(outcome$means, means)
      outcome$vars <- bind_rows(outcome$vars, vars)
      outcome$corrs <- bind_rows(outcome$corrs, corrs)
      outcome$var_imp <- bind_rows(outcome$var_imp, var_imp)
      outcome$prediction_values <- bind_rows(outcome$prediction_values, prediction_values)
      # update progress
      cli_progress_update()
    }
  }
  # finish progress
  cli_progress_done()
  return(outcome)
}

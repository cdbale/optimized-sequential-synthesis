library(tidyverse)
library(tidymodels)

set.seed(100)

# Churn Data
simulated_data <- read_csv(here::here("Data/IBM", "IBM_Telco_Cleaned.csv"))
synthesized_data <- read_csv(here::here("Data/IBM", "IBM_Telco_Synthesized.csv"))


# Prep Everything - Simulated
split <- initial_split(simulated_data, prop = .9)

training <- training(split)
testing <- testing(split)

recipe_partial <- training |>
  recipe(Churn ~ gender + SeniorCitizen + Partner + Dependents +
           tenure + PhoneService + MultipleLines + OnlineSecurity +
           OnlineBackup + DeviceProtection + TechSupport + PaperlessBilling +
           MontlyCharges + TotalCharges) |>
  step_log(c("tenure", "MonthlyCharges", "TotalCharges"), offset = 1)

# Has the private variables of Billing Type, Internet Service, Movie/TV,
# and Payment Method
recipe_full <- training |>
  recipe(Churn ~ gender + SeniorCitizen + Partner + Dependents +
           tenure + PhoneService + MultipleLines + OnlineSecurity +
           OnlineBackup + DeviceProtection + TechSupport + StreamingTV +
           StreamingMovies + PaperlessBilling + MontlyCharges +
           TotalCharges + `InternetService_Fiber optic` + InternetService_No +
           `Contract_One year` + `Contract_Two year` +
           `PaymentMethod_Credit card (automatic)` +
           `PaymentMethod_Electronic check` +
           `PaymentMethod_Mailed check`) |>
  step_log(c("tenure", "MonthlyCharges", "TotalCharges"), offset = 1)

# v-fold
training_cv <- vfold_cv(training, v = 10, strata = churn)

# Prep Everything - Synthesized
split_synth <- initial_split(synthesized_data, prop = .9)

training_synth <- training(split_synth)
testing_synth <- testing(split_synth)

# v-fold
training_cv_synth <- vfold_cv(training_synth, v = 10, strata = churn)

# Partial Fit: Company has data for Age, Region, and Amount Spent ---------
workflow_partial <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe_partial)

cv_results_partial <- workflow_partial |>
  fit_resamples(
    resamples = training_cv
  )

# Compute model accuracy - Threshold at .5
collect_metrics(cv_results_partial)

# Synthesized Fit: Company has Synthesized Data ---------------------------
workflow_synth <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe_full)

cv_results_synth <- workflow_synth |>
  fit_resamples(
    resamples = training_cv_synth
  )

# Compute model accuracy - Threshold at .5
collect_metrics(cv_results_synth)

# Complete Fit: Company has data for all ----------------------------------
workflow_full <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe_full)

cv_results_full <- workflow_full |>
  fit_resamples(
    resamples = training_cv
  )

# Compute model accuracy - Threshold at .5
collect_metrics(cv_results_full)


# Save Final Results ------------------------------------------------------

final_results_IBM <- bind_rows(
  collect_metrics(cv_results_partial)[1,] |>
    select(-.config) |>
    mutate(model = "partial"),
  collect_metrics(cv_results_synth)[1,] |>
    select(-.config) |>
    mutate(model = "synthetic"),
  collect_metrics(cv_results_full)[1,] |>
    select(-.config) |>
    mutate(model = "full")
) |>
  rename(
    metric = .metric,
    estimator = .estimator
  ) |>
  mutate(data = "IBM Model Comparison")


library(tidyverse)
library(tidymodels)

set.seed(100)

# Churn Data
source("Simulate Data - Churn Independence + Complete Dependence.R")
synthesized_data <- read_csv("independent_aa_churn.csv")

# Prep
synthesized_data <- synthesized_data |>
  mutate(churn = as.factor(churn),
         hiking_int = as.factor(hiking_int),
         sustain_int = as.factor(sustain_int),
         online_int = as.factor(online_int))

# Clean
# synthesized_data <- synthesized_data |>
#   mutate(amount_spent = ifelse(amount_spent < 43.7, 43.7, amount_spent)) |>
#   mutate(num_visits = ifelse(num_visits < 1, 1, num_visits))

simulated_data <- simulated_data |>
  mutate(churn = as.factor(churn),
         hiking_int = as.factor(hiking_int),
         sustain_int = as.factor(sustain_int),
         online_int = as.factor(online_int))

# Prep Everything - Simulated
split <- initial_split(simulated_data, prop = .9)

training <- training(split)
testing <- testing(split)

recipe_partial <- training |>
  recipe(churn ~ amount_spent + age + num_visits) |>
  step_log(c("amount_spent", "age", "num_visits"), offset = 1)

recipe_full <- training |>
  recipe(churn ~ amount_spent + age + num_visits +
           hiking_int + sustain_int + online_int) |>
  step_log(c("amount_spent", "age", "num_visits"), offset = 1)

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

final_results_ind <- bind_rows(
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
  mutate(data = "churn fully dependent, aa independent")

# rm(list = setdiff(ls(), "final_results_ind"))

library(tidyverse)
library(tidymodels)

set.seed(100)

# CAM Data
source("Read in Criteo Data.R")
synthesized_data <- read_csv("synthesized_criteo_data.csv")

# Prep Everything - Simulated
split <- initial_split(cam_data, prop = .9)

training <- training(split)
testing <- testing(split)

recipe_partial <- training |>
  recipe(conversions ~ prev_purchases + age + freq_purchases) |>
  step_log("age", offset = 1)

recipe_full <- training |>
  recipe(conversions ~ prev_purchases + age + freq_purchases +
           app_ads + yt_ads + gs_ads) |>
  step_log("age", offset = 1)

# v-fold
training_cv <- vfold_cv(training, v = 10, strata = conversions)

# Prep Everything - Synthesized
split_synth <- initial_split(synthesized_data, prop = .9)

training_synth <- training(split_synth)
testing_synth <- testing(split_synth)

recipe_full_synth <- training_synth |>
  recipe(conversions ~ prev_purchases + age + freq_purchases +
           app_ads + yt_ads + gs_ads) |>
  step_log("age", offset = 1)

# v-fold
training_cv_synth <- vfold_cv(training_synth, v = 10, strata = conversions)

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
  add_recipe(recipe_full_synth)

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

library(tidyverse)
library(tidymodels)

set.seed(100)

# Churn Data
source("Code/Examples/Simulated Examples/Simulate Data - Churn Partial Dependence + Affinity Attribute Independence.R")
synthesized_data <- read_csv("Data/Simulated/churnPART_aaIND_synthesized.csv")

# Prep
synthesized_data <- synthesized_data |>
  mutate(churn = ifelse(churn >= .5, 1, 0)) |>
  mutate(churn = as.factor(churn),
         hiking_int = as.factor(hiking_int),
         sustain_int = as.factor(sustain_int),
         online_int = as.factor(online_int))

simulated_data <- simulated_data |>
  mutate(churn = as.factor(churn),
         hiking_int = as.factor(hiking_int),
         sustain_int = as.factor(sustain_int),
         online_int = as.factor(online_int))

# Clean
synthesized_data <- synthesized_data |>
  mutate(amount_spent = ifelse(amount_spent < 43.7, 43.7, amount_spent)) |>
  mutate(num_visits = ifelse(num_visits < 1, 1, num_visits))

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
testing_synth <- testing

# v-fold
training_cv_synth <- vfold_cv(training_synth, v = 10, strata = churn)

# Partial Fit: Company has data for Age, Region, and Amount Spent ---------
workflow_partial <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe_partial)

# Fit the model, get predictive accuracy
log_fit_partial <- fit(workflow_partial, data = training)
preds_partial <- predict(log_fit_partial, new_data = testing, type = "class") |>
  bind_cols(testing)
metrics_partial <- preds_partial |>
  metrics(truth = churn, estimate = .pred_class)
accuracy_partial <- metrics_partial |>
  filter(.metric == "accuracy")

# Synthesized Fit: Company has Synthesized Data ---------------------------
workflow_synth <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe_full)

# Fit the model, get predictive accuracy
log_fit_synth <- fit(workflow_synth, data = training_synth)
preds_synth <- predict(log_fit_synth, new_data = testing_synth, type = "class") |>
  bind_cols(testing_synth)
metrics_synth <- preds_synth |>
  metrics(truth = churn, estimate = .pred_class)
accuracy_synth <- metrics_synth |>
  filter(.metric == "accuracy")

# Complete Fit: Company has data for all ----------------------------------
workflow_full <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe_full)

# Fit the model, get predictive accuracy
log_fit_full <- fit(workflow_full, data = training)
preds_full <- predict(log_fit_full, new_data = testing, type = "class") |>
  bind_cols(testing)
metrics_full <- preds_full |>
  metrics(truth = churn, estimate = .pred_class)
accuracy_full <- metrics_full |>
  filter(.metric == "accuracy")

# Final Results
final_results_churnPART_aaIND <- bind_rows(
  accuracy_partial |>
    mutate(model = "partial"),
  accuracy_synth |>
    mutate(model = "synthetic"),
  accuracy_full |>
    mutate(model = "full")
) |>
  rename(
    metric = .metric,
    estimator = .estimator
  ) |>
  mutate(data = "churn partial dependence, aa independence")

# rm(list = setdiff(ls(), "final_results_ind"))

library(tidyverse)
library(tidymodels)

set.seed(100)

# source("Simulate Data - 6 Variables + Churn.R")

original_data <- read_csv("../../Data/Simulations/Churn/churn_simulated.csv")
synthesized_data <- read_csv("../../Data/Simulations/Churn/mnl_0.csv")

# Prep
original_data <- original_data |>
  select(-id) |>
  mutate(churn = as.factor(churn),
         hiking_int = as.factor(hiking_int),
         sustain_int = as.factor(sustain_int),
         online_int = as.factor(online_int))

# Prep
synthesized_data <- synthesized_data |>
  mutate(churn = as.factor(churn),
         hiking_int = as.factor(hiking_int),
         sustain_int = as.factor(sustain_int),
         online_int = as.factor(online_int))

# summary statistics
summary(synthesized_data)

# Prep Everything - Simulated
split <- initial_split(original_data, prop = .9)

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

# examine model coefficients ----------------------------------------------

full_model_results <- logistic_reg() |>
  fit(churn ~ ., data = original_data) |>
  tidy(conf.int = TRUE) |>
  mutate(Type = "Original")

full_synth_model_results <- logistic_reg() |>
  fit(churn ~ ., data = synthesized_data) |>
  tidy(conf.int = TRUE) |>
  mutate(Type = "Synthetic")

# Compare parameter estimates.
full_model_results |> 
  bind_rows(full_synth_model_results) |>
  ggplot(aes(y = term, color = Type)) + 
  geom_point(aes(x = estimate)) + 
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = .1) +
  geom_vline(xintercept = 0, color = "red") +
  labs(x = "Parameter Estimate",
       y = "Variable",
       color = "Data Type",
       title = "Coefficient Comparison - Original vs. Synthetic Data")

# implement differentially private logistic regression --------------------

library(DPpack)

?LogisticRegressionDP

# define regularization function and constant
reg_func <- function(coeff) coeff%*%coeff/2
reg_func_grad <- function(coeff) coeff

# define upper and lower bounds for X variables
upper_bounds <- c(max(original_data$amount_spent),
                  max(original_data$num_visits),
                  max(original_data$age),
                  1,
                  1,
                  1)
lower_bounds <- c(0, 0, 0, 0, 0, 0)

dp_X <- original_data[, 2:ncol(original_data)]
dp_Y <- original_data[, 1]

lrdp <- LogisticRegressionDP$new(regularizer = reg_func,
                                 regularizer.gr = reg_func_grad,
                                 gamma = 0,
                                 eps = 5)

lrdp$fit(dp_X, dp_Y, upper_bounds, lower_bounds)

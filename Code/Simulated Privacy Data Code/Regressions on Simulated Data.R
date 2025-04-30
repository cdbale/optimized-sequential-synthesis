library(tidyverse)
library(tidymodels)

source("Simulate Data - 6 Variables + Churn.R")

# Prep Everything
split <- initial_split(simulated_data, prop = .9)

training <- training(split)
testing <- testing(split)

recipe <- training |>
  recipe(churn ~ amount_spent + age + region +
           hiking_int + sustain_int + online_int) |>
  step_log(c("amount_spent", "age"), offset = 1) |>
  step_dummy("region") |>
  prep()

baked_training <- recipe |>
  bake(training)

baked_testing <- recipe |>
  bake(testing)


# Partial Fit: Company has data for Age, Region, and Amount Spent ---------
partial_fit <- logistic_reg() |>
  set_engine(engine = "glm") |>
  fit(churn ~ age + amount_spent + region,
      data = simulated_data)

# Complete Fit: Company has data for all ----------------------------------
complete_fit <- logistic_reg() |>
  set_engine(engine = "glm") |>
  fit(churn ~ age + amount_spent + region +
        hiking_int + sustain_int + online_int,
      data = simulated_data)

# Amount spent and age into mixture model, round age
# Prob don't need mixture model, focus first on MNL and get it running

# Set up 3 cart/logit models to predict each variable based on the other four/5/6, sequentially

simulated_data |>
  summarize(
    hiking = mean(as.numeric(hiking_int) - 1),
    sustain = mean(as.numeric(sustain_int) - 1),
    online = mean(as.numeric(online_int) - 1)
  )

simulated_data |>
  select(hiking_int) |>
  transmute(hiking_int = as.numeric(hiking_int)) |>
  unique()




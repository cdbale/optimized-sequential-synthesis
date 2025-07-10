library(tidyverse)
library(tidymodels)

set.seed(100)

# CAM Data
source(here::here("Code/Examples/Criteo Display Advertising", "Read in Criteo Data.R"))
synthesized_data <- read_csv(here::here("Data/Criteo Display Advertising", "Criteo_Display_synthesized.csv"))

# Get testing Data
test_data <- read.delim(here::here("Data/Criteo Display Advertising", "train.txt"),
                                      sep = "\t",
                                      header = FALSE,
                                      nrows = 10000,
                                      na.strings = "",
                                      col.names = col_names,
                                      stringsAsFactors = FALSE,
                                      skip = 70000) |>
  tibble()

# Select a Subset of Variables
test_data <- test_data |>
  select(label, int1, int2, int3, int4, int5, int6,
         cat9, cat20) |>
  filter(!if_any(everything(), is.na)) |>
  dummy_cols(select_columns = c("cat9", "cat20"),
             remove_first_dummy = TRUE,
             remove_selected_columns = TRUE)

synthesized_data <- synthesized_data |>
  mutate(cat9_a73ee510 = as.factor(cat9_a73ee510),
         cat9_a18233ea = as.factor(cat9_a18233ea),
         cat20_a458ea53 = as.factor(cat20_a458ea53),
         cat20_b1252a9d = as.factor(cat20_b1252a9d),
         int1 = round(int1),
         int2 = round(int2),
         int3 = round(int3),
         int4 = round(int4),
         int5 = round(int5),
         int6 = round(int6),
         label = as.factor(label)
  ) |>
  select(-`...1`)
train_data <- train_data |>
  mutate(cat9_a73ee510 = as.factor(cat9_a73ee510),
         cat9_a18233ea = as.factor(cat9_a18233ea),
         cat20_a458ea53 = as.factor(cat20_a458ea53),
         cat20_b1252a9d = as.factor(cat20_b1252a9d),
         label = as.factor(label)
  )
test_data <- test_data |>
  mutate(cat9_a73ee510 = as.factor(cat9_a73ee510),
         cat9_a18233ea = as.factor(cat9_a18233ea),
         cat20_a458ea53 = as.factor(cat20_a458ea53),
         cat20_b1252a9d = as.factor(cat20_b1252a9d),
         label = as.factor(label)
         )

recipe <- train_data |>
  recipe(label ~ int1 + int2 + int3 + int4 + int5 + int6 +
           cat9_a73ee510 + cat9_a18233ea + cat20_a458ea53 + cat20_b1252a9d)

recipe_synth <- synthesized_data |>
  recipe(label ~ int1 + int2 + int3 + int4 + int5 + int6 +
           cat9_a73ee510 + cat9_a18233ea + cat20_a458ea53 + cat20_b1252a9d)



# Synthesized Fit: Company has Synthesized Data ---------------------------
workflow_synth <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe_synth)

# Fit the model, get predictive accuracy
log_fit_synth <- fit(workflow_synth, data = synthesized_data)
preds_synth <- predict(log_fit_synth, new_data = test_data, type = "class") |>
  bind_cols(test_data)
metrics_synth <- preds_synth |>
  metrics(truth = label, estimate = .pred_class)
accuracy_synth <- metrics_synth |>
  filter(.metric == "accuracy")

# Complete Fit: Company has data for all ----------------------------------
workflow_full <- workflow() |>
  add_model(logistic_reg()) |>
  add_recipe(recipe)

# Fit the model, get predictive accuracy
log_fit_full <- fit(workflow_full, data = train_data)
preds_full <- predict(log_fit_full, new_data = test_data, type = "class") |>
  bind_cols(test_data)
metrics_full <- preds_full |>
  metrics(truth = label, estimate = .pred_class)
accuracy_full <- metrics_full |>
  filter(.metric == "accuracy")

bind_rows(accuracy_synth |>
            mutate(data = "synthetic"),
          accuracy_full |>
            mutate(data = "full"))

# Should get .744 synthetic,
# .745 full


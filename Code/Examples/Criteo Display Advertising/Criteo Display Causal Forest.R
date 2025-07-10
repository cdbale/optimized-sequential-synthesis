library(tidyverse)
library(grf)
library(cli)

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
  select(-`...1`) |>
  select(label, everything())

# Original ----------------------------------------------------------------

outcome <- cf_comparison_checks(
train_data,
synthesized_data,
test_data,
outcome_position = 1
)

outcome$var_imp |>
  mutate(abs_diff = abs(diff)) |>
  arrange(desc(abs_diff)) |>
  print(n = 600)

outcome$prediction_values |>
  filter(!str_detect(treatment_variable, "int")) |>
    pivot_longer(cols = contains("vals"),
                 names_to = "val_type",
                 values_to = "prediction_vals") |>
    ggplot(mapping = aes(x = prediction_vals, fill = val_type)) +
    geom_density() +
    theme_minimal() +
  facet_wrap(~treatment_variable)


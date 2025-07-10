source(here::here("Code", "Causal Forest Diagnostics.R"))

# Convert to numeric and revert to binary
convert_to_numeric <- function(dataframe) {
  dataframe <- dataframe |>
    mutate(churn = as.numeric(as.character(churn)),
           hiking_int = as.numeric(as.character(hiking_int)),
           sustain_int = as.numeric(as.character(sustain_int)),
           online_int = as.numeric(as.character(online_int)))
  return(dataframe)
}

source("Code/Examples/Simulated Examples/Regressions - Churn Full Dependence + Affinity Attribute Full Dependence.R")

# Clean data
training <- training |>
  select(-id) |>
  convert_to_numeric()
synthesized_data <- synthesized_data |>
  select(-`...1`) |>
  select(churn, amount_spent, num_visits, everything()) |>
  convert_to_numeric()
testing <- testing |>
  select(-id) |>
  convert_to_numeric()

# Diagnostics
outcome_churnFULL_aaFULL <- cf_comparison_checks(
  training,
  synthesized_data,
  testing,
  outcome_position = 1
)

source("Code/Examples/Simulated Examples/Regressions - Churn Full Dependence + Affinity Attribute Independence.R")

# Clean data
training <- training |>
  select(-id) |>
  convert_to_numeric()
synthesized_data <- synthesized_data |>
  select(-`...1`) |>
  select(churn, amount_spent, num_visits, everything()) |>
  convert_to_numeric()
testing <- testing |>
  select(-id) |>
  convert_to_numeric()

# Diagnostics
outcome_churnFULL_aaIND <- cf_comparison_checks(
  training,
  synthesized_data,
  testing,
  outcome_position = 1
)

source("Code/Examples/Simulated Examples/Regressions - Churn Partial Dependence + Affinity Attribute Independence.R")
# Clean data
training <- training |>
  select(-id) |>
  convert_to_numeric()
synthesized_data <- synthesized_data |>
  select(-`...1`) |>
  select(churn, amount_spent, num_visits, everything()) |>
  convert_to_numeric()
testing <- testing |>
  select(-id) |>
  convert_to_numeric()

# Diagnostics
outcome_churnPART_aaIND <- cf_comparison_checks(
  training,
  synthesized_data,
  testing,
  outcome_position = 1
)

# Final Comparisons -------------------------------------------------------

outcome_churnPART_aaIND$prediction_values |>
  filter(!str_detect(treatment_variable, "int")) |>
  pivot_longer(cols = contains("vals"),
               names_to = "val_type",
               values_to = "prediction_vals") |>
  ggplot(mapping = aes(x = prediction_vals, fill = val_type)) +
  geom_density(alpha = .5) +
  theme_minimal() +
  facet_wrap(~treatment_variable)

outcome_churnFULL_aaIND$prediction_values |>
  filter(!str_detect(treatment_variable, "int")) |>
  pivot_longer(cols = contains("vals"),
               names_to = "val_type",
               values_to = "prediction_vals") |>
  ggplot(mapping = aes(x = prediction_vals, fill = val_type)) +
  geom_density(alpha = .5) +
  theme_minimal() +
  facet_wrap(~treatment_variable)

outcome_churnFULL_aaFULL$prediction_values |>
  filter(str_detect(treatment_variable, "int")) |>
  pivot_longer(cols = contains("vals"),
               names_to = "val_type",
               values_to = "prediction_vals") |>
  ggplot(mapping = aes(x = prediction_vals, fill = val_type)) +
  geom_density(alpha = .5) +
  theme_minimal() +
  facet_wrap(~treatment_variable)


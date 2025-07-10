library(tidyverse)
library(fastDummies)

telco <- read_csv(here::here("Data/IBM", "IBM_Telco.csv"))

# Gender (0 for Female, 1 for Male)
telco <- telco |>
  mutate(gender = ifelse(gender == "Female", 0, 1))

# Partner, Dependents, Phone Service, Multiple Lines, Online Security,
# Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies,
# Paperless Billing

telco <- telco |>
  mutate(Partner = ifelse(Partner == "Yes", 1, 0)) |>
  mutate(Dependents = ifelse(Dependents == "Yes", 1, 0)) |>
  mutate(PhoneService = ifelse(PhoneService == "Yes", 1, 0)) |>
  mutate(MultipleLines = ifelse(MultipleLines == "Yes", 1, 0)) |>
  mutate(OnlineSecurity = ifelse(OnlineSecurity == "Yes", 1, 0)) |>
  mutate(OnlineBackup = ifelse(OnlineBackup == "Yes", 1, 0)) |>
  mutate(DeviceProtection = ifelse(DeviceProtection == "Yes", 1, 0)) |>
  mutate(TechSupport = ifelse(TechSupport == "Yes", 1, 0)) |>
  mutate(StreamingTV = ifelse(StreamingTV == "Yes", 1, 0)) |>
  mutate(StreamingMovies = ifelse(StreamingMovies == "Yes", 1, 0)) |>
  mutate(PaperlessBilling = ifelse(PaperlessBilling == "Yes", 1, 0))

# Churn
telco <- telco |>
  mutate(Churn = ifelse(Churn == "Yes", 1, 0))


# Dummy Code
telco <- telco |>
  dummy_cols(select_columns = c("InternetService", "Contract", "PaymentMethod"),
             remove_first_dummy = TRUE,
             remove_selected_columns = TRUE) |>
  select(-customerID)

# Remove NAs
telco <- telco |>
  filter(if_all(everything(), ~ !is.na(.)))

# telco |> write_csv(here::here("Data/IBM", "IBM_Telco_Cleaned.csv"))


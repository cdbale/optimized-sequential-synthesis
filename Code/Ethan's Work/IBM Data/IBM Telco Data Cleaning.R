library(tidyverse)

telco <- read_csv("IBM_Telco.csv")

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

glimpse(telco)

telco |> write_csv("IBM_Telco_Cleaned.csv")

any(is.na(telco))

colSums(is.na(telco)) > 0


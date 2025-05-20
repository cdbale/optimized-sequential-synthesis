library(tidyverse)

# Number of Customers
num_cust <- 10000

# Customer ID
customer_id <- c()
for (i in 1:num_cust) {
  customer_id[i] <- 1000 + i
}


# Variable 1 - Age --------------------------------------------------------
# Truly Independent

sample_ages <- function(n) {
  ## custom age pdf
  age_pdf <- function(x) {
    if (x < 18 || x > 96) return(0)
    if (x >= 21 && x <= 60) return(1)
    if (x >= 18 && x < 21) return((x - 18) / (21 - 18))  # Ramps up from 18 to 21
    if (x > 60 && x <= 96) return((96 - x) / (96 - 60))  # Tapers down from 60 to 96
  }
  v_pdf <- Vectorize(age_pdf)
  ## rejection sampling
  samples <- c()
  while (length(samples) < n) {
    candidates <- runif(n, min = 18, max = 96)
    probs <- v_pdf(candidates)
    keep <- runif(n) < probs
    samples <- c(samples, candidates[keep])
    samples <- samples[1:min(length(samples), n)]  # Keep only up to n samples
  }
  return(round(samples))
}

age <- sample_ages(num_cust)

## Check distribution
tibble(age = age) |>
  ggplot(aes(x = age)) +
  geom_density()


# Variable 2 - Region -----------------------------------------------------
# Truly Independent


# region_options <- c(
#   "Pacific Northwest",
#   "East Coast",
#   "Midwest",
#   "Great Plains",
#   "South",
#   "West",
#   "California")
#
# region <- sample(region_options,
#                  num_cust,
#                  replace = TRUE,
#                  prob = c(.35,.05,.07,.05,.05,.25,.18))
#
# ## Check Distribution
# tibble(region = region) |>
#   ggplot(mapping = aes(x = region)) +
#   geom_bar()


# Revamped Variable 2 - Number of Online Visits ---------------------------
# Dependent on Age

num_visits <- c()

for (i in 1:num_cust) {
  num_visits[i] <- round(100 - age[i] + rnorm(1, 10, 10))
  num_visits[i] <- ifelse(num_visits[i] < 1, 1, num_visits[i])
}

# Variable 3 - Interest in Hiking -----------------------------------------
# A Partial Function of Age

hiking_int <- c()

abs_age_vec <- c()
age_impact_vec <- c()

for (i in 1:num_cust) {
  abs_age <- abs(age[i] - 28)
  age_impact <- abs_age/age[i]
  hiking_int[i] <- rbinom(1, 1, prob = .71 - age_impact)
}

## Check Distribution
tibble(hiking_int = hiking_int,
       age = age) |>
  ggplot(mapping = aes(x = age)) +
  geom_density() +
  facet_wrap(~as.character(hiking_int))

# Variable 4 - Interest in Sustainability ---------------------------------
# A Partial Function of Interest in Hiking and Region

# region_probs <- c(
#   .7, .6, .4, .3, .2, .5, .5
# )


sustain_int <- c()
for (i in 1:num_cust) {
  prob_hiking <- hiking_int[i] * .2
  # prob_region <- region_probs[which(region_options == region[i])]
  sustain_int[i] <- rbinom(1, 1, prob = prob_hiking + rnorm(1, .5, .1))
  sustain_int[i] <- ifelse(is.na(sustain_int[i]), 0, sustain_int[i])
}

## Check Distribution
tibble(sustain_int = sustain_int,
       hiking_int = hiking_int) |>
ggplot(mapping = aes(x = sustain_int, fill = as.character(hiking_int))) +
  geom_bar(position = "fill")

## Check Distribution - Region
# tibble(sustain_int = sustain_int,
#        region = region) |>
#   ggplot(mapping = aes(x = sustain_int, fill = as.character(region))) +
#   geom_bar(position = "fill")


# Variable 5 - Active Online ----------------------------------------------
# A Partial Function of Interest in Hiking

online_int <- c()
for (i in 1:num_cust) {
  prob <- hiking_int[i] * .1
  online_int[i] <- rbinom(1, 1, prob = .9 - prob)
}

# Variable 6 - Amount Spent -----------------------------------------------
# A Partial Function of all 5 previous variables

amount_spent <- c()

# region_probs <- c(
#   400, 120, 110, 100, 50, 220, 450
# )

for (i in 1:num_cust) {
  abs_age <- abs(age[i] - 46)
  age_impact <- abs_age/age[i]
  amount_spent[i] <- 600 +
    age_impact * -600 +
    # region_probs[which(region_options == region[i])] +
    num_visits[i] * 4 +
    200 * hiking_int[i] +
    400 * sustain_int[i] +
    100 * online_int[i] +
    rnorm(1, mean = 0, sd = 50)
}

for(i in 1:num_cust) {
  amount_spent[i] <- ifelse(amount_spent[i] < 0, 0, amount_spent[i])
}

## Check Distribution
tibble(amount_spent = amount_spent) |>
  ggplot(mapping = aes(x = amount_spent)) +
  geom_density()


# Variable 7 - Churn ------------------------------------------------------
# A Function of All 6 Variables

# Create Churn first as a linear equation with different weights

# region_probs <- c(
#   4, 1, .5, .5, 1, 3, 3
# )

churn <- c()

for (i in 1:num_cust) {
  abs_age <- abs(age[i] - 25)
  age_impact <- abs_age/age[i]
  churn[i] <- age_impact * -5 +
    # region_probs[which(region_options == region[i])] +
    (num_visits[i] / -30) +
    6 * hiking_int[i] +
    10 * sustain_int[i] +
    4 * online_int[i] +
    rnorm(1, mean = 0, sd = 2.5)
}

## Check Distribution
tibble(churn = churn) |>
  ggplot(mapping = aes(x = churn)) +
  geom_density()


churn_binomial <- ifelse(churn < median(churn), 1, 0)

## Create a Dataframe
simulated_data <- tibble(
  churn = churn_binomial,
  amount_spent = amount_spent,
  num_visits = num_visits,
  age = age,
  hiking_int = hiking_int,
  sustain_int = sustain_int,
  online_int = online_int,
  id = customer_id
)

# save simulated data

path_to_save <- "../../Data/Simulations/"

if (!dir.exists(path_to_save)){
  dir.create(path_to_save, recursive=TRUE)
  write.csv(simulated_data, paste0(path_to_save, "churn_simulated.csv"), row.names=FALSE)
} else {
  write.csv(simulated_data, paste0(path_to_save, "churn_simulated.csv"), row.names=FALSE)
}


# Check Relationships -----------------------------------------------------

# Sustainability
simulated_data |>
  ggplot(mapping = aes(x = churn, fill = as.character(sustain_int))) +
  geom_bar(position = "fill")

# Hiking
simulated_data |>
  ggplot(mapping = aes(x = churn, fill = as.character(hiking_int))) +
  geom_bar(position = "fill")

# Online
simulated_data |>
  ggplot(mapping = aes(x = churn, fill = as.character(online_int))) +
  geom_bar(position = "fill")

# Age
simulated_data |>
  ggplot(mapping = aes(x = age)) +
  geom_density() +
  facet_wrap(~churn)

# Region
simulated_data |>
  ggplot(mapping = aes(x = churn, fill = as.character(region))) +
  geom_bar(position = "fill")

# Amount Spent
simulated_data |>
  ggplot(mapping = aes(x = amount_spent)) +
  geom_density() +
  facet_wrap(~churn)

# Transform Simulated Data
simulated_data <- simulated_data |>
  mutate(churn = as.factor(churn),
         hiking_int = as.factor(hiking_int),
         sustain_int = as.factor(sustain_int),
         online_int = as.factor(online_int))

simulated_data |>
  ggplot(mapping = aes(x = num_visits)) +
  geom_density() +
  facet_wrap(~churn)

rm(list = setdiff(ls(), "simulated_data"))


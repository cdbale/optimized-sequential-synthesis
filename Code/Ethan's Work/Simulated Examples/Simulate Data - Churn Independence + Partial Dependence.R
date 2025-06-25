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

# Revamped Variable 2 - Number of Online Visits ---------------------------
# Dependent on Age

num_visits <- c()

for (i in 1:num_cust) {
  num_visits[i] <- round(100 - age[i] + rnorm(1, 10, 10))
  num_visits[i] <- ifelse(num_visits[i] < 1, 1, num_visits[i])
}

tibble(num_visits = num_visits,
       age = age) |>
  ggplot(aes(x = num_visits, y = age)) +
  geom_point()

# Variable 3 - Interest in Hiking -----------------------------------------
# Completely Independent, 30% chance

hiking_int <- rbinom(num_cust, 1, .3)

# Variable 4 - Interest in Sustainability ---------------------------------
# Completely Independent, 60% chance

sustain_int <- rbinom(num_cust, 1, .6)

# Variable 5 - Active Online ----------------------------------------------
# Completely Independent, 80% chance

online_int <- rbinom(num_cust, 1, .8)

# Variable 6 - Amount Spent -----------------------------------------------
# A Partial Function of all 5 previous variables

amount_spent <- c()
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

churn <- c()

for (i in 1:num_cust) {
  # abs_age <- abs(age[i] - 25)
  # age_impact <- abs_age/age[i]
  churn[i] <- age_impact * -5 +
    # region_probs[which(region_options == region[i])] +
    # (num_visits[i] / -30) +
    6 * hiking_int[i] +
    10 * sustain_int[i] +
    4 * online_int[i] +
    rnorm(1, mean = 0, sd = 2.5)
}

## Check Distribution
tibble(churn = churn) |>
  ggplot(mapping = aes(x = churn)) +
  geom_density()

churn_binary <- 1 - rbinom(num_cust, size=1, prob=exp(churn)/(1 + exp(churn)))

## Create a Dataframe
simulated_data <- tibble(
  churn = churn_binary,
  amount_spent = amount_spent,
  num_visits = num_visits,
  age = age,
  hiking_int = hiking_int,
  sustain_int = sustain_int,
  online_int = online_int,
  id = customer_id
)

# simulated_data |>
#   write_csv("part_dependent_aa_churn.csv")

# rm(list = setdiff(ls(), "simulated_data"))


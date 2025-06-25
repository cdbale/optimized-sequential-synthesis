# Packages
library(tidyverse)
library(scales)

set.seed(42)

# Number of Customers
num_cust <- 10000

# Customer ID
customer_id <- c()
for (i in 1:num_cust) {
  customer_id[i] <- 1000 + i
}

### Use Case Idea: Sonic is launching a new slush that they want to market.
### They're using advertising to launch it and want to know which ad
### campaigns they attribute sales to. Sonic knows which demographics they're
### targeting with which ads, and Google has the data of which people actually
### saw which ad. If Sonic uses LiveRamp, the data can be joined to Google.
### However, there are privacy concerns. Using synthesized data can allow this
### data to be shared while bypassing these concerns of individual recognition.

# Data: Brick and Mortar Variables (BM), and Google Variables (G)
## Outcome: Conversion (Sale)
## BM: Age
## BM: Previous Purchases
## BM: Frequency of Purchase
## G: Exposure to Push Notification in-app
## G: Exposure to YouTube Video Ad
## G: Exposure to Google Search Ad

# Variable 1: Age ---------------------------------------------------------

sample_ages <- function(n) {
  ## custom age pdf
  age_pdf <- function(x) {
    if (x < 18 || x > 96) return(0)
    if (x >= 35 && x <= 45) return(1)
    if (x >= 18 && x < 35) return((x - 18) / (35 - 18))  # Ramps up from 18 to 21
    if (x > 45 && x <= 96) return((96 - x) / (96 - 45))  # Tapers down from 60 to 96
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

# Distribution Check
ggplot(mapping = aes(x = age)) +
  geom_density()

# Variable 2: Previous Purchases ------------------------------------------

prev_pur <- round(rnorm(num_cust, mean = 20, sd = 10))

prev_pur <- ifelse(prev_pur < 1, 1, prev_pur)

# Distribution Check
ggplot(mapping = aes(x = prev_pur)) +
  geom_density()


# Variable 3: Frequency of Purchase ---------------------------------------
## Scaled; a frequency of 0 means they've only purchased once. A frequency
## of .5 means they purchase on average once a month, and as frequency
## approaches 1, they get closer to every single day.

# Raw frequency signal: log relationship to prev_pur + noise
raw_freq <- log(prev_pur + 1) + rnorm(num_cust, mean = 0, sd = 0.5)

# Rescale to [0,1] (min-max normalization)
freq_pur <- (raw_freq - min(raw_freq)) / (max(raw_freq) - min(raw_freq))
freq_pur <- ifelse(prev_pur == 1, 0, freq_pur)
freq_pur <- ifelse(freq_pur == 1, freq_pur - .0001, freq_pur)

# Distribution Check
ggplot(mapping = aes(x = freq_pur)) +
  geom_density()


# Variable 4: App Notification --------------------------------------------
## The idea being that if they have the app, they get the notification.
## So, we calculate the probability of who has the app.

# Added the +.2 and + 1 to make non-definitive probabilities
probs <- (1 - (abs(age - 24)+.2)/(max(abs(age - 24)) + 1))

app_visits <- rbinom(num_cust, size = 1, prob = probs)

# Distribution Check
ggplot(mapping = aes(x = app_visits)) +
  geom_bar()

# Variable 5: YouTube Videos ----------------------------------------------
## let's say the YouTube Ads were only targeted to people 35-44.

yt_probs <- ifelse(age < 35 | age > 44, .005, .95)

yt_ads <- rbinom(num_cust, size = 1, prob = yt_probs)

# Distribution Check
tibble(age = age, yt_ads = as.factor(yt_ads)) |>
  ggplot(mapping = aes(x = age, fill = yt_ads)) +
  geom_density()

# Variable 6: Google Search Ads -------------------------------------------
## let's say the Google Search Ads were only targeted to people under 35

gs_probs <- ifelse(age < 35, .80, .0025)

gs_ads <- rbinom(num_cust, size = 1, prob = gs_probs)

# Distribution Check
tibble(age = age, gs_ads = as.factor(gs_ads)) |>
  ggplot(mapping = aes(x = age, fill = gs_ads)) +
  geom_density()

# Outcome: Conversions ----------------------------------------------------
## The push notification was most successful, and younger people were more
## curious about the slush.

conv_probs <- (rescale(age * -1, to = c(0, 1)) * .3) +
(rescale(prev_pur, to = c(0, 1)) * .1) +
(rescale(freq_pur, to = c(0, 1)) * .2) +
(app_visits * .3) +
(yt_ads * .05) +
(gs_ads *.05) +
  rnorm(num_cust, mean = 0, sd = .005)

conv <- rbinom(num_cust, size = 1, prob = conv_probs)


# Convert to Dataframe ----------------------------------------------------

cam_data <- tibble(
  conversions = conv,
  age = age,
  prev_purchases = prev_pur,
  freq_purchases = freq_pur,
  app_ads = app_visits,
  yt_ads = yt_ads,
  gs_ads = gs_ads
)


rm(list = setdiff(ls(), "cam_data"))


## Generate synthetic IPUMS data using synthpop

# import libraries
library(tidyverse)
library(synthpop)

# import training and adversarial data
train_data <- read_csv("train_data.csv")
adv_data <- read_csv("adv_data.csv")

# view summary statistics of training data
summary(train_data)

# convert state and sex to factors
train_data$state <- as.factor(train_data$state)
train_data$sex <- as.factor(train_data$sex)
adv_data$state <- as.factor(adv_data$state)
adv_data$sex <- as.factor(adv_data$sex)

# generate synthetic data using synthpop
# we'll keep the default settings

synth_train_data <- syn(train_data, m = 20)
synth_adv_data <- syn(adv_data, m = 20)

# save synthetic data sets
if (!dir.exists("synthetic_train")) {
  dir.create("synthetic_train")
}
if (!dir.exists("synthetic_adv")) {
  dir.create("synthetic_adv")
}

for (i in seq_along(synth_train_data$syn)) {
  write_csv(synth_train_data$syn[[i]], 
            paste0("synthetic_train/synthpop_", i - 1, ".csv"))
  write_csv(synth_adv_data$syn[[i]], 
            paste0("synthetic_adv/synthpop_", i - 1, ".csv"))
}

## Generate synthetic IPUMS data using synthpop

# import libraries
library(tidyverse)
library(synthpop)

# import training and adversarial data
train_data <- read_csv("train_data.csv")
adv_data <- read_csv("adv_data.csv")

# view summary statistics of training data
summary(train_data)

# convert non_white and sex to factors
train_data$non_white <- as.factor(train_data$non_white)
train_data$sex <- as.factor(train_data$sex)
adv_data$non_white <- as.factor(adv_data$non_white)
adv_data$sex <- as.factor(adv_data$sex)

# generate synthetic data using synthpop
# we'll keep the default settings

synth_train_data <- syn(train_data, m = 3)
synth_adv_data <- syn(adv_data, m = 3)

# save synthetic data sets
for (i in seq_along(synth_train_data$syn)) {
  write_csv(synth_train_data$syn[[i]], paste0("synthetic_train/synthpop_train_data_", i-1, ".csv"))
  write_csv(synth_adv_data$syn[[i]], paste0("synthetic_adv/synthpop_adv_data_", i-1, ".csv"))
}

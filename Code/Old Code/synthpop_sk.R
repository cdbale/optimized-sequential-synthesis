## Testing Baseline Synthpop
## with Synthpop.

## Author: Cameron Bale

library(synthpop)

# import ipums data
train <- read.csv("../Data/SK/cleaned_sk_data.csv")

# transform 'non_white' and 'sex' into factors

train$age <- as.factor(train$age)
train$sex <- as.factor(train$sex)
train$state <- as.factor(train$state)

# produce 20 synthetic data sets using the default synthpop settings
synth <- syn(train, m=20, visit.sequence=c("sex", "age", "state", "latitude", "longitude"), smoothing=list("sex"="", "age"="", "state"="", "latitude"="spline", "longitude"="spline"))

# save synthetic data sets

for(i in 0:19){
  write.csv(synth$syn[[i+1]], paste0("../Data/SK/Synthetic Datasets/synthpop_baseline_", i, ".csv"), row.names=FALSE)
}

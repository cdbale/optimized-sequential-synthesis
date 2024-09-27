## Testing Baseline Synthpop
## with Synthpop.

## Author: Cameron Bale

library(synthpop)

# import ipums data
train <- read.csv("../Data/IPUMS/cleaned_ipums_data.csv")

# transform 'non_white' and 'sex' into factors

train$non_white <- as.factor(train$non_white)
train$sex <- as.factor(train$sex)

# produce 20 synthetic data sets using the default synthpop settings
synth <- syn(train, m=20, visit.sequence=c("sex", "incwage", "years_of_educ", "potential_experience", "non_white"), smoothing=list("sex"="", "incwage"="spline", "years_of_educ"="", "potential_experience"="spline", "non_white"=""))

# save synthetic data sets

for(i in 0:19){
  write.csv(synth$syn[[i+1]], paste0("../Data/IPUMS/Synthetic Datasets/synthpop_baseline_", i, ".csv"), row.names=FALSE)
}

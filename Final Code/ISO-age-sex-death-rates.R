## Model for Death Rate by age group at ISO Level

rm(list = ls())

library(tidyverse)

# First we read in the data and define a few things...
points = read.csv("../Data/pre-aggregate.csv", stringsAsFactors=TRUE)

# aggregate to the number of deaths per ISO and number observed per ISO
iso_points <- points %>%
  filter(!is.na(ISO)) %>%
  group_by(sex, ISO, age, .drop=FALSE) %>%
  summarize(num_deaths = sum(state),
            n = n(),
            .groups="drop")

##################################################
##################################################
##################################################
##################################################

# age group labels
alabs = as.character(unique(iso_points$age))

# number of age groups
Ng = length(alabs)

# iso labels
isolabs = as.character(unique(iso_points$ISO))

# number of ISOs
Ni = length(isolabs)

# sex labels
sexlabs = as.character(unique(iso_points$sex))

# number of sex levels
Ns = length(sexlabs)

# Next we organize things a bit...
# Make an array of death count where every column is a county and every row is an age group
Y = array(iso_points$num_deaths, dim = c(Ng,Ni,Ns))
# Make an array of population where every column is a county and every row is an age group
n = array(iso_points$n, dim=c(Ng,Ni,Ns))

###################
###################
# we aren't dealing with suppression
###################

# Suppression threshold
# thres = 10  

# FALSE for suppressed, TRUE for observed
# dY = !is.na(Y)

# how many suppressed per age
# nsupp = apply(!dY, 1, sum)

# note: we do not know the true Y's
# CDC did the suppression, not me

###################
###################
# insert your prior info here
###################

# vector of prior guesses at the age-sex specific mortality rates for each group. 
# This is estimated using population level rate from this website: https://covid19.who.int/region/wpro/country/kr

lambda0 <- rep(152/9583, Ng*Ns)

# proportion of population belonging to each age group
total_pop <- sum(iso_points$n)

pi <- iso_points %>%
  group_by(sex, age) %>%
  summarize(age_sex_prop = sum(n)/total_pop, .groups='drop') %>%
  pull(age_sex_prop)

pi <- array(pi, dim=c(Ng, Ns))

n0 <- array(dim=dim(n))

for (i in seq_along(sexlabs)){
  n0[,,i] <- sapply(apply(n[,,i], 2, sum), function(x) x * pi[,i])
}

# just use one lambda0 since they are currently all the same
Y0 = lambda0[1] * n0

###################
###################

# initialize your Gibbs sampler here
# we initialize lambda with random draws from the prior distribution for lambdai
# we initialize the censored Y values with a value just under the threshold
nsims=10000
# Ymiss1 = array(dim = c(length(Y[1, !dY[1,]]), nsims))
# Ymiss2 = array(dim = c(length(Y[2, !dY[2,]]), nsims))
# Ymiss3 = array(dim = c(length(Y[3, !dY[3,]]), nsims))
lami=array(dim=c(Ng,Ni,Ns,nsims))
for(s in seq_along(sexlabs)){ # 2 values for sex
  for(a in seq_along(alabs)){ # 10 age groups
    lami[a,,s,1] = rgamma(Ni, Y0[a,,s], n0[a,,s])
    # Y[a, !dY[a,]] = thres - 1
  }
}


# Ymiss1[,1] = thres - 1
# Ymiss2[,1] = thres - 1
# Ymiss3[,1] = thres - 1

for(it in 2:nsims){
  for(s in seq_along(sexlabs)){
    for(a in seq_along(alabs)){
      for(i in seq_along(isolabs)) {
        shape = Y[a, i, s] + Y0[a, i, s]
        rate = n[a, i, s] + n0[a, i, s]
        lami[a, i, s, it] = rgamma(1, shape = shape, rate = rate)
      }
    }
  }
}

# Drop the first 1000 draws for 'burn in'
# Ymiss1 = Ymiss1[,1001:nsims]
# Ymiss2 = Ymiss2[,1001:nsims]
# Ymiss3 = Ymiss3[,1001:nsims]
lami = lami[,,,1001:nsims]


#################
#################
#Get posterior samples
#of the age-adjusted rates
#################
aalami=array(dim=c(Ni, Ns, nsims-1000))

for(s in seq_along(sexlabs)){
  for(i in seq_along(isolabs)){
    aalami[i, s, ] =  pi[,s] %*% lami[,i,s,]
  }
}

aa.med <- array(dim=c(Ni, Ns))

for(s in seq_along(sexlabs)){
  aa.med[, s] <- apply(aalami[,s,],1,median)
}

aa.med <- tibble("female" = aa.med[,1],
                 "male" = aa.med[,2],
                 "ISO" = isolabs)

aa.med %>%
  gather(key="sex", value="aa_rate", -ISO) %>%
  ggplot(aes(x=ISO, y=aa_rate)) +
  geom_col(aes(fill=sex), position='dodge') +
  ylim(0.0, 0.015)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

# libraries for lookup
library(RCurl)
library(RJSONIO)
library(MBA)

# import the original training data
synth_points <- read.csv("../Data/synthetic_datasets/back_synth.csv")

# function for fetching geographical labels
zips <- function(lat, lon) {
  url <- sprintf("http://nominatim.openstreetmap.org/reverse?format=json&lat=%f&lon=%f&zoom=18&addressdetails=1", lat, lon)
  res <- fromJSON(url)
  return(res[["address"]])#[["postcode"]])
}

# test a latitude and longitude
zips(lat=37.61525, lon=126.7156)

zipcodes2 <- list()

for(i in 1:nrow(synth_points)){
  # Sys.sleep(0.95)
  zipcodes2[[i]] <- zips(synth_points$latitude[i], synth_points$longitude[i])
  print(i)
}

####__________________-___________-________--__#####

# extract ISO
ISO <- unname(sapply(zipcodes2, function(x) x["ISO3166-2-lvl4"]))

sum(is.na(ISO))

# right now postcodes has the fewest missing values

# there are 17 unique ISO codes in the data
length(unique(ISO))

## let's focus on modeling at two aggregation levels: postcode and ISO

# https://en.wikipedia.org/wiki/ISO_3166-2:KR

## note that we don't have any observations showing up in KR-50. Are these
# the NA observations? We should check.

# how many age groups?
length(unique(points$age))

# add ISO and postcode to data and save
synth_points$ISO <- ISO

write.csv(synth_points, file="../Data/synthetic_datasets/synth_points.csv", row.names=FALSE)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

## Model for Death Rate by age group at ISO Level

# First we read in the data and define a few things...
synth_points = read.csv("../Data/synthetic_datasets/synth_points.csv", stringsAsFactors=TRUE)

synth_points[,c('age', 'sex')] <- lapply(synth_points[,c('age', 'sex')], as.factor)

##### convert sex, age, state to factors #####

# aggregate to the number of deaths per ISO and number observed per ISO
iso_points <- synth_points %>%
  filter(!is.na(ISO)) %>%
  group_by(sex, ISO, age, .drop=FALSE) %>%
  summarize(num_deaths = sum(state),
            n = n(),
            .groups="drop")

# Next we organize things a bit...
# Make an array of death count where every column is a county and every row is an age group
Y = array(iso_points$num_deaths, dim = c(Ng,Ni,Ns))
# Make an array of population where every column is a county and every row is an age group
n = array(iso_points$n, dim=c(Ng,Ni,Ns))

###################
###################
# we aren't dealing with suppression
###################

# Suppression threshold
# thres = 10  

# FALSE for suppressed, TRUE for observed
# dY = !is.na(Y)

# how many suppressed per age
# nsupp = apply(!dY, 1, sum)

# note: we do not know the true Y's
# CDC did the suppression, not me

###################
###################
# insert your prior info here
###################

# vector of prior guesses at the age-sex specific mortality rates for each group. 
# This is estimated using population level rate from this website: https://covid19.who.int/region/wpro/country/kr

lambda0 <- rep(152/9583, Ng*Ns)

# proportion of population belonging to each age group
total_pop <- sum(iso_points$n)

pi <- iso_points %>%
  group_by(sex, age) %>%
  summarize(age_sex_prop = sum(n)/total_pop, .groups='drop') %>%
  pull(age_sex_prop)

pi <- array(pi, dim=c(Ng, Ns))

n0 <- array(dim=dim(n))

for (i in seq_along(sexlabs)){
  n0[,,i] <- sapply(apply(n[,,i], 2, sum), function(x) x * pi[,i])
}

# just use one lambda0 since they are currently all the same
Y0 = lambda0[1] * n0

###################
###################

# initialize your Gibbs sampler here
# we initialize lambda with random draws from the prior distribution for lambdai
# we initialize the censored Y values with a value just under the threshold
nsims=10000
# Ymiss1 = array(dim = c(length(Y[1, !dY[1,]]), nsims))
# Ymiss2 = array(dim = c(length(Y[2, !dY[2,]]), nsims))
# Ymiss3 = array(dim = c(length(Y[3, !dY[3,]]), nsims))
lami=array(dim=c(Ng,Ni,Ns,nsims))
for(s in seq_along(sexlabs)){ # 2 values for sex
  for(a in seq_along(alabs)){ # 10 age groups
    lami[a,,s,1] = rgamma(Ni, Y0[a,,s], n0[a,,s])
    # Y[a, !dY[a,]] = thres - 1
  }
}


# Ymiss1[,1] = thres - 1
# Ymiss2[,1] = thres - 1
# Ymiss3[,1] = thres - 1

for(it in 2:nsims){
  for(s in seq_along(sexlabs)){
    for(a in seq_along(alabs)){
      for(i in seq_along(isolabs)) {
        shape = Y[a, i, s] + Y0[a, i, s]
        rate = n[a, i, s] + n0[a, i, s]
        lami[a, i, s, it] = rgamma(1, shape = shape, rate = rate)
      }
    }
  }
}

# Drop the first 1000 draws for 'burn in'
# Ymiss1 = Ymiss1[,1001:nsims]
# Ymiss2 = Ymiss2[,1001:nsims]
# Ymiss3 = Ymiss3[,1001:nsims]
lami = lami[,,,1001:nsims]


#################
#################
#Get posterior samples
#of the age-adjusted rates
#################
aalami=array(dim=c(Ni, Ns, nsims-1000))

for(s in seq_along(sexlabs)){
  for(i in seq_along(isolabs)){
    aalami[i, s, ] =  pi[,s] %*% lami[,i,s,]
  }
}

aa.med <- array(dim=c(Ni, Ns))

for(s in seq_along(sexlabs)){
  aa.med[, s] <- apply(aalami[,s,],1,median)
}

aa.med <- tibble("female" = aa.med[,1],
                 "male" = aa.med[,2],
                 "ISO" = isolabs)

aa.med %>%
  gather(key="sex", value="aa_rate", -ISO) %>%
  ggplot(aes(x=ISO, y=aa_rate)) +
  geom_col(aes(fill=sex), position='dodge') +
  ylim(0.0, 0.015)





################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

# impute the ISO and zip code for backtransformed synthetic datasets

# libraries for lookup
library(RCurl)
library(RJSONIO)
library(MBA)
library(tidyverse)

# function for fetching geographical labels
zips <- function(lat, lon) {
  Sys.sleep(runif(n=1, min=0.01, max=0.1))
  url <- sprintf("http://nominatim.openstreetmap.org/reverse?format=json&lat=%f&lon=%f&zoom=18&addressdetails=1", lat, lon)
  res <- fromJSON(url)
  return(res[["address"]])#[["postcode"]])
}

# import the original training data
synth_points <- read.csv("../Data/synthetic_datasets/synthetic_full_back_0.csv")

zips(36.912132, 127.507936)

zipcodes <- list()

for(i in 1:nrow(synth_points[1:5,])){
  zipcodes[[i]] <- zips(synth_points$latitude[i], synth_points$longitude[i])
  print(i)
}

# extract ISO
ISO <- unname(sapply(zipcodes, function(x) x["ISO3166-2-lvl4"]))

ZIPS <- unname(sapply(zipcodes, function(x) x["postcode"]))

# add ISO and postcode to data and save
synth_points$ISO <- ISO

synth_points$post <- ZIPS

write.csv(synth_points, file="../Data/synthetic_datasets/synth_points_0.csv", row.names=FALSE)





address_fetcher <- function(data_index){
  
  # import the original training data
  synth_points <- read.csv(paste0("../Data/synthetic_datasets/synthetic_full_back_", data_index, ".csv"))
  
  zipcodes <- list()
  
  for(i in 1:nrow(synth_points)){
    zipcodes[[i]] <- zips(synth_points$latitude[i], synth_points$longitude[i])
    print(i)
  }
  
  # extract ISO
  ISO <- unname(sapply(zipcodes, function(x) x["ISO3166-2-lvl4"]))
  
  ZIPS <- unname(sapply(zipcodes, function(x) x["postcode"]))
  
  # add ISO and postcode to data and save
  synth_points$ISO <- ISO
  
  synth_points$post <- ZIPS
  
  write.csv(synth_points, file=paste0("../Data/synthetic_datasets/synth_points_", data_index, ".csv"), row.names=FALSE)
  
  print(paste0("Finished ", data_index))
  
  return(data_index)
  
}

address_fetcher(0)

for(i in 15:19){
  address_fetcher(i)
  print(i)
}























##################
##################
#THE BELOW CODE SHOULD BE LEFT AS-IS!
#IT ASSUMES YOU NAMED
#THE POSTERIOR MEDIANS OF THE AGE-ADJUSTED RATES
#"aalami" USING THE CODE ABOVE,
#AND WILL CREATE A MAP "PAmap.png"
#THAT WILL BE SAVED TO YOUR CURRENT DIRECTORY
##################

# load('penn.rdata')
# 
# library(maptools)
# library(RColorBrewer)
# ncols=7
# cols=brewer.pal(ncols,'RdYlBu')[ncols:1]
# tcuts=quantile(aa.med*100000,1:(ncols-1)/ncols)
# tcolb=array(rep(aa.med*100000,each=ncols-1) > tcuts,
#             dim=c(ncols-1,Ns))
# tcol =apply(tcolb,2,sum)+1
# 
# png('PAmap.png',height=520,width=1000)
# par(mar=c(0,0,0,10),cex=1)
#     plot(penn,col=cols[tcol],border='lightgray',lwd=.5)
#     legend('right',inset=c(-.15,0),xpd=TRUE,
#            legend=c(paste(
#            c('Below',round(tcuts[-(ncols-1)],0),'Over'),
#            c(' ',rep( ' - ',ncols-2),' '),
#            c(round(tcuts,0),round(tcuts[ncols-1],0)),sep='')),
#            fill=cols,title='Deaths per 100,000',bty='n',cex=1.5,
#            border='lightgray')
# dev.off()
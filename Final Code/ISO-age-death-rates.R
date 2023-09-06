## Model for Death Rate by age group at ISO Level

rm(list = ls())

library(tidyverse)

# First we read in the data and define a few things...
points = read.csv("../Data/pre-aggregate.csv", stringsAsFactors=TRUE)

# aggregate to the number of deaths per ISO and number observed per ISO
iso_points <- points %>%
  filter(!is.na(ISO)) %>%
  group_by(ISO, age, .drop=FALSE) %>%
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
Ns = length(isolabs)

# Next we organize things a bit...
# Make an array of death count where every column is a county and every row is an age group
Y = array(iso_points$num_deaths, dim = c(Ng,Ns))
# Make an array of population where every column is a county and every row is an age group
n = array(iso_points$n, dim=c(Ng,Ns))

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

# vector of prior guesses at the age-specific mortality rates for each group per 100,000. This is estimated
# from 2015 data across the U.S.

lambda0 <- rep(152/9583, Ng)

# proportion of Pennsylvania population belonging to each age group
total_pop <- sum(iso_points$n)

pi <- iso_points %>%
  group_by(age) %>%
  summarize(age_prop = sum(n)/total_pop) %>%
  pull(age_prop)

# get matrix of prior population values
n0 = sapply(apply(n, 2, sum), function(x) x * pi)

#
Y0 = lambda0 * n0

###################
###################

# initialize your Gibbs sampler here
# we initialize lambda with random draws from the prior distribution for lambdai
# we initialize the censored Y values with a value just under the threshold
nsims=10000
# Ymiss1 = array(dim = c(length(Y[1, !dY[1,]]), nsims))
# Ymiss2 = array(dim = c(length(Y[2, !dY[2,]]), nsims))
# Ymiss3 = array(dim = c(length(Y[3, !dY[3,]]), nsims))
lami=array(dim=c(Ng,Ns,nsims))
for(a in 1:Ng){ # three age groups
  lami[a,,1] = rgamma(Ns, Y0[a,], n0[a,])
  # Y[a, !dY[a,]] = thres - 1
}

# Ymiss1[,1] = thres - 1
# Ymiss2[,1] = thres - 1
# Ymiss3[,1] = thres - 1

for(it in 2:nsims){
  for(a in 1:Ng){
    ###################
    ###################
    #ADDRESS SUPPRESSED Y HERE
    ###################
    ###################
    # ncen = length(Y[a, !dY[a,]])
    # for(yc in 1:ncen) {
    #   u = runif(1, 0, ppois(thres-1, n[a, !dY[a,]][yc] * lami[a, !dY[a,], it-1][yc]))
    #   if (a == 1) {
    #     Ymiss1[yc, it] = qpois(u, n[a, !dY[a,]][yc] * lami[a, !dY[a,], it-1][yc])
    #   } else if (a == 2) {
    #     Ymiss2[yc, it] = qpois(u, n[a, !dY[a,]][yc] * lami[a, !dY[a,], it-1][yc])
    #   } else {
    #     Ymiss3[yc, it] = qpois(u, n[a, !dY[a,]][yc] * lami[a, !dY[a,], it-1][yc])
    #   }
    #   Y[a, !dY[a,]][yc] = qpois(u, n[a, !dY[a,]][yc] * lami[a, !dY[a,], it-1][yc])
    # }

    ###################
    ###################
    #ESTIMATE LAMBDA_{ia} HERE
    ###################
    ###################
    for(county in 1:Ns) {
      shape = Y[a, county] + Y0[a, county]
      rate = n[a, county] + n0[a, county]
      lami[a, county, it] = rgamma(1, shape = shape, rate = rate)
    }
  }
}


# Drop the first 1000 draws for 'burn in'
# Ymiss1 = Ymiss1[,1001:nsims]
# Ymiss2 = Ymiss2[,1001:nsims]
# Ymiss3 = Ymiss3[,1001:nsims]
lami = lami[,,1001:nsims]
#################
#################
#Get posterior samples
#of the age-adjusted rates


#################
aalami=array(dim=c(Ns, nsims-1000))
for(i in 1:Ns){
  aalami[i,]=  pi %*% lami[,i,]
}

aa.med=apply(aalami,1,median)

##################
##################
#THE BELOW CODE SHOULD BE LEFT AS-IS!
#IT ASSUMES YOU NAMED
#THE POSTERIOR MEDIANS OF THE AGE-ADJUSTED RATES
#"aalami" USING THE CODE ABOVE,
#AND WILL CREATE A MAP "PAmap.png"
#THAT WILL BE SAVED TO YOUR CURRENT DIRECTORY
##################

load('penn.rdata')

library(maptools)
library(RColorBrewer)
ncols=7
cols=brewer.pal(ncols,'RdYlBu')[ncols:1]
tcuts=quantile(aa.med*100000,1:(ncols-1)/ncols)
tcolb=array(rep(aa.med*100000,each=ncols-1) > tcuts,
            dim=c(ncols-1,Ns))
tcol =apply(tcolb,2,sum)+1

png('PAmap.png',height=520,width=1000)
par(mar=c(0,0,0,10),cex=1)
    plot(penn,col=cols[tcol],border='lightgray',lwd=.5)
    legend('right',inset=c(-.15,0),xpd=TRUE,
           legend=c(paste(
           c('Below',round(tcuts[-(ncols-1)],0),'Over'),
           c(' ',rep( ' - ',ncols-2),' '),
           c(round(tcuts,0),round(tcuts[ncols-1],0)),sep='')),
           fill=cols,title='Deaths per 100,000',bty='n',cex=1.5,
           border='lightgray')
dev.off()
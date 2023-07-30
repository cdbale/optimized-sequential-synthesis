# libraries for lookup
library(RCurl)
library(RJSONIO)
library(MBA)

# data cleaning and prep
library(tidyverse)

# import the original training data
points <- read.csv("../Data/original_unstandardized.csv")

# function for fetching geographical labels
zips <- function(lat, lon) {
  url <- sprintf("http://nominatim.openstreetmap.org/reverse?format=json&lat=%f&lon=%f&zoom=18&addressdetails=1", lat, lon)
  res <- fromJSON(url)
  return(res[["address"]])#[["postcode"]])
}

# test a latitude and longitude
zips(lat=37.61525, lon=126.7156)

# vectorize the function
Vzips <- Vectorize(zips)

# lookup geographical information
start <- Sys.time()
zipcodes2 <- Vzips(points$latitude, points$longitude)
stop <- Sys.time()
stop-start

# look at an entry
zipcodes2[[1]]

# extract quarter
quarters <- unname(sapply(zipcodes2, function(x) x["quarter"]))

# extract city
cities <- unname(sapply(zipcodes2, function(x) x["city"]))

# extract province
provinces <- unname(sapply(zipcodes2, function(x) x["province"]))

# extract postcodes
postcodes <- unname(sapply(zipcodes2, function(x) x["postcode"]))

# extract ISO
ISO <- unname(sapply(zipcodes2, function(x) x["ISO3166-2-lvl4"]))

# check for how many missing values are in each
sum(is.na(quarters))

sum(is.na(cities))

sum(is.na(provinces))

sum(is.na(postcodes))

sum(is.na(ISO))

# right now postcodes has the fewest missing values

# there are 1601 unique zip codes in the data
length(unique(postcodes))

# there are 17 unique ISO codes in the data
length(unique(ISO))

## let's focus on modeling at two aggregation levels: postcode and ISO

# https://en.wikipedia.org/wiki/ISO_3166-2:KR

## note that we don't have any observations showing up in KR-50. Are these
# the NA observations? We should check.

# how many age groups?
length(unique(points$age))

# add ISO and postcode to data and save
points$ISO <- ISO
points$postcode <- postcodes

write.csv(points, file="../Data/pre-aggregate.csv", row.names=FALSE)

################################################################################

points <- read.csv(file="../Data/pre-aggregate.csv")

points

library(spBayes)

# for the binomial model, the response vector is the number of successful
# trials at each location, and weights is the total number of trials at
# each location

bin_points_data <- points %>%
  group_by(longitude, latitude) %>%
  summarize(n = n(),
            sum_state = sum(state),
            .groups = 'drop')

bin_coords <- as.matrix(bin_points_data %>% select(longitude, latitude))

# knots can be a vector of length two with the elements corresponding to
# the number of columns and rows in the desired knot grid

knot_vec <- c(5, 5)

# response vector

y <- bin_points_data$sum_state

# weights vector

w <- bin_points_data$n

n.batch <- 200
batch.length <- 50
n.samples <- n.batch*batch.length

x <- as.matrix(rep(1, length(y)))

fit <- glm((y/w) ~ x-1, weights=w, family="binomial")
beta.starting <- coefficients(fit)
beta.tuning <- t(chol(vcov(fit)))

bin_model <- spGLM(y ~ 1, 
                   family='binomial', 
                   weights=w, 
                   coords=bin_coords,
                   knots=knot_vec,
                   starting=list("beta"=beta.starting, "phi"=0.06, "sigma.sq"=1, "w"=0),
                   tuning=list("beta"=beta.tuning, "phi"=0.5, "sigma.sq"=0.5, "w"=0.5),
                   priors=list("beta.Normal"=list(0,10), "phi.Unif"=c(0.03, 0.3), "sigma.sq.IG"=c(2, 1)),
                   amcmc=list("n.batch"=n.batch, "batch.length"=batch.length, "accept.rate"=0.43),
                   cov.model="exponential", verbose=TRUE, n.report=10)







burn.in <- 0.9*n.samples
sub.samps <- burn.in:n.samples
print(summary(window(bin_model$p.beta.theta.samples, start=burn.in)))
beta.hat <- bin_model$p.beta.theta.samples[sub.samps,"(Intercept)"]
w.hat <- bin_model$p.w.samples[,sub.samps]
p.hat <- 1/(1+exp(-(x%*%beta.hat+w.hat)))
y.hat <- apply(p.hat, 2, function(x){rbinom(length(y), size=w, prob=p.hat)})

y.hat.mu <- apply(y.hat, 1, mean)
y.hat.var <- apply(y.hat, 1, var)
##Take a look
par(mfrow=c(1,2))
surf <- mba.surf(cbind(bin_coords,y.hat.mu),no.X=100, no.Y=100, extend=TRUE)$xyz.est
image(surf, main="Interpolated mean of posterior rate\n(observed rate)")
contour(surf, add=TRUE)
# text(bin_coords, label=paste("(",y,")",sep=""))
surf <- mba.surf(cbind(bin_coords,y.hat.var),no.X=100, no.Y=100, extend=TRUE)$xyz.est
image(surf, main="Interpolated variance of posterior rate\n(observed #
of trials)")
contour(surf, add=TRUE)
# text(bin_coords, label=paste("(",w,")",sep=""))


################################################################################


## try with the full data

bin_coords <- as.matrix(points %>% select(longitude, latitude))

# knots can be a vector of length two with the elements corresponding to
# the number of columns and rows in the desired knot grid

knot_vec <- c(5, 5)

# response vector

y <- points$state

# covariates

age <- points$age

sex <- points$sex

# weights vector

w <- rep(1, length(y))

n.batch <- 200
batch.length <- 50
n.samples <- n.batch*batch.length

x <- as.matrix(rep(1, length(y)))

fit <- glm((y/w) ~ age + sex, weights=w, family="binomial")
beta.starting <- coefficients(fit)
beta.tuning <- t(chol(vcov(fit)))

bin_model <- spGLM(y ~ sex + age, 
                   family='binomial', 
                   weights=w, 
                   coords=bin_coords,
                   knots=knot_vec,
                   starting=list("beta"=beta.starting, "phi"=0.06, "sigma.sq"=1, "w"=0),
                   tuning=list("beta"=beta.tuning, "phi"=0.5, "sigma.sq"=0.5, "w"=0.5),
                   priors=list("beta.Normal"=list(rep(0, 11), rep(10, 11)), "phi.Unif"=c(0.03, 0.3), "sigma.sq.IG"=c(2, 1)),
                   amcmc=list("n.batch"=n.batch, "batch.length"=batch.length, "accept.rate"=0.43),
                   cov.model="exponential", verbose=TRUE, n.report=10)







burn.in <- 0.9*n.samples
sub.samps <- burn.in:n.samples
print(summary(window(bin_model$p.beta.theta.samples, start=burn.in)))
beta.hat <- bin_model$p.beta.theta.samples[sub.samps,"(Intercept)"]
w.hat <- bin_model$p.w.samples[,sub.samps]
p.hat <- 1/(1+exp(-(x%*%beta.hat+w.hat)))
y.hat <- apply(p.hat, 2, function(x){rbinom(length(y), size=w, prob=p.hat)})

y.hat.mu <- apply(y.hat, 1, mean)
y.hat.var <- apply(y.hat, 1, var)
##Take a look
par(mfrow=c(1,2))
surf <- mba.surf(cbind(bin_coords,y.hat.mu),no.X=100, no.Y=100, extend=TRUE)$xyz.est
image(surf, main="Interpolated mean of posterior rate\n(observed rate)")
contour(surf, add=TRUE)
# text(bin_coords, label=paste("(",y,")",sep=""))
surf <- mba.surf(cbind(bin_coords,y.hat.var),no.X=100, no.Y=100, extend=TRUE)$xyz.est
image(surf, main="Interpolated variance of posterior rate\n(observed #
of trials)")
contour(surf, add=TRUE)
# text(bin_coords, label=paste("(",w,")",sep=""))




## repeat with synthetic data











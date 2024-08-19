# Author: Cameron Bale

library(tidyverse)
library(ggplot2)

# Plot for privacy metrics for SK data
sk_privacy_metrics <- read_csv("../Results/SK/privacy_metrics.csv")

sk_privacy_metrics %>%
  mutate(Metric = factor(Metric, levels=c("IMS", "DCR", "NNDR")),
         Type = factor(Type, levels=c("Train", "MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"), labels=c("Original", "MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  facet_wrap(~Metric, scales='free') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  labs(x = "Data Type",
       y = "Metric Value",
       title = "Privacy Metrics for Original and Synthetic South Korea COVID-19 Data")

################################################################################

# plot for IMS over range of delta values

sk_ims <- read_csv("../Results/SK/ims_metrics.csv")

sk_ims %>%
  mutate(Type = factor(Type, levels=c("Original", "MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.6) +
  geom_point(aes(shape=Type), size=2.5) +
  labs(x = "Delta",
       y = "IMS",
       title = "Average IMS Across Synthetic Data Sets",
       color = "Data Type",
       shape = "Data Type")

################################################################################

# plot attribute disclosure risk results

sk_ad <- read_csv("../Results/SK/all_ad_results.csv")

sk_ad %>%
  mutate(Type = factor(Type, levels=c("MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.6) +
  geom_point(aes(shape=Type), size=2.5) +
  labs(x = "Delta",
       y = "Multiplicative Increase in Attribute Disclosure Probability",
       title = "Average Maximum Multiplicative Increase in Attribute Disclosure Probability",
       color = "Data Type",
       shape = "Data Type")
  
  
################################################################################

# plot pMSE ratios

sk_pmse <- read_csv("../Results/SK/pmse_metrics.csv")

sk_pmse_plot <- sk_pmse %>%
  mutate(Type = factor(Type, levels=c("MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  labs(x = "Data Type",
       y = "pMSE Ratio",
       title = "pMSE Ratio Distributions") +
  scale_y_continuous(name="pMSE Ratio", breaks=c(0, 1, 2, 3, 4, 5, 6, 7, 8), limits=c(0, 7.5)) +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12))

ggsave(filename="pmse_ratio_dists.pdf", plot=sk_pmse_plot, path="../Results/SK/Figures/")

################################################################################

# plot original vs. synthesized locations

sk_locs <- read_csv("../Results/SK/locations.csv")

colnames(sk_locs) <- c('Type', 'Latitude', 'Longitude', 'State')

sk_locs <- sk_locs %>%
  mutate(Type = factor(Type, levels=c("Original", "CART", "AD - CART", "MOSTLY.AI")),
         State = factor(State, levels=c(0, 1), labels=c('Living', 'Deceased')))

sk_locs %>%
  ggplot(aes(x=Longitude, y=Latitude, shape=State, color=State)) +
  geom_point(size=2.5) +
  facet_wrap(~Type)

sk_locs %>%
  ggplot(aes(x=Longitude, y=Latitude)) +
  geom_point(alpha = 0.5, color="#00BFC4", size=2.5) +
  geom_point(data=sk_locs[sk_locs$State=='Deceased',], aes(x=Longitude, y=Latitude), color="#F8766D", shape=17, size=3) +
  facet_wrap(~Type)

################################################################################
################################################################################
################################################################################
################################################################################

###### IPUMS Plots ######

################################################################################
################################################################################
################################################################################
################################################################################

# Plot for privacy metrics for IPUMS data
ipums_privacy_metrics <- read_csv("../Results/IPUMS/privacy_metrics.csv")

ipums_privacy_metrics %>%
  mutate(Metric = factor(Metric, levels=c("IMS", "DCR", "NNDR")),
         Type = factor(Type, levels=c("Train", "MNL", "CART", "MOSTLY.AI"), labels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  facet_wrap(~Metric, scales='free') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  labs(x = "Data Type",
       y = "Metric Value",
       title = "Privacy Metrics for Original and Synthetic IPUMS Data")

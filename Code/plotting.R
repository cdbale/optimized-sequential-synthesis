# Author: Cameron Bale

library(tidyverse)
library(ggplot2)
library(grid)
library(gridExtra)

# Plot for privacy metrics for SK data
sk_privacy_metrics <- read_csv("../Results/SK/privacy_metrics.csv")

sk_pm_plot <- sk_privacy_metrics %>%
  mutate(Metric = factor(Metric, levels=c("IMS", "DCR", "NNDR")),
         Type = factor(Type, levels=c("Train", "MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"), labels=c("Original", "MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  facet_wrap(~Metric, scales='free') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  labs(x = "Data Type",
       y = "Metric Value",
       title = "Privacy Metrics for Original and Synthetic South Korea COVID-19 Data")

# # Plot for privacy metrics for IPUMS data
# ipums_privacy_metrics <- read_csv("../Results/IPUMS/privacy_metrics.csv")
# 
# ipums_pm_plot <- ipums_privacy_metrics %>%
#   mutate(Metric = factor(Metric, levels=c("IMS", "DCR", "NNDR")),
#          Type = factor(Type, levels=c("Train", "MNL", "CART", "MOSTLY.AI", "Synthpop"), labels=c("Original", "MNL", "CART", "MOSTLY.AI", "Synthpop"))) %>%
#   ggplot(aes(x=Type, y=value)) +
#   geom_boxplot() +
#   facet_wrap(~Metric, scales='free') +
#   theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
#   labs(x = "Data Type",
#        y = "Metric Value",
#        title = "Privacy Metrics for Original and Synthetic IPUMS Data")

# Plot for privacy metrics for IPUMS data
ipums_privacy_metrics <- read_csv("../Results/IPUMS/non_normalized_privacy_metrics.csv")

ipums_pm_plot <- ipums_privacy_metrics %>%
  mutate(Metric = factor(Metric, levels=c("IMS", "DCR", "NNDR")),
         Type = factor(Type, levels=c("Train", "MNL", "CART", "MOSTLY.AI", "Synthpop"), labels=c("Original", "MNL", "CART", "MOSTLY.AI", "Synthpop"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  facet_wrap(~Metric, scales='free') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  labs(x = "Data Type",
       y = "Metric Value",
       title = "Privacy Metrics for Original and Synthetic IPUMS Data")

combined_pm_plot <- grid.arrange(sk_pm_plot, ipums_pm_plot)

ggsave("../Images/combined_privacy_metrics.pdf", combined_pm_plot)

################################################################################

# plot for IMS for SK data over range of delta values

sk_ims <- read_csv("../Results/SK/ims_metrics.csv")

################################################################################

## plot for presentation

sk_ims_plot <- sk_ims %>%
  mutate(Type = factor(Type, levels=c("Original", "MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  filter(Type %in% c("Original", "MNL", "CART", "MOSTLY.AI")) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.8) +
  geom_point(aes(shape=Type), size=4) +
  labs(x = "Delta",
       y = "IMS",
       title = "Identical Match Share (IMS) - South Korean COVID-19 Data",
       color = "Data Type",
       shape = "Data Type") +
  # theme(legend.position="bottom") +
  theme(plot.title = element_text(size=20, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=19, face="bold", colour = "black"),
        axis.title.y = element_text(size=19, face="bold", colour = "black"),
        axis.text=element_text(size=12)) +
  theme(legend.key.size = unit(1, 'cm'), #change legend key size
        legend.key.height = unit(1, 'cm'), #change legend key height
        legend.key.width = unit(1, 'cm'), #change legend key width
        legend.title = element_text(size=16), #change legend title font size
        legend.text = element_text(size=14)) 

sk_ims_plot

ggsave("../Images/sk_ims.pdf", sk_ims_plot)

################################################################################

sk_ims_plot <- sk_ims %>%
  mutate(Type = factor(Type, levels=c("Original", "MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.6) +
  geom_point(aes(shape=Type), size=2) +
  labs(x = "Delta",
       y = "IMS",
       title = "South Korean COVID-19 Data",
       color = "Data Type",
       shape = "Data Type") +
  theme(legend.position="bottom") +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12))

ggsave("../Images/sk_ims.pdf", sk_ims_plot)

# plot for IMS for IPUMS data over range of delta values

ipums_ims <- read_csv("../Results/IPUMS/ims_metrics.csv")

ipums_ims_plot <- ipums_ims %>%
  mutate(Type = factor(Type, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.6) +
  geom_point(aes(shape=Type), size=2) +
  labs(x = "Delta",
       y = "IMS",
       title = "CPS ASEC Data",
       color = "Data Type",
       shape = "Data Type") +
  theme(legend.position="bottom") 

combined_ims_plot <- grid.arrange(sk_ims_plot, ipums_ims_plot, nrow=1, top=textGrob("Average IMS Across Synthetic Datasets",gp=gpar(fontsize=17)))

ggsave("../Images/combined_ims.pdf", combined_ims_plot)

################################################################################

# plot attribute disclosure risk results for SK

sk_ad <- read_csv("../Results/SK/all_ad_results.csv")

################################################################################
################################################################################

# plots for presentation
sk_ad_plot <- sk_ad %>%
  mutate(Type = factor(Type, levels=c("MNL", "CART", "MOSTLY.AI", "AD-MNL", "AD-CART"))) %>%
  filter(Type %in% c("MNL", "CART", "MOSTLY.AI")) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.8) +
  geom_point(aes(shape=Type), size=5) +
  labs(x = "Delta",
       y = "Multiplicative Increase",
       title = "Attribute Disclosure Risk Assessment: South Korean COVID-19 Data",
       color = "Data Type",
       shape = "Data Type") +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12)) +
  theme(legend.key.size = unit(1, 'cm'), #change legend key size
        legend.key.height = unit(1, 'cm'), #change legend key height
        legend.key.width = unit(1, 'cm'), #change legend key width
        legend.title = element_text(size=16), #change legend title font size
        legend.text = element_text(size=14)) 

sk_ad_plot

# plots for presentation
sk_full_ad_plot <- sk_ad %>%
  mutate(Type = factor(Type, levels=c("MNL", "CART", "MOSTLY.AI", "AD-MNL", "AD-CART"))) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.8) +
  geom_point(aes(shape=Type), size=5) +
  labs(x = "Delta",
       y = "Multiplicative Increase",
       title = "Attribute Disclosure Risk Assessment: South Korean COVID-19 Data",
       color = "Data Type",
       shape = "Data Type") +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12)) +
  theme(legend.key.size = unit(1, 'cm'), #change legend key size
        legend.key.height = unit(1, 'cm'), #change legend key height
        legend.key.width = unit(1, 'cm'), #change legend key width
        legend.title = element_text(size=16), #change legend title font size
        legend.text = element_text(size=14)) 

sk_full_ad_plot

sk_ad_plot <- sk_ad %>%
  mutate(Type = factor(Type, levels=c("MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.6) +
  geom_point(aes(shape=Type), size=2.5) +
  labs(x = "Delta",
       y = "Multiplicative Increase",
       title = "South Korean COVID-19 Data",
       color = "Data Type",
       shape = "Data Type") +
  theme(legend.position="bottom")

################################################################################
################################################################################

sk_ad_plot <- sk_ad %>%
  mutate(Type = factor(Type, levels=c("MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.6) +
  geom_point(aes(shape=Type), size=2.5) +
  labs(x = "Delta",
       y = "Multiplicative Increase",
       title = "South Korean COVID-19 Data",
       color = "Data Type",
       shape = "Data Type") +
  theme(legend.position="bottom")

# plot attribute disclosure risk results for IPUMS

ipums_ad <- read_csv("../Results/IPUMS/all_ad_results.csv")

ipums_ad_plot <- ipums_ad %>%
  ggplot(aes(x=Delta, y=value, color=Type)) +
  geom_line(linewidth=0.6) +
  geom_point(aes(shape=Type), size=2.5) +
  ylim(c(NA, 8)) +
  labs(x = "Delta",
       y = "Multiplicative Increase",
       title = "CPS ASEC Data",
       color = "Data Type",
       shape = "Data Type") +
  theme(legend.position="bottom")

combined_ad_plot <- grid.arrange(sk_ad_plot, ipums_ad_plot, nrow=1, top=textGrob("Average Maximum Multiplicative Increase in Attribute Disclosure Probability",gp=gpar(fontsize=17)))

ggsave("../Images/combined_ad.pdf", combined_ad_plot)
  
################################################################################

# plot pMSE ratio distributions for SK data

sk_pmse <- read_csv("../Results/SK/pmse_metrics.csv")

################################################################################

# make plot for presentation

sk_pmse_plot <- sk_pmse %>%
  mutate(Type = factor(Type, levels=c("MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  labs(x = "Data Type",
       y = "pMSE Ratio",
       title = "pMSE Ratio Distributions - South Korean COVID-19 Data") +
  scale_y_continuous(name="pMSE Ratio", breaks=c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)) +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12)) +
  theme(legend.key.size = unit(1, 'cm'), #change legend key size
        legend.key.height = unit(1, 'cm'), #change legend key height
        legend.key.width = unit(1, 'cm'), #change legend key width
        legend.title = element_text(size=16), #change legend title font size
        legend.text = element_text(size=14)) 

sk_pmse_plot

################################################################################

sk_pmse_plot <- sk_pmse %>%
  mutate(Type = factor(Type, levels=c("MNL", "AD-MNL", "CART", "AD-CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  labs(x = "Data Type",
       y = "pMSE Ratio",
       title = "South Korean COVID-19 Data") +
  scale_y_continuous(name="pMSE Ratio", breaks=c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
  # theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
  #       axis.title.x = element_text(size=15, face="bold", colour = "black"),
  #       axis.title.y = element_text(size=15, face="bold", colour = "black"),
  #       axis.text=element_text(size=12))

ggsave(filename="../Images/sk_pmse_ratio_dists.pdf", plot=sk_pmse_plot)

# plot pMSE ratio distributions for IPUMS data

ipums_pmse <- read_csv("../Results/IPUMS/pmse_metrics.csv")

ipums_pmse_plot <- ipums_pmse %>%
  mutate(Type = factor(Type, levels=c("MNL", "CART", "MOSTLY.AI"))) %>%
  ggplot(aes(x=Type, y=value)) +
  geom_boxplot() +
  labs(x = "Data Type",
       y = "pMSE Ratio",
  title = "CPS ASEC Data") 
  # scale_y_continuous(name="pMSE Ratio", breaks=c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
  # theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
  #       axis.title.x = element_text(size=15, face="bold", colour = "black"),
  #       axis.title.y = element_text(size=15, face="bold", colour = "black"),
  #       axis.text=element_text(size=12))

combined_pmse_plot <- grid.arrange(sk_pmse_plot, ipums_pmse_plot, nrow=1, top=textGrob("pMSE Ratio Distributions",gp=gpar(fontsize=17)))

ggsave(filename="../Images/pmse_ratio_dists.pdf", plot=combined_pmse_plot)

################################################################################

# analysis-specific utility measurements

sk_asr <- read_csv("../Results/SK/analysis_specific.csv")

# calculate mean L1 distance

sk_mean_l1 <- sk_asr %>%
  filter(Measure == "L1 Distance") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean L1 Distance` = mean(value)) %>%
  ungroup()

# calculate mean CIR

sk_mean_cir <- sk_asr %>%
  filter(Measure == "CI Ratio") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean CIR` = mean(value)) %>%
  ungroup()

# need to merge confidence intervals and PIs with utility measures

sk_cis <- read_csv("../Results/SK/point_estimates_and_intervals.csv") %>%
  mutate(Parameter = factor(Parameter, levels=c("const", "latitude", "longitude", "sex", "age"), labels=c("Intercept", "Latitude", "Longitude", "Sex", "Age")))

# compute SSO

sk_all <- sk_asr %>%
  left_join(sk_cis, by=c("Variable"="Parameter", "Data Type"="Type", "index"))

sk_sso_data <- sk_all %>%
  filter(Measure %in% c("Sign Match", "Significance Match", "CI Overlap")) %>%
  pivot_wider(names_from=Measure, values_from=value) %>%
  mutate(sso_check = if_else(`Sign Match` == 1, if_else(`Significance Match` == 1, if_else(`CI Overlap` == 1, if_else(`Data Type` == "Original", "Original", "TRUE"), "FALSE"), "FALSE"), "FALSE"))

orig_row <- sk_sso_data %>%
  filter(`Data Type` == "Original")

data_types <- c("MNL", "AD - MNL", "CART", "AD - CART", "MOSTLY.AI")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

sk_sso_data <- sk_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "AD - MNL", "CART", "AD - CART", "MOSTLY.AI"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

sk_sso_data <- sk_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "AD - MNL", "CART", "AD - CART", "MOSTLY.AI"))) %>%
  group_by(Variable, `Data Type`) %>%
  arrange(`Point Estimate`) %>%
  mutate(Dataset = 1:n())

sk_lat_sso_plot <- sk_sso_data %>%
  filter(Variable == "Latitude") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Latitude")

sk_age_sso_plot <- sk_sso_data %>%
  filter(Variable == "Age") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Age")

combined_sk_sso_plot <- grid.arrange(sk_lat_sso_plot, sk_age_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for COVID-19 Data Coefficients",gp=gpar(fontsize=17)))

ggsave(filename="../Images/sk_sso.pdf", plot=combined_sk_sso_plot)

## plot for all SK coefficients

sk_int_sso_plot <- sk_sso_data %>%
  filter(Variable == "Intercept") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(NA, 0)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Intercept")

sk_long_sso_plot <- sk_sso_data %>%
  filter(Variable == "Longitude") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Longitude")

sk_sex_sso_plot <- sk_sso_data %>%
  filter(Variable == "Sex") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Sex")

full_sk_sso_plot <- grid.arrange(sk_int_sso_plot, sk_long_sso_plot, sk_sex_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for COVID-19 Data Coefficients",gp=gpar(fontsize=17)))

ggsave(filename="../Images/full_sk_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################
################################################################################
################################################################################

### FEMALE ###

# repeat the SSO plots for the IPUMS data

# analysis-specific utility measurements for IPUMS data

ipums_asr <- read_csv("../Results/IPUMS/female_analysis_specific.csv")

# calculate mean L1 distance

ipums_mean_l1 <- ipums_asr %>%
  filter(Measure == "L1 Distance") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean L1 Distance` = mean(value)) %>%
  ungroup() %>%
  mutate(`Mean L1 Distance` = round(`Mean L1 Distance`, 3)) %>%
  arrange(Variable)

# calculate mean CIR

ipums_mean_cir <- ipums_asr %>%
  filter(Measure == "CI Ratio") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean CIR` = mean(value)) %>%
  ungroup() %>%
  arrange(Variable)

################################################################################

# need to merge confidence intervals and PIs with utility measures

ipums_cis <- read_csv("../Results/IPUMS/female_point_estimates_and_intervals.csv") %>%
  mutate(Parameter = factor(Parameter, levels=c("const", "years_of_educ", "non_white", "potential_experience", "potential_experience_2", "potential_experience_3"), 
                            labels=c("Intercept", "Years of Education", "Non-white", "Potential Experience", "Potential Experience^2", "Potential Experience^3")))

# compute SSO

ipums_all <- ipums_asr %>%
  left_join(ipums_cis, by=c("Variable"="Parameter", "Data Type"="Type", "index"))

ipums_sso_data <- ipums_all %>%
  filter(Measure %in% c("Sign Match", "Significance Match", "CI Overlap")) %>%
  pivot_wider(names_from=Measure, values_from=value) %>%
  mutate(sso_check = if_else(`Sign Match` == 1, if_else(`Significance Match` == 1, if_else(`CI Overlap` == 1, if_else(`Data Type` == "Original", "Original", "TRUE"), "FALSE"), "FALSE"), "FALSE"))

orig_row <- ipums_sso_data %>%
  filter(`Data Type` == "Original")

data_types <- c("MNL", "CART", "MOSTLY.AI")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  group_by(Variable, `Data Type`) %>%
  arrange(`Point Estimate`) %>%
  mutate(Dataset = 1:n())

ipums_educ_plot <- ipums_sso_data %>%
  filter(Variable == "Years of Education") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Years of Education")

ipums_non_white_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Non-white") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Non-white")

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

ggsave(filename="../Images/female_ipums_sso.pdf", plot=combined_ipums_sso_plot)

## plot for remaining CPS ASEC coefficients

ipums_int_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Intercept") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Intercept")

ipums_exp_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience")

ipums_exp2_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^2") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(NA, 0)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Squared")

ipums_exp3_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^3") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Cubed")

full_sk_sso_plot <- grid.arrange(ipums_int_sso_plot, ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=4, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

ggsave(filename="../Images/female_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################

### MALE ###

# repeat the SSO plots for the IPUMS data

# analysis-specific utility measurements for IPUMS data

ipums_asr <- read_csv("../Results/IPUMS/male_analysis_specific.csv")

# calculate mean L1 distance

ipums_mean_l1 <- ipums_asr %>%
  filter(Measure == "L1 Distance") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean L1 Distance` = mean(value)) %>%
  ungroup() %>%
  mutate(`Mean L1 Distance` = round(`Mean L1 Distance`, 3)) %>%
  arrange(Variable)

# calculate mean CIR

ipums_mean_cir <- ipums_asr %>%
  filter(Measure == "CI Ratio") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean CIR` = mean(value)) %>%
  ungroup() %>%
  arrange(Variable)

################################################################################

# need to merge confidence intervals and PIs with utility measures

ipums_cis <- read_csv("../Results/IPUMS/male_point_estimates_and_intervals.csv") %>%
  mutate(Parameter = factor(Parameter, levels=c("const", "years_of_educ", "non_white", "potential_experience", "potential_experience_2", "potential_experience_3"), 
                            labels=c("Intercept", "Years of Education", "Non-white", "Potential Experience", "Potential Experience^2", "Potential Experience^3")))

# compute SSO

ipums_all <- ipums_asr %>%
  left_join(ipums_cis, by=c("Variable"="Parameter", "Data Type"="Type", "index"))

ipums_sso_data <- ipums_all %>%
  filter(Measure %in% c("Sign Match", "Significance Match", "CI Overlap")) %>%
  pivot_wider(names_from=Measure, values_from=value) %>%
  mutate(sso_check = if_else(`Sign Match` == 1, if_else(`Significance Match` == 1, if_else(`CI Overlap` == 1, if_else(`Data Type` == "Original", "Original", "TRUE"), "FALSE"), "FALSE"), "FALSE"))

orig_row <- ipums_sso_data %>%
  filter(`Data Type` == "Original")

data_types <- c("MNL", "CART", "MOSTLY.AI")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  group_by(Variable, `Data Type`) %>%
  arrange(`Point Estimate`) %>%
  mutate(Dataset = 1:n())

ipums_educ_plot <- ipums_sso_data %>%
  filter(Variable == "Years of Education") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Years of Education")

ipums_non_white_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Non-white") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Non-white")

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

ggsave(filename="../Images/male_ipums_sso.pdf", plot=combined_ipums_sso_plot)

## plot for remaining CPS ASEC coefficients

ipums_int_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Intercept") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Intercept")

ipums_exp_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience")

ipums_exp2_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^2") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(NA, 0)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Squared")

ipums_exp3_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^3") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Cubed")

full_sk_sso_plot <- grid.arrange(ipums_int_sso_plot, ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=4, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

ggsave(filename="../Images/male_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################
################################################################################
################################################################################

### Repeat analysis specific IPUMS with non-normalized data
### FEMALE ###

# repeat the SSO plots for the IPUMS data

# analysis-specific utility measurements for IPUMS data

ipums_asr <- read_csv("../Results/IPUMS/non_normalized_female_analysis_specific.csv")

# calculate mean L1 distance

ipums_mean_l1 <- ipums_asr %>%
  filter(Measure == "L1 Distance") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean L1 Distance` = mean(value)) %>%
  ungroup() %>%
  mutate(`Mean L1 Distance` = round(`Mean L1 Distance`, 3)) %>%
  arrange(Variable)

# calculate mean CIR

ipums_mean_cir <- ipums_asr %>%
  filter(Measure == "CI Ratio") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean CIR` = mean(value)) %>%
  ungroup() %>%
  arrange(Variable)

################################################################################

# need to merge confidence intervals and PIs with utility measures

ipums_cis <- read_csv("../Results/IPUMS/non_normalized_female_point_estimates_and_intervals.csv") %>%
  mutate(Parameter = factor(Parameter, levels=c("const", "years_of_educ", "non_white", "potential_experience", "potential_experience_2", "potential_experience_3"), 
                            labels=c("Intercept", "Years of Education", "Non-white", "Potential Experience", "Potential Experience^2", "Potential Experience^3")))

# compute SSO

ipums_all <- ipums_asr %>%
  left_join(ipums_cis, by=c("Variable"="Parameter", "Data Type"="Type", "index"))

ipums_sso_data <- ipums_all %>%
  filter(Measure %in% c("Sign Match", "Significance Match", "CI Overlap")) %>%
  pivot_wider(names_from=Measure, values_from=value) %>%
  mutate(sso_check = if_else(`Sign Match` == 1, if_else(`Significance Match` == 1, if_else(`CI Overlap` == 1, if_else(`Data Type` == "Original", "Original", "TRUE"), "FALSE"), "FALSE"), "FALSE"))

orig_row <- ipums_sso_data %>%
  filter(`Data Type` == "Original")

data_types <- c("MNL", "CART", "MOSTLY.AI")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  group_by(Variable, `Data Type`) %>%
  arrange(`Point Estimate`) %>%
  mutate(Dataset = 1:n())

ipums_educ_plot <- ipums_sso_data %>%
  filter(Variable == "Years of Education") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Years of Education")

ipums_non_white_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Non-white") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Non-white")

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

ggsave(filename="../Images/non_normalized_female_ipums_sso.pdf", plot=combined_ipums_sso_plot)

## plot for remaining CPS ASEC coefficients

ipums_int_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Intercept") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Intercept")

ipums_exp_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience")

ipums_exp2_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^2") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(NA, 0)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Squared")

ipums_exp3_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^3") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Cubed")

full_sk_sso_plot <- grid.arrange(ipums_int_sso_plot, ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=4, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

ggsave(filename="../Images/non_normalized_female_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################

### MALE ###

# repeat the SSO plots for the non-normalized IPUMS data

# analysis-specific utility measurements for IPUMS data

ipums_asr <- read_csv("../Results/IPUMS/non_normalized_male_analysis_specific.csv")

# calculate mean L1 distance

ipums_mean_l1 <- ipums_asr %>%
  filter(Measure == "L1 Distance") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean L1 Distance` = mean(value)) %>%
  ungroup() %>%
  mutate(`Mean L1 Distance` = round(`Mean L1 Distance`, 3)) %>%
  arrange(Variable)

# calculate mean CIR

ipums_mean_cir <- ipums_asr %>%
  filter(Measure == "CI Ratio") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Mean CIR` = mean(value)) %>%
  ungroup() %>%
  arrange(Variable)

################################################################################

# need to merge confidence intervals and PIs with utility measures

ipums_cis <- read_csv("../Results/IPUMS/non_normalized_male_point_estimates_and_intervals.csv") %>%
  mutate(Parameter = factor(Parameter, levels=c("const", "years_of_educ", "non_white", "potential_experience", "potential_experience_2", "potential_experience_3"), 
                            labels=c("Intercept", "Years of Education", "Non-white", "Potential Experience", "Potential Experience^2", "Potential Experience^3")))

# compute SSO

ipums_all <- ipums_asr %>%
  left_join(ipums_cis, by=c("Variable"="Parameter", "Data Type"="Type", "index"))

ipums_sso_data <- ipums_all %>%
  filter(Measure %in% c("Sign Match", "Significance Match", "CI Overlap")) %>%
  pivot_wider(names_from=Measure, values_from=value) %>%
  mutate(sso_check = if_else(`Sign Match` == 1, if_else(`Significance Match` == 1, if_else(`CI Overlap` == 1, if_else(`Data Type` == "Original", "Original", "TRUE"), "FALSE"), "FALSE"), "FALSE"))

orig_row <- ipums_sso_data %>%
  filter(`Data Type` == "Original")

data_types <- c("MNL", "CART", "MOSTLY.AI")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "MOSTLY.AI"))) %>%
  group_by(Variable, `Data Type`) %>%
  arrange(`Point Estimate`) %>%
  mutate(Dataset = 1:n())

ipums_educ_plot <- ipums_sso_data %>%
  filter(Variable == "Years of Education") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Years of Education")

ipums_non_white_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Non-white") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Non-white")

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

ggsave(filename="../Images/non_normalized_male_ipums_sso.pdf", plot=combined_ipums_sso_plot)

## plot for remaining CPS ASEC coefficients

ipums_int_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Intercept") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Intercept")

ipums_exp_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#3399FF", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience")

ipums_exp2_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^2") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(NA, 0)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Squared")

ipums_exp3_sso_plot <- ipums_sso_data %>%
  filter(Variable == "Potential Experience^3") %>%
  ggplot(aes(x=Dataset, y=`Point Estimate`, color=factor(sso_check, levels=c("Original", "TRUE", "FALSE")))) +
  geom_point() +
  geom_errorbar(aes(ymax = `Upper Bound`, ymin = `Lower Bound`)) +
  # ylim(c(0, NA)) +
  facet_wrap(~`Data Type`, nrow=1) + 
  scale_color_manual(values=c("#000000", "#CC0000")) +
  theme(legend.position="bottom") +
  labs(y = "Estimate",
       color="Sign, Significance Match, and Overlap",
       title="Potential Experience Cubed")

full_sk_sso_plot <- grid.arrange(ipums_int_sso_plot, ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=4, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

ggsave(filename="../Images/non_normalized_male_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################



































































































################################################################################

## for presentation, make zoomed in plot of a single point with state=1

sk_locs <- read_csv("../Results/SK/locations.csv")

colnames(sk_locs) <- c('Type', 'Latitude', 'Longitude', 'State')

sk_locs <- sk_locs %>%
  mutate(Type = factor(Type, levels=c("Original", "CART", "AD - CART", "MOSTLY.AI")),
         State = factor(State, levels=c(0, 1), labels=c('Living', 'Deceased')))

sk_locs %>%
  filter(Type == "Original", State=="Living") %>%
  ggplot(aes(x=Longitude, y=Latitude)) +
  geom_point(alpha = 0.75, color="#00BFC4", size=4.5) +
  geom_point(data=sk_locs[sk_locs$State=='Deceased' & sk_locs$Type == "Original",], aes(x=Longitude, y=Latitude), color="#F8766D", shape=17, size=5) +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12))

small_locs <- sk_locs %>%
  filter(Latitude > 0.75 & Latitude < 1 & Longitude > -1 & Longitude < -0.5)

small_locs %>%
  filter(Type == "Original", State=="Living") %>%
  ggplot(aes(x=Longitude, y=Latitude)) +
  geom_point(alpha = 0.75, color="#00BFC4", size=4.5) +
  geom_point(data=small_locs[small_locs$State=='Deceased' & small_locs$Type == "Original",], aes(x=Longitude, y=Latitude), color="#F8766D", shape=17, size=5) +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12))

small_locs <- sk_locs %>%
  filter(Latitude > 0.875 & Latitude < 0.925 & Longitude > -0.75 & Longitude < -0.65)

small_locs %>%
  filter(Type == "Original", State=="Living") %>%
  ggplot(aes(x=Longitude, y=Latitude)) +
  geom_point(alpha = 0.75, color="#00BFC4", size=4.5) +
  geom_point(data=small_locs[small_locs$State=='Deceased' & small_locs$Type == "Original",], aes(x=Longitude, y=Latitude), color="#F8766D", shape=17, size=5) +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12))

small_locs %>%
  filter(Type == "Original") %>%
  mutate(State = factor(State, levels=c("Deceased", "Living"), labels=c("Deceased", "Survived"))) %>%
  ggplot(aes(x=Longitude, y=Latitude, color=State, shape=State)) +
  geom_point(alpha = 0.75, size=5) +
  labs(title="Attribute Disclosure Example - South Korean COVID-19 Data") +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12)) +
  theme(legend.key.size = unit(1, 'cm'), #change legend key size
        legend.key.height = unit(1, 'cm'), #change legend key height
        legend.key.width = unit(1, 'cm'), #change legend key width
        legend.title = element_text(size=16), #change legend title font size
        legend.text = element_text(size=14)) 

















small_locs %>%
  filter(Type == "Original") %>%
  ggplot(aes(x=Longitude, y=Latitude)) +
  geom_point(alpha = 0.75, color="#00BFC4", size=4.5) +
  geom_point(data=small_locs[small_locs$State=='Deceased',], aes(x=Longitude, y=Latitude), color="#F8766D", shape=17, size=5) +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12))
  
small_locs %>%
  filter(Type == "Original") %>%
  ggplot(aes(x=Longitude, y=Latitude, color=State, shape=State)) +
  geom_point(alpha = 0.75, size=4.5) +
  geom_point(data=small_locs[small_locs$State=='Deceased',], aes(x=Longitude, y=Latitude), color="#F8766D", shape=17, size=5) +
  theme(plot.title = element_text(size=16, face= "bold", colour= "black" ),
        axis.title.x = element_text(size=15, face="bold", colour = "black"),
        axis.title.y = element_text(size=15, face="bold", colour = "black"),
        axis.text=element_text(size=12))
  


# plot original vs. synthesized locations



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

# calculate proportion sign match

prop_sign <- asr %>%
  filter(Measure == "Sign Match") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Proportion Sign Match` = mean(value)) %>%
  ungroup()

# calculate proportion significance match

prop_sig <- asr %>%
  filter(Measure == "Significance Match") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Proportion Significance Match` = mean(value)) %>%
  ungroup()

# calculate proportion overlap 

prop_overlap <- asr %>%
  filter(Measure == "CI Overlap") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Proportion CI Overlap` = mean(value)) %>%
  ungroup()

sso_results %>%
  group_by(`Data Type`) %>%
  summarize(mean_SSO = mean(SSO))

################################################################################
################################################################################
################################################################################
################################################################################

###### IPUMS Plots ######

################################################################################
################################################################################
################################################################################
################################################################################

################################################################################



# calculate proportion sign match

ipums_prop_sign <- ipums_asr %>%
  filter(Measure == "Sign Match") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Proportion Sign Match` = mean(value)) %>%
  ungroup()

# calculate proportion significance match

ipums_prop_sig <- ipums_asr %>%
  filter(Measure == "Significance Match") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Proportion Significance Match` = mean(value)) %>%
  ungroup()

# calculate proportion overlap 

ipums_prop_overlap <- ipums_asr %>%
  filter(Measure == "CI Overlap") %>%
  group_by(`Data Type`, Variable) %>%
  summarize(`Proportion CI Overlap` = mean(value)) %>%
  ungroup()

# compute SSO

ipums_sso_data <- ipums_asr %>%
  group_by(`Data Type`, Measure, Variable) %>%
  mutate(id = 1:n()) %>%
  filter(Measure %in% c("Sign Match", "Significance Match", "CI Overlap")) %>%
  pivot_wider(names_from=Measure, values_from=value)

ipums_sso_results <- ipums_sso_data %>%
  mutate(sso_check = if_else(`Sign Match` == 1, if_else(`Significance Match` == 1, if_else(`CI Overlap` == 1, 1, 0), 0), 0)) %>%
  group_by(`Data Type`, Variable) %>%
  summarize(SSO = mean(sso_check))

ipums_sso_results %>%
  group_by(`Data Type`) %>%
  summarize(mean_SSO = mean(SSO))


























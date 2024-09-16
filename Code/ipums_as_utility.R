################################################################################
############## Examine Analysis Specific Utility of CPS ASEC Data ##############
################################################################################

# Author: Cameron Bale

################################################################################

library(tidyverse)
library(grid)
library(gridExtra)

# Examine female regression using the normalized data (no transformations applied)

ipums_asr <- read_csv("../Results/IPUMS/female_analysis_specific.csv")

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

data_types <- c("MNL", "CART", "Synthpop")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
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

ipums_educ_plot

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

ipums_non_white_sso_plot

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_ipums_sso.pdf", plot=combined_ipums_sso_plot)

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

ipums_int_sso_plot

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

ipums_exp_sso_plot

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

ipums_exp2_sso_plot

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

ipums_exp3_sso_plot

sk_sso_plot1 <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, ipums_int_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

sk_sso_plot2 <- grid.arrange(ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################
################################################################################

# now repeat but using the non-normalized data (but not with log income)

# Examine female regression using the normalized data (no transformations applied)

ipums_asr <- read_csv("../Results/IPUMS/nn_female_analysis_specific.csv")

################################################################################

# need to merge confidence intervals and PIs with utility measures

ipums_cis <- read_csv("../Results/IPUMS/nn_female_point_estimates_and_intervals.csv") %>%
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

data_types <- c("MNL", "CART", "Synthpop")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
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

ipums_educ_plot

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

ipums_non_white_sso_plot

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_ipums_sso.pdf", plot=combined_ipums_sso_plot)

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

ipums_int_sso_plot

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

ipums_exp_sso_plot

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

ipums_exp2_sso_plot

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

ipums_exp3_sso_plot

sk_sso_plot1 <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, ipums_int_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

sk_sso_plot2 <- grid.arrange(ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################
################################################################################

# now repeat but using the non-normalized data with the truncated and log-transformed income variable

# Examine female regression using the normalized data (no transformations applied)

ipums_asr <- read_csv("../Results/IPUMS/log_nn_female_analysis_specific.csv")

################################################################################

# need to merge confidence intervals and PIs with utility measures

ipums_cis <- read_csv("../Results/IPUMS/log_nn_female_point_estimates_and_intervals.csv") %>%
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

data_types <- c("MNL", "CART", "Synthpop")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
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

ipums_educ_plot

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

ipums_non_white_sso_plot

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_ipums_sso.pdf", plot=combined_ipums_sso_plot)

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

ipums_int_sso_plot

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

ipums_exp_sso_plot

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

ipums_exp2_sso_plot

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

ipums_exp3_sso_plot

sk_sso_plot1 <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, ipums_int_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

sk_sso_plot2 <- grid.arrange(ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)

################################################################################
################################################################################

# now repeat but using the non-normalized data with the non-truncated (shifted up by the minimum value + 1) and log-transformed income variable

# Examine female regression using the normalized data (no transformations applied)

ipums_asr <- read_csv("../Results/IPUMS/no_trunc_log_nn_female_analysis_specific.csv")

################################################################################

# need to merge confidence intervals and PIs with utility measures

ipums_cis <- read_csv("../Results/IPUMS/no_trunc_log_nn_female_point_estimates_and_intervals.csv") %>%
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

data_types <- c("MNL", "CART", "Synthpop")

new_df <- tibble()

for (i in seq_along(data_types)){
  orig_row$`Data Type` <- data_types[i]
  new_df <- new_df %>% bind_rows(orig_row)
}

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
  filter(`Data Type` != "Original") %>%
  bind_rows(new_df)

# plot the confidence intervals

ipums_sso_data <- ipums_sso_data %>%
  mutate(`Data Type` = factor(`Data Type`, levels=c("Original", "MNL", "CART", "Synthpop"))) %>%
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

ipums_educ_plot

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

ipums_non_white_sso_plot

combined_ipums_sso_plot <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, nrow=2, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients",gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_ipums_sso.pdf", plot=combined_ipums_sso_plot)

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

ipums_int_sso_plot

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

ipums_exp_sso_plot

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

ipums_exp2_sso_plot

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

ipums_exp3_sso_plot

sk_sso_plot1 <- grid.arrange(ipums_educ_plot, ipums_non_white_sso_plot, ipums_int_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

sk_sso_plot2 <- grid.arrange(ipums_exp_sso_plot, ipums_exp2_sso_plot, ipums_exp3_sso_plot, nrow=3, top=textGrob("Sign, Signifiance, and Overlap for CPS ASEC Data Coefficients", gp=gpar(fontsize=17)))

# ggsave(filename="../Images/female_full_ipums_sso.pdf", plot=full_sk_sso_plot, height=9.5)
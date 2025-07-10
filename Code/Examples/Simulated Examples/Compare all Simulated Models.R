library(tidyverse)
library(tidymodels)

source("Code/Examples/Simulated Examples/Regressions - Churn Full Dependence + Affinity Attribute Full Dependence.R")
source("Code/Examples/Simulated Examples/Regressions - Churn Full Dependence + Affinity Attribute Independence.R")
source("Code/Examples/Simulated Examples/Regressions - Churn Partial Dependence + Affinity Attribute Independence.R")

final_results <- bind_rows(final_results_churnFULL_aaFULL,
                           final_results_churnFULL_aaIND,
                           final_results_churnPART_aaIND)

# final_results |>
#   ggplot(mapping = aes(x = model, y = mean, fill = model)) +
#   geom_col(position = "dodge") +
#   facet_wrap(~data, labeller = label_wrap_gen(width = 30)) +
#   theme_minimal()


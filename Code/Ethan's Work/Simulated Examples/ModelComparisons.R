library(tidyverse)
library(tidymodels)

source("Independent AA Regressions on Simulated Data.R")
source("Regressions on Simulated Data.R")
source("Part-dependent AA Regressions on Simulated Data.R")

final_results <- bind_rows(final_results_full,
                           final_results_ind,
                           final_results_partial)

final_results |>
  ggplot(mapping = aes(x = model, y = mean, fill = model)) +
  geom_col(position = "dodge") +
  facet_wrap(~data, labeller = label_wrap_gen(width = 30)) +
  theme_minimal()

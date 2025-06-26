library(tidyverse)

# Read in Criteo Data
col_names <- c("label",
               paste0("int", 1:13),
               paste0("cat", 1:26))

train_data <- read.delim(here::here("dac", "train.txt"),
                              sep = "\t",
                              header = FALSE,
                              nrows = 30000,
                              na.strings = "",
                              col.names = col_names,
                              stringsAsFactors = FALSE) |>
  tibble()


# Select a Subset of Variables
train_data <- train_data |>
  select(label, int1, int2, int3, int4, int5, int6) |>
  filter(!if_any(everything(), is.na))

# Write the CSV
# train_data |>
#  write_csv(here::here("dac", "filtered_training_data.csv"))

# cat 6, 9, 17, 20, 22, 23

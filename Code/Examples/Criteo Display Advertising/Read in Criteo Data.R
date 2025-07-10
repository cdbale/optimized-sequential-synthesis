library(tidyverse)
library(fastDummies)

# Read in Criteo Data
col_names <- c("label",
               paste0("int", 1:13),
               paste0("cat", 1:26))

train_data <- read.delim(here::here("Data/Criteo Display Advertising", "train.txt"),
                              sep = "\t",
                              header = FALSE,
                              nrows = 30000,
                              na.strings = "",
                              col.names = col_names,
                              stringsAsFactors = FALSE) |>
  tibble()


# Select a Subset of Variables
train_data <- train_data |>
  select(label, int1, int2, int3, int4, int5, int6,
         cat9, cat20) |>
  filter(!if_any(everything(), is.na)) |>
  dummy_cols(select_columns = c("cat9", "cat20"),
             remove_first_dummy = TRUE,
             remove_selected_columns = TRUE)

colnames(train_data)


any(is.na(train_data))

# Write the CSV
 train_data |>
  write_csv(here::here("Data/Criteo Display Advertising", "Criteo Training.csv"))

# cat 6, 9, 17, 20, 22, 23

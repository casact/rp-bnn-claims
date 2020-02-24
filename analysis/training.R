# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

source("analysis/initialize-environment.R")
# source("analysis/data-prep.R")
source("analysis/model.R")

set.seed(49)

# saveRDS(simulated_cashflows, file = "simulated_cashflows.rds")

simulated_cashflows <- readRDS("data/simulated_cashflows.rds")

# dataset <- prep_datasets(simulated_cashflows, n = 0)
# saveRDS(dataset, file = "dataset.rds")

dataset <- readRDS("data/dataset.rds")

train_data_keras <- prep_keras_data(filter(dataset$training_data, year > 0))
validation_data_keras <- dataset$training_data %>% 
  filter(year > 0) %>% 
  sample_frac(0.05) %>% 
  prep_keras_data()

batch_size <- 100000

model <- make_model(n_rows = dim(train_data_keras$x[[1]])[[1]], 
                    ln_scale_bound = 0.7,
                    scale_c = 0.01)

model %>%
  compile(
    loss = list(cust_loss, cust_loss),
    loss_weights = list(1, 1),
    optimizer = optimizer_sgd(lr = 0.01, clipnorm = 1)
  )

history <- model %>%
  fit(
    x = train_data_keras$x,
    y = unname(train_data_keras$y),
    validation_data = list(validation_data_keras$x, unname(validation_data_keras$y)),
    batch_size = batch_size,
    epochs = 100,
    view_metrics = FALSE,
    verbose = 1,
    callbacks = list(callback_early_stopping(monitor = "val_loss",
                                             patience = 10, 
                                             min_delta = 0.001,
                                             restore_best_weights = FALSE),
                     callback_reduce_lr_on_plateau(factor = 0.5, patience = 5)
    )
  )

scoring_data <- bind_rows(dataset$training_data, dataset$dev_year_zero_records)

claims_to_keep <- scoring_data %>% 
  group_by(ClNr) %>% 
  summarize(max_dev = max(year)) %>% 
  # don't forecast claims already at maturity 12 (which we assume to be ultimate)
  filter(max_dev < 11)

scoring_data <- scoring_data %>% 
  inner_join(claims_to_keep, by = "ClNr")

records_to_score <- scoring_data %>% 
  group_by(ClNr) %>%
  arrange(year) %>% 
  # get the latest valuation of claim
  slice(n()) %>%
  mutate(
    paid_loss_lags = map2(paid_loss_lags, paid_loss, ~ c(.x[-1], .y)),
    claim_open_indicator_lags = map2(claim_open_indicator_lags, claim_open_indicator, ~c(.x[-1], .y)),
    scaled_dev_year = scaled_dev_year + 1/11
  )

scoring_data_keras <- records_to_score %>%
  prep_keras_data()

scoring_batch_size <- 2000

ds <- tensor_slices_dataset(scoring_data_keras$x) %>% 
  dataset_map(function(record) {
    record$claim_open_indicator_lags_ <- tf$cast(record$claim_open_indicator_lags_, k_floatx())
    record$paid_loss_lags_ <- tf$cast(record$paid_loss_lags_, k_floatx())
    record$recovery_lags_ <- tf$cast(record$recovery_lags_, k_floatx())
    record$lob_ <- tf$cast(record$lob_, tf$int32)
    record$claim_code_ <- tf$cast(record$lob, tf$int32)
    record$age_ <- tf$cast(record$age_, k_floatx())
    record$injured_part_ <- tf$cast(record$injured_part_, tf$int32)
    
    record
  }) %>% 
  dataset_batch(scoring_batch_size)

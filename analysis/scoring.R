# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

iter <- make_iterator_one_shot(ds)
dists_list <- list()
i <- 1

until_out_of_range({
  cat("scoring batch number: ", i, "\n")
  # Use CPU for inference due to memory leak issue
  with(tf$device("/cpu:0"), {
    batch <- iterator_get_next(iter)
    dist <- map(1, ~model(unname(batch)))
    dists_list <- c(dists_list, list(dist))
  })
  i <- i + 1
})

predicted_cashflows <- dists_list %>%
  map(function(batch) {
    map(batch, function(sample_dists) {
      means <- map(sample_dists, function(output) tfd_mean(output))# %>%
      k_expand_dims(means[[1]] - means[[2]])
    }) %>% 
      k_concatenate(axis = 3) %>%
      k_mean(axis = 3)
  }) %>% 
  k_concatenate(axis = 1) %>% 
  as.array()

dev_years <- records_to_score %>%
  distinct(ClNr, year) %>%
  transmute(
    development_year = list(year + 1:11)
  ) %>%
  ungroup() %>% 
  unnest() %>% 
  mutate(type = "predicted")

tidied <- predicted_cashflows %>% 
  apply(1, tibble::enframe, name = NULL) %>%
  bind_rows() %>% 
  bind_cols(dev_years) %>% 
  rename(paid_loss = value) %>% 
  filter(development_year <= 11)

claim_ids_for_comparison <- tidied %>%
  distinct(ClNr)

actual_ultimate <- dataset$cashflow_history %>%
  inner_join(claim_ids_for_comparison, by = "ClNr") %>%
  summarize(ultimate = sum(Pay)) %>%
  pull(ultimate)


predicted_future_paid <- tidied %>%
  group_by(ClNr) %>%
  summarize(mean_predicted_future_paid = sum(paid_loss))

predicted_df <- scoring_data %>%
  group_by(ClNr) %>%
  summarize(paid = sum(Pay)) %>%
  right_join(predicted_future_paid, by = "ClNr") %>%
  mutate(ultimate = paid + mean_predicted_future_paid)

nn_ultimate <- sum(predicted_df$ultimate)

mack_ultimate <- compute_mack_ultimate(
  bind_rows(dataset$training_data, dataset$dev_year_zero_records), 
  claim_ids_for_comparison
)

already_paid <- scoring_data %>% 
  group_by(ClNr) %>% 
  summarize(paid = sum(Pay)) %>% 
  pull(paid) %>% 
  sum()

# Calculate unpaid estimates
actual_future_paid <- actual_ultimate - already_paid
nn_future_paid <- predicted_future_paid$mean_predicted_future_paid %>% sum()
nn_future_paid
nn_future_paid / actual_future_paid - 1
mack_future_paid <- mack_ultimate - already_paid
mack_future_paid / actual_future_paid - 1
(actual_future_paid + already_paid) / actual_ultimate - 1

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

make_model <- function(n_rows, ln_scale_bound = 1, scale_c = 0.01) {
  
  #' Input layers
  
  #' Past paid values
  paid_loss_lags <- layer_input(shape = c(11, 1), name = "paid_loss_lags_")
  recovery_lags <- layer_input(shape = c(11, 1), name = "recovery_lags_")
  
  #' Past claim status, 1 for open 0 for closed, one-hot encoded
  claim_open_indicator_lags <- layer_input(shape = c(11, 2), name = "claim_open_indicator_lags_")
  
  #' How long has the claim developed, in years divided by 11, so this input is in [1/11, 1]
  scaled_dev_year <- layer_input(shape = 1, name = "scaled_dev_year_")
  
  #' Line of business, integer indexed
  lob <- layer_input(shape = c(1), name = "lob_")
  
  #' Occupation of claimant, integer indexed
  claim_code <- layer_input(shape = c(1), name = "claim_code_")
  
  #' Age, normalized
  age <- layer_input(shape = 1, name = "age_")
  
  #' Injured body part, integer indexed
  injured_part <- layer_input(shape = 1, name = "injured_part_")
  
  #' Condition encoder
  
  #' Categorical variables go into embedding layers
  lob_embedding <- lob %>%
    layer_embedding(4, 2) %>%
    layer_flatten()
  claim_code_embedding <- claim_code %>%
    layer_embedding(53, 2) %>%
    layer_flatten()
  injured_part_embedding <- injured_part %>%
    layer_embedding(90, 2) %>%
    layer_flatten()
  
  paid_loss_lags_gru_out <- layer_concatenate(list(paid_loss_lags, recovery_lags), axis = 2) %>%
    layer_masking(mask_value = 99999) %>%
    layer_lstm(3)
  claim_open_indicator_lags_gru_out <- claim_open_indicator_lags %>%
    layer_masking(99999) %>%
    layer_lstm(3)
  
  #' Concatenate encoded conditions
  cond <- layer_concatenate(list(
    scaled_dev_year %>% layer_flatten(),
    age %>% layer_flatten(),
    lob_embedding,
    claim_code_embedding,
    injured_part_embedding,
    paid_loss_lags_gru_out,
    claim_open_indicator_lags_gru_out
  ))
  
  prior <- function(kernel_size, bias_size = 0, dtype = NULL) {
    n <- kernel_size + bias_size
    keras_model_sequential() %>%
      layer_variable(n, dtype = dtype, trainable = FALSE) %>%
      layer_distribution_lambda(function(t) {
        tfd_independent(
          tfd_normal(loc = t, scale = 1),
          reinterpreted_batch_ndims = 1
        )
      })
  }
  
  posterior_mean_field <- function(kernel_size, bias_size = 0, dtype = NULL) {
    n <- kernel_size + bias_size
    c <- 0.01
    keras_model_sequential(list(
      layer_variable(shape = 2 * n, dtype = dtype), 
      layer_distribution_lambda(make_distribution_fn = function(t) {
        tfd_independent(
          tfd_normal(loc = t[1:n], scale = 1e-3 + k_sigmoid(c  * t[(n+1):(2*n)])),
          reinterpreted_batch_ndims = 1
        )
      })
    ))
  }
  
  out_sequence <- cond %>%
    layer_repeat_vector(11) %>%
    layer_lstm(3, return_sequences = TRUE)
  
  paid_out <- out_sequence %>% 
    layer_dense_variational(units = 4,
                            make_posterior_fn = posterior_mean_field,
                            make_prior_fn = prior,
                            kl_weight = 1 / n_rows,
                            activation = "linear") %>%
    layer_distribution_lambda(
      function(x) {
        
        d <- tfd_mixture(
          cat = tfd_categorical(logits = x[,,1:2]),
          components = list(
            tfd_transformed_distribution(
              tfd_log_normal(x[,,3], 1e-3 + ln_scale_bound * k_sigmoid(scale_c * x[,,4])),
              tfb_affine_scalar(shift = -1e-3)
            ),
            tfd_deterministic(loc = k_zeros_like(x[,,3]))
          )
        )
        
        tfd_independent(d, reinterpreted_batch_ndims = 1)
        
      },
      name = "paid_out_"
    )
  
  recovery_out <- out_sequence %>% 
    layer_dense_variational(units = 4,
                            make_posterior_fn = posterior_mean_field,
                            make_prior_fn = prior,
                            kl_weight = 1 / n_rows,
                            activation = "linear") %>%
    layer_distribution_lambda(
      function(x) {
        
        d <- tfd_mixture(
          cat = tfd_categorical(logits = x[,,1:2]),
          components = list(
            tfd_transformed_distribution(
              tfd_log_normal(x[,,3], 1e-3 + ln_scale_bound * k_sigmoid(scale_c * x[,,4])),
              tfb_affine_scalar(shift = -1e-3)
            ),
            tfd_deterministic(loc = k_zeros_like(x[,,3]))
          )
        )
        tfd_independent(d, reinterpreted_batch_ndims = 1)
      },
      name = "recovery_out_"
    )
  
  keras_model(
    inputs = c(paid_loss_lags, recovery_lags,
               claim_open_indicator_lags,
               scaled_dev_year,
               lob, claim_code, age, injured_part),
    outputs = c(paid_out, recovery_out)
  )
}

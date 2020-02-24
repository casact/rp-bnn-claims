# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#' Given a time series, return a list
#'  where each element is a vector representing a window
#'  of the time series determined by the offsets
make_series <- function(v, start_offset, end_offset, na_pad = 99999) {
  prepad_mask <- function(v, l = 11) {
    length_diff <- l - length(v)
    if (length_diff > 0) {
      c(rep(na_pad, length_diff), v)
    } else {
      v
    }
  }
  
  purrr::map(
    seq_along(v),
    function(x) {
      start <- max(0, x + start_offset)
      end <- max(0, x + end_offset)
      out <- v[start:end]
      ifelse(is.na(out), na_pad, out)
    } %>%
      prepad_mask()
  )
}

compute_mack_ultimate <- function(df, claim_ids) {
  # Make adjustment to use report year
  df <- df %>%
    mutate(
      year = year - (report_year - AY),
      AY = report_year,
    ) %>% 
    # These records account for 0 paid losses
    filter(year >= 0)
  
  triangle_data <- df %>%
    group_by(AY, year) %>%
    summarize(paid = sum(Pay)) %>%
    mutate(cumulative_paid = cumsum(paid))
  
  atas <- triangle_data %>%
    arrange(year) %>%
    group_by(year) %>%
    summarize(
      year_sum = sum(cumulative_paid),
      year_sum_less_latest = cumulative_paid %>%
        head(-1) %>%
        sum()
    ) %>%
    mutate(ata = year_sum / lag(year_sum_less_latest))
  
  ldfs <- atas %>%
    mutate(
      ldf = ata %>%
        rev() %>%
        cumprod() %>%
        rev()
    )
  
  df %>%
    inner_join(claim_ids, by = c("ClNr")) %>%
    group_by(AY, year) %>%
    summarize(paid = sum(Pay)) %>%
    mutate(cumulative_paid = cumsum(paid)) %>%
    group_by(AY) %>%
    summarize(latest_year = max(year), latest_paid = last(cumulative_paid)) %>%
    left_join(
      ldfs %>%
        mutate(ldf = lead(ldf)),
      by = c(latest_year = "year")
    ) %>%
    mutate(ultimate = latest_paid * ldf) %>%
    summarize(total_ultimate = sum(ultimate)) %>%
    pull(total_ultimate)
}

prep_keras_data <- function(x, mask = 99999) {
  ind_lags <- x$claim_open_indicator_lags %>%
    simplify2array(higher = FALSE) %>%
    t()
  
  ind_lags_flipped <- apply(ind_lags, 1, function(v) {
    masked <- v[v == mask]
    nonmasked <- v[v != mask]
    c(masked, ifelse(nonmasked == 1, 0, 1))
  }) %>% t()
  
  claim_open_indicator_lags <- cbind(ind_lags, ind_lags_flipped) %>%
    array(dim = c(nrow(.), ncol(.) / 2, 2))
  
  list(
    x = list(
      paid_loss_lags_ = x$paid_loss_lags %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1)),
      recovery_lags_ = x$recovery_lags %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1)),
      claim_open_indicator_lags_ = claim_open_indicator_lags,
      scaled_dev_year_ = x$scaled_dev_year,
      lob_ = x$lob,
      claim_code_ = x$claim_code,
      age_ = x$age,
      injured_part_ = x$injured_part
    ),
    y = list(
      paid_loss_target_ = x$paid_loss_target %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1)),
      recovery_target_ = x$recovery_target %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1))
    )
  )
}

mutate_sequences <- function(data, recipe, timesteps, training = TRUE) {
  # Single thread
  #
  # output <- data %>%
  #   bake(recipe, .) %>%
  #   group_by(ClNr) %>%
  #   mutate(
  #     claim_open_indicator_lags = make_series(claim_open_indicator, -timesteps, -1),
  #     paid_loss_lags = make_series(paid_loss, -timesteps, -1),
  #     recovery_lags = make_series(recovery, -timesteps, -1)
  #   )
  
  # Parallel
  output <- data %>%
    bake(recipe, .) %>% 
    split(.$ClNr) %>% 
    parallel::mclapply(function(x) dplyr::mutate(x,
                                                 claim_open_indicator_lags = make_series(claim_open_indicator, -timesteps, -1),
                                                 paid_loss_lags = make_series(paid_loss, -timesteps, -1),
                                                 recovery_lags = make_series(recovery, -timesteps, -1)
    )) %>% 
    dplyr::bind_rows()
  
  if (training) {
    # output <- output %>%
    #   mutate(
    #     paid_loss_target = make_series(paid_loss_original, 0, timesteps - 1),
    #     recovery_target = make_series(recovery_original, 0, timesteps - 1)
    #   )
    
    output <- output %>% 
      split(.$ClNr) %>%
      parallel::mclapply(function(x) dplyr::mutate(x, 
                                                   paid_loss_target = make_series(paid_loss_original, 0, timesteps - 1),
                                                   recovery_target = make_series(recovery_original, 0, timesteps - 1)
      )) %>% 
      dplyr::bind_rows()
  }
  
  output
}

prep_datasets <- function(simulated_cashflows, n = 0, timesteps = 11, record_year_cutoff = 2005) {
  claim_ids <- simulated_cashflows %>%
    distinct(ClNr)
  
  if (n > 0) {
    claim_ids <- sample_n(claim_ids, !!n)
  }
  
  cashflow_history <- simulated_cashflows %>%
    inner_join(claim_ids, by = "ClNr")
  
  training_data <- cashflow_history %>%
    filter(record_year <= !!record_year_cutoff)
  
  rec <- recipe(training_data, ~.) %>%
    step_integer(lob, claim_code, injured_part, zero_based = TRUE) %>%
    step_center(age, paid_loss, recovery) %>%
    step_scale(age, paid_loss, recovery) %>%
    step_mutate(scaled_dev_year = year / 11) %>%
    prep(training_data, verbose = TRUE, retain = FALSE, strings_as_factors = FALSE)
  
  #' Capture mean/sd for paids so we can recover after prediction
  mean_paid <- rec$step[[2]]$means[["paid_loss"]]
  sd_paid <- rec$steps[[3]]$sds[["paid_loss"]]
  
  training_data <- training_data %>%
    mutate_sequences(rec, timesteps)
  
  dev_year_zero_records <- training_data %>%
    filter(year == 0)
  
  training_data <- training_data %>%
    filter(year > 0)
  
  list(
    training_data = training_data,
    dev_year_zero_records = dev_year_zero_records,
    cashflow_history = cashflow_history,
    mean_paid = mean_paid,
    sd_paid = sd_paid
  )
}

masked_negloglik <- function(mask_value) {
  function(y_true, y_pred) {
    keep_value <- k_cast(k_not_equal(
      tf$squeeze(y_true),
      mask_value), k_floatx())
    
    logprob <- y_pred$distribution$log_prob(
      tf$squeeze(y_true)
    )
    -k_sum(keep_value * logprob, axis = 2)
  }
}

cust_loss <- function(x, rv_x) masked_negloglik(99999)(x, rv_x)
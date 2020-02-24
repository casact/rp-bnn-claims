# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

timesteps <- 11

simulated_cashflows <- readr::read_delim("data/Simulated.Cashflow.txt", delim = ";", col_types = "dccdddcddddddddddddddddddddddddd")
record_year_cutoff <- min(simulated_cashflows$AY) + timesteps

#' Initial feature engineering
simulated_cashflows <- simulated_cashflows %>%
  gather(key, value, Pay00:Open11) %>%
  arrange(ClNr) %>%
  separate(key, into = c("variable", "year"), sep = "(?<=[a-z])(?=[0-9])") %>%
  mutate(year = as.integer(year)) %>%
  mutate(
    # Report year denotes when the claim was reported
    report_year = AY + RepDel,
    # Calendar year is the accounting year of the transaction
    calendar_year = AY + year,
    # Record year is the year in which the data becomes available,
    #  it must be after the claim was reported
    record_year = pmax(report_year, calendar_year)
  ) %>%
  # Keep only claims reported as of cutoff
  filter(report_year <= !!record_year_cutoff) %>%
  spread(variable, value) %>%
  mutate(
    paid_loss = Pay,
    lob = LoB,
    claim_code = cc,
    injured_part = inj_part,
    claim_open_indicator = Open
  ) %>%
  group_by(ClNr) %>%
  mutate(cumulative_paid_loss = cumsum(paid_loss)) %>%
  ungroup() %>%
  mutate(
    cumulative_paid_loss_original = cumulative_paid_loss,
    recovery = ifelse(paid_loss < 0, -paid_loss, 0),
    paid_loss = ifelse(paid_loss >= 0, paid_loss, 0),
    paid_loss_original = paid_loss,
    recovery_original = recovery,
  )

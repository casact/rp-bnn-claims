# bnn-claims

<!-- badges: start -->
[![Travis build status](https://travis-ci.org/kasaai/bnn-claims.svg?branch=master)](https://travis-ci.org/kasaai/bnn-claims)
<!-- badges: end -->

Repository for the paper "Individual Claims Forecasting with Bayesian Mixture Density Networks." This work is supported by the Casualty Actuarial Society.

## Data

The data files used in the code are part of the [release](https://github.com/kasaai/bnn-claims/releases/tag/v0.0.1). They can be downloaded after cloning the repo by running

```r
# remotes::install_github("ropensci/piggyback")
piggyback::pb_download()
```

The raw data file `Simulated.Cashflow.txt` was created using the simulation machine available at [https://people.math.ethz.ch/~wueth/simulation.html](https://people.math.ethz.ch/~wueth/simulation.html).

## Dependencies

(GPU only) To install the necessary dependencies, install the latest dev version of renv using

```r
remotes::install_github("rstudio/renv")
```

then run `renv::init()` followed by `renv::restore()`.

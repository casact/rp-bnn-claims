reticulate::use_virtualenv("~/tf2", required = TRUE)
library(keras)
library(tensorflow)
library(tfprobability)
library(tidyverse)
library(recipes)
library(tfdatasets)

source("analysis/utils.R")
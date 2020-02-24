# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

renv::use_python()
library(keras)
library(tensorflow)
library(tfprobability)
library(tidyverse)
library(recipes)
library(tfdatasets)

source("analysis/utils.R")

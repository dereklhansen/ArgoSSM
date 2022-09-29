
# A suite of unit tests for the penalized Covariance estimation
library(testthat)
library(Matrix)
library(ncdf4)
library(GPvecchia)
library(fields)
library(tidyverse)
source("code/src_R/cov_est_funs.R")

# first we test if creating a grid and profile_data gives the same number of rows
grid <- create_grid(T, long_range = c(-60, 20), lat_range = c(-80, -60))
test_that("grid_stable_nrow", {
  expect_equal(nrow(grid), 4320)
})

profile_data <- load_profile_data(145, 155)
test_that("profile_data_stable_nrow", {
  expect_equal(nrow(profile_data), 56266)
})

# we try out local regression at random gridpoints to make sure consistent
test_that("local_regression_stable", {
  set.seed(40)
  r_indexes <- sample(1:nrow(grid), 10)
  mean_funs <- lapply(r_indexes, mean_est,
    h_space = 300, profile_data,
    profile_data
  )
  temp_NW <- sapply(mean_funs, function(x) as.double(x[[2]][[1]]))
  temp_NW <- sapply(temp_NW, function(x) {
    if (length(x) == 0) {
      NA
    } else {
      x
    }
  })
  vec_to_compare <- c(
    0.1870, 0.6939, NA, 0.5132, -1.5058, 0.2791,
    -0.2577, NA, 0.7402, -0.4164
  )
  expect_equal(is.na(temp_NW), is.na(vec_to_compare),
    failure_message = "local regression different gridpoints estimated"
  )
  expect(mean(abs(vec_to_compare - temp_NW), na.rm = T) < .001,
    failure_message = "local regression not stable"
  )
})

profile_data$sample <- "linterp"
set.seed(50)
r_prof <- sample(1:nrow(profile_data), 1000)
profile_data_use <- profile_data %>% filter(1:n() %in% r_prof)
test_that("mean_est_can_run", {
  plot_df <- mean_est_grid(
    profile_data = profile_data_use, grid = grid, h_space = 500,
    mc.cores = NULL
  )
})

plot_df <- mean_est_grid(
  profile_data = profile_data_use, grid = grid, h_space = 500,
  mc.cores = NULL
)

test_that("cov_est_can_run_same_params", {
  set.seed(50)
  sp_model <- fit_spatial_model("linterp",
    plot_df_use = plot_df,
    profile_data_use = profile_data_use,
    m_seq = c(10, 30, 50),
    start_params = c(0.6643, 0.2, 35, 0.8, 0.1)
  )
  params <- sp_model[[1]]$covparms
  params_old <- c(6.606168e-01, 4.091778e-02, 3.541688e+02, 3.911775e-01, 1.158288e-03)
  expect_equal(params, params_old, tolerance = .0001)
})
set.seed(50)
sp_model <- fit_spatial_model("linterp",
  plot_df_use = plot_df,
  profile_data_use = profile_data_use,
  m_seq = c(10, 30, 50),
  start_params = c(0.6643, 0.2, 35, 0.8, 0.1)
)

test_that("pred_can_run", {
  preds <- predict_fun("linterp", list("linterp" = sp_model),
    date = "2017-08-01",
    plot_df = plot_df, nn = 15
  )
  expect_equal(preds$mu.pred[1:6], c(
    -0.2100227, -0.2147493, -0.2154212,
    -0.2118444, -0.2175530, -0.2167513
  ), tolerance = .0001)
})

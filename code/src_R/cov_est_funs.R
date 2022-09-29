
create_grid <- function(grid_size, long_range, lat_range) {
  sose_grid <- nc_open("temp/data/grid.nc")
  sose_grid_x <- ncvar_get(sose_grid, "XC")[, 1]
  sose_grid_y <- ncvar_get(sose_grid, "YC")[1, ]
  depth <- ncvar_get(sose_grid, "Depth")

  sose_grid_x <- ifelse(sose_grid_x >= 180, sose_grid_x - 360, sose_grid_x)

  if (!(grid_size %in% c(1, 2, 3, 6))) {
    stop("provide 1, 2, 3, or 6 to grid_size")
  }
  depth <- depth[
    seq(1, length(sose_grid_x), by = 6 / grid_size),
    seq(1, length(sose_grid_y), by = 6 / grid_size)
  ]
  sose_grid_x <- sose_grid_x[seq(1, length(sose_grid_x), by = 6 / grid_size)]
  sose_grid_y <- sose_grid_y[seq(1, length(sose_grid_y), by = 6 / grid_size)]

  depth <- as.double(depth)
  # save grid sizes
  width <- 1
  height <- sose_grid_y[2:length(sose_grid_y)] - sose_grid_y[1:(length(sose_grid_y) - 1)]
  height <- data.frame("lat" = sose_grid_y, height = c(height[1], height))

  # make grid
  grid <- expand.grid(
    "long" = sose_grid_x,
    "lat" = sose_grid_y
  )
  grid$width <- 1
  grid <- merge(grid, height, by = "lat")
  grid <- grid[, c("long", "lat", "width", "height")]
  grid <- cbind(grid, depth)
  grid <- grid[grid$long <= long_range[2] & grid$long >= long_range[1] &
    grid$lat < lat_range[2] & grid$lat > lat_range[1], ]

  return(grid)
}


load_profile_data <- function(p_min, p_max) {
  # remove bad pressure/temperature/salinity measurements, and focus on one pressure level
  profile_data <- readRDS("temp/data/prof_data_06_2021.RDS")
  profile_data <- profile_data %>%
    mutate(profile = paste(float, cycle, sep = "_")) %>%
    filter(
      pressure_qc %in% c(1, 2) & temperature_qc %in% c(1, 2) &
        salinity_qc %in% c(1, 2) &
        pressure > p_min & pressure < p_max,
      !is.na(temperature), !is.na(salinity)
    ) %>%
    group_by(float, cycle) %>%
    filter(!duplicated(pressure)) %>%
    ungroup()
}

load_argossm_results <- function(n_samples) {
  float_tbl <- read_rds("output/float_tbl.rds") %>%
    group_by(float) %>%
    mutate(t_new = 1:n())
  unique_floats <- unique(float_tbl$float)
  ## Make tbls from the samples
  read_results_h5 <- function(float, model = "smc2", file = "output/floats_results.h5", n_samples = 5) {
    paths <- h5read(file, sprintf("%s/%s/states", float, model))
    plyr::adply(paths[, 1:n_samples, ], .margins = 2, .id = "sample", function(x) {
      tibble(
        float = float, long = x[1, ], lat = x[2, ],
        v_long = x[3, ], v_lat = x[4, ]
      ) %>%
        mutate(t_new = 1:n())
    }) %>%
      as_tibble() %>%
      dplyr::select(float, sample, t_new, long, lat, v_long, v_lat)
  }
  results <- purrr::map(unique_floats, read_results_h5, n_samples = n_samples) %>%
    bind_rows()
  results_with_data <- inner_join(float_tbl, results, by = c("float", "t_new"))
  estimated_locs <- results_with_data %>%
    rename(long_est = long.y, lat_est = lat.y) %>%
    dplyr::select(date, float, sample, long_est, lat_est, v_long, v_lat) %>%
    mutate(sample = as.character(sample))

  estimated_locs_mean <- estimated_locs %>%
    group_by(date, float) %>%
    summarize(
      long_est = mean(long_est), lat_est = mean(lat_est),
      v_long = mean(v_long), v_lat = mean(v_lat)
    ) %>%
    mutate(sample = "mean") %>%
    select(date, float, sample, long_est, lat_est, v_long, v_lat) %>%
    ungroup()
  # talk linear interpolation
  estimated_locs_linterp <- inner_join(estimated_locs_mean, profile_data, by = c("float", "date")) %>%
    mutate(sample = "linterp", long_est = long, lat_est = lat) %>%
    arrange(sample, float, day) %>%
    group_by(sample, float) %>%
    mutate(
      v_long = (lead(long) - (long)) / (lead(day) - (day)),
      v_lat = (lead(lat) - (lat)) / (lead(day) - (day))
    ) %>%
    select(date, float, sample, long_est, lat_est, v_long, v_lat)

  estimated_locs_all <- bind_rows(
    estimated_locs, estimated_locs_mean,
    estimated_locs_linterp
  ) %>%
    mutate(sample = factor(sample, levels = c("mean", "linterp", 1:n_samples))) %>%
    mutate(float = as.factor(float))
}

mean_est_grid <- function(profile_data, grid, h_space = 200,
                          mc.cores = NULL, vars = c("temperature", "salinity")) {
  print(paste0("Number of grid points to estimate at: ", nrow(grid)))

  if (is.null(mc.cores)) {
    mean_funs <- lapply(1:nrow(grid), mean_est,
      h_space = h_space, profile_data = profile_data,
      profile_data_unique = profile_data,
      vars = vars
    )
  } else {
    mean_funs <- mclapply(1:nrow(grid), mean_est,
      h_space = h_space, profile_data = profile_data,
      profile_data_unique = profile_data,
      mc.cores = mc.cores, mc.preschedule = T,
      vars = vars
    )
  }

  # put results in nice form, remove gridpoints with no estimates
  have_results <- sapply(1:length(mean_funs), function(x) {
    if (is.null(mean_funs[[x]][[2]])) {
      return(F)
    } else if (is.na(as.double(mean_funs[[x]][[2]][[1]][1, 1]))) {
      return(F)
    } else {
      T
    }
  })
  have_ll <- sapply(mean_funs[have_results], function(x) {
    length(as.double(x[[1]][[1]][, 1])) > 0
  })


  closest_dist <- sapply(mean_funs[have_results], function(x) x[[3]][3])
  n_prof <- sapply(mean_funs[have_results], function(x) x[[3]][4])
  plot_df <- data.frame(
    grid[have_results, ],
    closest_dist = closest_dist, n_prof = n_prof,
    sample = rep(profile_data$sample[1], length(n_prof))
  )
  for (variable in vars) {
    plot_df[[paste0(variable, "_NW")]] <-
      sapply(mean_funs[have_results], function(x) as.double(x[[2]][[which(vars == variable)]]))
    plot_df[[paste0(variable, "_ll")]] <-
      t(sapply(1:length(mean_funs[have_results]), function(x) {
        if (have_ll[x]) {
          as.double(mean_funs[have_results][[x]][[1]][[1]])
        } else {
          c(plot_df[[paste0(variable, "_NW")]][[x]], 0, 0, 0, 0)
        }
      }))
  }
  plot_df
}

mean_est <- function(index, h_space, profile_data, profile_data_unique, vars) {
  a <- proc.time()
  # print(index)
  # for each grid point
  long <- grid[index, 1]
  lat <- grid[index, 2]

  # find distances to grid point
  dist <- rdist.earth.vec(matrix(nrow = 1, c(long, lat)),
    profile_data_unique[, c("long", "lat")],
    miles = F
  )
  # set epanich... kernel
  wt <- ifelse(dist / h_space <= 1, 3 / 4 / h_space * (1 - (dist / h_space)^2), 0)
  closest_prof <- min(dist)

  # only look at profiles with positive weight
  data_grid <- profile_data %>%
    filter(profile %in% profile_data_unique$profile[wt > 0]) %>%
    left_join(data.frame(profile = profile_data_unique$profile[wt > 0], wt = wt[wt > 0]),
      by = "profile"
    )
  # if not much data nearby, skip gridpoint
  if (closest_prof > h_space) {
    b <- proc.time()
    if (index %% 15000 == 0) {
      print(paste0("Progress: ", index))
    }
    return(list(
      NULL, NULL,
      c(index, (b - a)[3], Inf, 0, 0)
    ))
  }

  # build covariates for local regression - space and day of year
  dist_EW <- rdist.earth.vec(matrix(nrow = 1, c(long, lat)),
    cbind(data_grid[, "long"], lat),
    miles = F
  )
  dist_NS <- rdist.earth.vec(matrix(nrow = 1, c(long, lat)),
    cbind(long, data_grid[, "lat"]),
    miles = F
  )
  dist_EW <- ifelse(data_grid[, "long"] > long, dist_EW, -dist_EW)
  dist_NS <- ifelse(data_grid[, "lat"] > lat, dist_NS, -dist_NS)

  day <- data_grid$day %% 365.25
  constants <- 2 * pi * (1:6) / 365.25

  # Roemmich and Gilson type covariates (local regression in space, fourier basis for time)
  covariates <- cbind(
    1, sin(constants[1] * day), cos(constants[1] * day),
    sin(constants[2] * day), cos(constants[2] * day)
  )
  W <- Diagonal(x = data_grid$wt, n = nrow(data_grid))
  U <- t(covariates) %*% W %*%
    covariates
  V <- t(covariates) %*% W
  # do local regression with the extended covariates
  # guard against a ill-conditioned matrix
  ll_estimators <- tryCatch(
    expr = {
      U_inv <- Matrix::solve(U)
      mean_list <- list()
      for (variable in vars) {
        ll_mean_est <- U_inv %*% V %*% data_grid[[variable]]
        ll_sigma2 <- mean((data_grid[[variable]] - covariates %*% ll_mean_est)^2)
        W2 <- Diagonal(x = data_grid$wt^2 * ll_sigma2, n = nrow(data_grid))
        ll_est_var <- U_inv %*% t(covariates) %*% W2 %*% covariates %*% U_inv
        mean_list[[which(vars == variable)]] <- ll_mean_est
        mean_list[[which(vars == variable) + length(vars)]] <- ll_sigma2
        mean_list[[which(vars == variable) + 2 * length(vars)]] <- ll_est_var
      }
      mean_list
    },
    error = function(x) {
      replicate(length(vars) * 3, NULL, simplify = F)
    }
  )


  # do same for Nadaraya-watson type estimator - more simple
  covariates <- cbind(covariates[, 1])
  U <- t(covariates) %*% W %*%
    covariates
  V <- t(covariates) %*% W
  U_inv <- Matrix::solve(U)
  nw_estimators <- list()
  for (variable in vars) {
    nw_mean_est <- U_inv %*% V %*% data_grid[[variable]]
    nw_sigma2 <- mean((data_grid[[variable]] - covariates %*% nw_mean_est)^2)
    W2 <- Diagonal(x = data_grid$wt^2 * nw_sigma2, n = nrow(data_grid))
    nw_est_var <- U_inv %*% t(covariates) %*% W2 %*% covariates %*% U_inv
    nw_estimators[[which(vars == variable)]] <- nw_mean_est
    nw_estimators[[which(vars == variable) + length(vars)]] <- nw_sigma2
    nw_estimators[[which(vars == variable) + 2 * length(vars)]] <- nw_est_var
  }

  b <- proc.time()
  n_prof <- length(unique(data_grid$profile))

  if (index %% 15000 == 0) {
    print(paste0("Progress: ", index))
  }
  return(list(
    ll_estimators, nw_estimators,
    c(index, (b - a)[3], closest_prof, n_prof, nrow(data_grid))
  ))
}

subtract_mean <- function(df, samp, mean_field) {
  df_sample <- df %>%
    filter(sample == samp) %>%
    mutate(day_of_year = day %% 365.25)
  constants <- 2 * pi * (1:6) / 365.25
  mean_field_sample <- mean_field %>%
    filter(sample == samp)

  # find nearest grid point
  mean_est_prof <- matrix(nrow = nrow(df_sample), ncol = 2)
  long_u <- unique(mean_field_sample$long)
  lat_u <- unique(mean_field_sample$lat)
  for (i in 1:nrow(df_sample)) {
    long <- df_sample$long[i]
    lat <- df_sample$lat[i]
    long_grid <- long_u[which.min(abs(long_u - long))]
    lat_grid <- lat_u[which.min(abs(lat_u - lat))]
    day_of_year <- df_sample$day_of_year[i]
    mean_est_prof[i, ] <- mean_field_sample %>%
      filter(lat == lat_grid, long == long_grid) %>%
      mutate(
        mean_temp = temp_ll.1 +
          temp_ll.2 * sin(constants[1] * day_of_year) +
          temp_ll.3 * cos(constants[1] * day_of_year) +
          temp_ll.2 * sin(constants[2] * day_of_year) +
          temp_ll.3 * cos(constants[2] * day_of_year),
        mean_psal = psal_ll.1 +
          psal_ll.2 * sin(constants[1] * day_of_year) +
          psal_ll.3 * cos(constants[1] * day_of_year) +
          psal_ll.2 * sin(constants[2] * day_of_year) +
          psal_ll.3 * cos(constants[2] * day_of_year)
      ) %>%
      select(mean_temp, mean_psal) %>%
      as.double()
  }

  df_sample <- df_sample %>%
    mutate(
      temp_NW = mean_est_prof[, 1],
      temp_m0 = temperature - temp_NW,
      psal_NW = mean_est_prof[, 2],
      psal_m0 = salinity - psal_NW,
      profile_unique = paste(float, cycle, sep = "_")
    )
  df_sample
}

subtract_mean_velocity <- function(df, samp, mean_field) {
  df_sample <- df %>%
    filter(sample == samp) %>%
    mutate(day_of_year = day %% 365.25)
  constants <- 2 * pi * (1:6) / 365.25
  mean_field_sample <- mean_field %>%
    filter(sample == samp)

  # find nearest grid point
  mean_est_prof <- matrix(nrow = nrow(df_sample), ncol = 2)
  long_u <- unique(mean_field_sample$long)
  lat_u <- unique(mean_field_sample$lat)
  for (i in 1:nrow(df_sample)) {
    long <- df_sample$long[i]
    lat <- df_sample$lat[i]
    long_grid <- long_u[which.min(abs(long_u - long))]
    lat_grid <- lat_u[which.min(abs(lat_u - lat))]
    day_of_year <- df_sample$day_of_year[i]
    mean_est_prof[i, ] <- mean_field_sample %>%
      filter(lat == lat_grid, long == long_grid) %>%
      mutate(
        mean_v_long = v_long_NW,
        mean_v_lat = v_lat_NW
      ) %>%
      select(mean_v_long, mean_v_lat) %>%
      as.double()
  }
  df_sample$v_long_NW <- mean_est_prof[, 1]
  df_sample$v_lat_NW <- mean_est_prof[, 2]
  df_sample <- df_sample %>%
    mutate(
      v_long_m0 = v_long - v_long_NW,
      v_lat_m0 = v_lat - v_long_NW,
      profile_unique = paste(float, cycle, sep = "_")
    )
  df_sample
}

fit_spatial_model_velocity <- function(samp, plot_df_use, profile_data_use,
                                       silent = T, m_seq = c(10, 30, 50),
                                       start_params = c(.002, 0.2, 0.8, 0.1),
                                       variable) {
  df_sample <- subtract_mean_velocity(profile_data_use,
    samp = samp, mean_field = plot_df_use
  )
  df_sample_rem <- df_sample %>%
    filter(!duplicated(long), !duplicated(lat))

  sp_mod <- GpGp::fit_model(
    y = df_sample_rem[[variable]],
    locs = as.matrix(df_sample_rem[, c("long", "lat")]),
    covfun_name = "matern_sphere", m_seq = m_seq,
    start_parms = start_params, silent = silent
  )
  print(samp)
  list(sp_mod, df_sample_rem)
}


fit_spatial_model <- function(samp, plot_df_use, profile_data_use,
                              silent = T, m_seq = c(10, 30, 50),
                              start_params = c(0.6643, 0.2, 35, 0.8, 0.1),
                              variable) {
  df_sample <- subtract_mean(profile_data_use,
    samp = samp, mean_field = plot_df_use
  )
  df_sample_rem <- df_sample %>%
    filter(!duplicated(long), !duplicated(lat))

  sp_mod <- GpGp::fit_model(
    y = df_sample_rem[[variable]],
    locs = as.matrix(df_sample_rem[, c("long", "lat", "day")]),
    covfun_name = "matern_spheretime", m_seq = m_seq,
    start_parms = start_params, silent = silent
  )
  print(samp)
  list(sp_mod, df_sample_rem)
}



predict_fun <- function(samp, models, date, plot_df, nn) {
  print(samp)
  day_to_pred <- as.numeric(julian(as.Date(date, format = "%Y-%m-%d"),
    origin = as.Date("1950-01-01")
  ))
  locs_pred <- as.matrix(bind_cols(plot_df[plot_df$sample == samp, c("long", "lat")],
    day = day_to_pred
  ))
  sp_mod <- models[[samp]][[1]]
  locs_obs <- sp_mod$locs
  scale_space <- sp_mod$covparms[2]
  scale_time <- sp_mod$covparms[3]
  locs_obs_new <- cbind(
    locs_obs[, 1] / (scale_space * 6371) * 111 * cos(locs_obs[, 2] * pi / 180),
    locs_obs[, 2] / (scale_space * 6371) * 111,
    locs_obs[, 3] / scale_time
  )
  locs_pred_new <- cbind(
    locs_pred[, 1] / (scale_space * 6371) * 111 * cos(locs_pred[, 2] * pi / 180),
    locs_pred[, 2] / (scale_space * 6371) * 111,
    locs_pred[, 3] / scale_time
  )
  vs <- vecchia_specify(locs = locs_obs_new, m = nn, locs.pred = locs_pred_new)
  vecchia_prediction(sp_mod$y,
    vecchia.approx = vs,
    covparms = c(sp_mod$covparms[1], 1, sp_mod$covparms[4]),
    nuggets = sp_mod$covparms[5] * sp_mod$covparms[1]
  )
}

predict_fun_velocity <- function(samp, models, plot_df, nn) {
  print(samp)
  locs_pred <- as.matrix(plot_df[plot_df$sample == samp, c("long", "lat")])
  sp_mod <- models[[samp]][[1]]
  locs_obs <- sp_mod$locs
  scale_space <- sp_mod$covparms[2]
  locs_obs_new <- cbind(
    locs_obs[, 1] / (scale_space * 6371) * 111 * cos(locs_obs[, 2] * pi / 180),
    locs_obs[, 2] / (scale_space * 6371) * 111
  )
  locs_pred_new <- cbind(
    locs_pred[, 1] / (scale_space * 6371) * 111 * cos(locs_pred[, 2] * pi / 180),
    locs_pred[, 2] / (scale_space * 6371) * 111
  )
  remove_index <- which(!duplicated(locs_obs_new))
  vs <- vecchia_specify(locs = locs_obs_new[remove_index, ], m = nn, locs.pred = locs_pred_new)
  vecchia_prediction(sp_mod$y[remove_index],
    vecchia.approx = vs,
    covparms = c(sp_mod$covparms[1], 1, sp_mod$covparms[3]),
    nuggets = sp_mod$covparms[4]
  )
}


get_streamfun <- function(plot_df, samp) {
  vels_use <- filter(plot_df, n_prof > 55) %>%
    arrange(long, lat)
  vels_use_test <- vels_use %>%
    filter(sample == samp, depth > 2000) %>%
    # filter(sample == samp) %>%
    select(c("long", "lat", "cond_exp_u", "cond_exp_v")) %>%
    mutate(id = 1:length(long)) %>%
    arrange(long, lat)
  lat_vals_vec <- unique(vels_use_test$lat)[order(unique(vels_use_test$lat))]
  lat_id_df <- data.frame(
    lat_id = 1:length(unique(vels_use_test$lat)),
    lat = lat_vals_vec
  )

  long_vals <- vels_use_test %>%
    pivot_wider(values_from = c("long"), id_cols = c("lat"), names_from = c("long")) %>%
    arrange(lat)
  lat_vals <- vels_use_test %>%
    left_join(lat_id_df, by = "lat") %>%
    pivot_wider(values_from = c("lat"), id_cols = c("lat_id"), names_from = c("long")) %>%
    arrange(lat_id)
  v_long_vals <- vels_use_test %>%
    pivot_wider(values_from = c("cond_exp_u"), id_cols = c("lat"), names_from = c("long")) %>%
    arrange(lat)
  v_lat_vals <- vels_use_test %>%
    pivot_wider(values_from = c("cond_exp_v"), id_cols = c("lat"), names_from = c("long")) %>%
    arrange(lat)

  x <- as.double(colnames(long_vals[, -1]))
  y <- unlist(long_vals[, 1])
  long_vals <- t(as.matrix(long_vals[, -1]))
  lat_vals <- t(as.matrix(lat_vals[, -1]))
  v_long_vals <- t(as.matrix(v_long_vals[, -1]))
  v_lat_vals <- t(as.matrix(v_lat_vals[, -1]))

  lag_x <- x - lag(x)
  lag_x[1] <- x[2] - x[1]
  lag_x <- matrix(lag_x, nrow(long_vals), ncol(long_vals))
  lag_y <- y - lag(y)
  lag_y[1] <- y[2] - y[1]
  lag_y <- matrix(lag_y, nrow(long_vals), ncol(long_vals), byrow = T)
  dist_long_vals <- matrix(
    nrow = nrow(lat_vals),
    ncol = ncol(lat_vals),
    rdist.earth.vec(cbind(as.double(long_vals), as.double(lat_vals)),
      cbind(as.double(long_vals + lag_x), as.double(lat_vals)),
      miles = F
    )
  ) * 1000
  dist_lat_vals <- matrix(
    nrow = nrow(lat_vals),
    ncol = ncol(lat_vals),
    rdist.earth.vec(cbind(as.double(long_vals), as.double(lat_vals)),
      cbind(as.double(long_vals), as.double(lat_vals + lag_y)),
      miles = F
    )
  ) * 1000
  N <- nrow(lat_vals)
  M <- ncol(lat_vals)
  indexes <- 1:(M * N)
  j_use <- cbind(indexes - 1, indexes + 1, indexes - N, indexes + N)
  j_use <- cbind(indexes, indexes + 1, indexes, indexes + N)
  mat_prelim <- matrix(indexes, nrow = length(indexes), ncol = 4)
  j_use[j_use < 1 | j_use > M * N | (mat_prelim %% N == 1 & j_use %% N == 0) |
    (mat_prelim %% N == 0 & j_use %% N == 1)] <- NA
  j_use_NS <- as.vector(t(j_use[, 3:4])) # switching latitude for fixed longitude
  j_use_EW <- as.vector(t(j_use[, 1:2])) # switching longitude for fixed latitude
  i_use_NS <- rep(1:length(indexes), each = 2)[!is.na(j_use_NS)]
  i_use_EW <- rep(1:length(indexes), each = 2)[!is.na(j_use_EW)]
  x_vals_NS <- (rep(c(-1, 1) / 2, times = length(indexes)) / rep(as.double(dist_lat_vals), each = 2))[!is.na(j_use_NS)]
  x_vals_EW <- (rep(c(-1, 1) / 2, times = length(indexes)) / rep(as.double(dist_long_vals), each = 2))[!is.na(j_use_EW)]
  j_use_NS <- j_use_NS[!is.na(j_use_NS)]
  j_use_EW <- j_use_EW[!is.na(j_use_EW)]
  A_NS <- sparseMatrix(i = i_use_NS, j = j_use_NS, x = x_vals_NS)
  A_EW <- sparseMatrix(i = i_use_EW, j = j_use_EW, x = x_vals_EW)
  A <- rbind(A_EW, A_NS)
  Y <- c(as.double(-v_lat_vals), as.double(v_long_vals))
  good <- which(!is.na(Y))
  good2 <- which(!is.na(v_long_vals))

  sol_vec <- solve(
    t(A[good, good2]) %*% A[good, good2],
    t(A[good, good2]) %*% Y[good]
  )
  sol_vec_long <- rep(NA, nrow(lat_vals) * ncol(lat_vals))
  sol_vec_long[good2] <- sol_vec
  stream_vals <- matrix(
    nrow = nrow(lat_vals),
    ncol = ncol(lat_vals), sol_vec_long
  )

  stream_df <- data.frame(
    long = as.double(apply(long_vals, 1, mean, na.rm = T)),
    lat = rep(apply(lat_vals, 2, mean, na.rm = T), each = nrow(lat_vals)),
    stream = as.double(stream_vals)
  ) %>%
    left_join(filter(plot_df, sample == samp), by = c("long", "lat"))
}

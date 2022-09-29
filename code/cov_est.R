source("code/src_R/cov_est_funs.R")
pressure_val <- 150 # pressure level to look at
pressure_range <- 5 # range above/below level to accept measurements
h_space <- 250 # in km, how far to look for data
grid_size <- 2 # 1 is the plot for the presentations, 6 is original grid, 2 and 3 should also work
n_samples <- 20 # Number of samples to read from the estimation
nn_neigh_prediction <- 60
mc.cores <- 5
library(tidyverse)
library(fields)
library(Matrix)
library(parallel)
library(ncdf4)
library(rhdf5)
library(ggplot2)

# Step 1: create a grid on which to compute estimates
grid <- create_grid(grid_size, long_range = c(-60, 20), lat_range = c(-80, -60))


# load in temp/salinity data near a fixed depth in the ocean
profile_data <- load_profile_data(
  p_min = pressure_val - pressure_range,
  p_max = pressure_val + pressure_range
) %>%
  group_by(float, cycle) %>%
  summarise(
    temperature = mean(temperature), salinity = mean(salinity),
    float = float[1], cycle = cycle[1],
    lat = lat[1], long = long[1], pos_qc = pos_qc[1],
    day = day[1], date = date[1], date_time = date_time[1],
    profile = profile[1], .groups = "drop"
  )
estimated_locs_all <- load_argossm_results(n_samples)

# merge datas together
profile_data_list <- lapply(
  1:length(unique(estimated_locs_all$sample)),
  function(x) {
    profile_data_use <- profile_data %>%
      mutate(sample = unique(estimated_locs_all$sample)[x])
  }
)
profile_data_all <- bind_rows(profile_data_list)

profile_data <- left_join(profile_data_all, estimated_locs_all,
  by = c("float", "date", "sample")
) %>%
  mutate(
    long = ifelse(is.na(long_est), long, long_est),
    lat = ifelse(is.na(long_est), lat, lat_est)
  )
# where we have estimated locations, use it


ggplot(
  data = profile_data[profile_data$sample == "mean", ],
  aes(x = long, y = lat, color = !is.na(long_est))
) +
  geom_point(size = .1)


# Step 4: The mean estimation at each point in grid, using bandwidth h_space
# and estimated locations in profile_data
plot_df <- plyr::ddply(profile_data,
  .variables = "sample", mean_est_grid, grid = grid, h_space = h_space,
  mc.cores = mc.cores
) %>%
  as_tibble()

a <- Sys.time()
sp_mods_samples <- mclapply(c("mean", "linterp", 1:20),
  fit_spatial_model,
  plot_df_use = plot_df, profile_data_use = profile_data,
  variable = "temp_m0",
  mc.cores = mc.cores, mc.preschedule = F
)
names(sp_mods_samples) <- c("mean", "linterp", 1:20)
b <- Sys.time()
b - a

file_name <- paste("temp/spatial_models", pressure_val, h_space, "temp", "time.RData",
  sep = "_"
)
save(sp_mods_samples, plot_df, file = file_name)
# predictions
###
data_used <- lapply(sp_mods_samples, function(x) x[[2]])
library(dplyr)
library(GPvecchia)

date <- "2015-08-01"
preds <- mclapply(c("mean", "linterp", 1:20), predict_fun,
  models = sp_mods_samples,
  date = date, plot_df = plot_df, nn = nn_neigh_prediction,
  mc.cores = mc.cores, mc.preschedule = F
)

cond_exp <- unlist(lapply(preds, function(x) as.double(x$mu.pred)))
cond_var <- unlist(lapply(preds, function(x) as.double(x$var.pred)))

plot_df <- plot_df %>%
  mutate(cond_exp = cond_exp, cond_var = cond_var)

file_name <- paste0(
  "temp/spatial_pred_", pressure_val, "_",
  h_space, "_time_", date, ".RData"
)

# Plots for the mean

world_map <- ggplot() +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2
  ) +
  labs(
    title = "Temperature Predictions", fill = "°C",
    y = "Latitude", x = "Longitude"
  ) +
  coord_cartesian(xlim = c(-55, 20), ylim = c(-75, -58)) +
  facet_wrap(~sample)

theme_set(theme_bw())
ggplot() +
  scale_fill_viridis_c(limits = c(.18, 1.16)) +
  scale_color_viridis_c(limits = c(.18, 1.16)) +
  # scale_fill_viridis_b(limits = c(0, 1.2), breaks = c(.5, .6, .7, .8, 1),
  #                      values = scales::rescale(c(.5, .6, .7, .8, 1))) +
  geom_tile(
    data = filter(plot_df, !is.na(temp_NW), sample %in% 1:4),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = cond_sd, color = cond_sd
    )
  ) +
  geom_point(
    data = locations[locations$sample %in% 1:4, ] %>%
      filter(date > "2015-06-01", date < "2015-10-01"),
    aes(x = long, y = lat), color = "white", size = .4, shape = 17, stroke = .4,
  ) +
  # geom_polygon(
  #   data = map_data("world"), aes(x = long, y = lat, group = group),
  #   fill = "white", color = "black", size = .2,
  # ) +
  # coord_cartesian(xlim = c(-55, 20), ylim = c(-75, -58)) +
  coord_map(xlim = c(-55, 20), ylim = c(-75, -58)) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Temperature Conditional SD", fill = "SD",
    color = "SD"
  ) +
  facet_wrap(~sample, ncol = 2)
ggsave("temp/cond_sd_samples.png", width = 6, height = 4, scale = 2)


plot_df_sum <- plot_df %>%
  filter(sample %in% 1:20) %>%
  group_by(long, lat) %>%
  summarise(
    height = head(height, 1), width = head(width, 1),
    temp_var_cond_exp = var(cond_exp, na.rm = T),
    temp_mean_cond_var = mean(cond_var, na.rm = T),
    temp_mean_cond_exp = mean(cond_exp, na.rm = T),
  )

ggplot() +
  scale_fill_viridis_b(
    limits = c(0, .5), breaks = c(.001, .01, .1, .2, .3)
  ) +
  geom_tile(
    data = filter(plot_df_sum, !is.na(temp_var_cond_exp)),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = sqrt(temp_var_cond_exp)
    )
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_cartesian(xlim = c(-55, 20), ylim = c(-75, -58)) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Temperature Conditional SD", fill = "SD"
  )
ggplot() +
  scale_fill_viridis_c(limits = c(.18, 1.16)) +
  scale_color_viridis_c(limits = c(.18, 1.16)) +
  # scale_fill_viridis_b(limits = c(0, 1.2), breaks = c(.5, .6, .7, .8, 1),
  #                      values = scales::rescale(c(.5, .6, .7, .8, 1))) +
  geom_tile(
    data = filter(
      plot_df_sum, !is.na(temp_mean_cond_var),
      !is.na(temp_var_cond_exp)
    ),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = sqrt(temp_mean_cond_var), color = sqrt(temp_mean_cond_var)
    )
  ) +
  coord_map(xlim = c(-55, 20), ylim = c(-75, -58)) +
  # geom_polygon(
  #   data = map_data("world"), aes(x = long, y = lat, group = group),
  #   fill = "white", color = "black", size = .2,
  # ) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Averaged Temperature Conditional SD", fill = "SD", color = "SD"
  )
ggsave("temp/cond_sd_avg_samples.png", width = 6, height = 4, scale = 2)

ggplot() +
  scale_fill_viridis_c() +
  geom_tile(
    data = filter(
      plot_df_sum, !is.na(temp_mean_cond_var),
      !is.na(temp_var_cond_exp)
    ),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = sqrt(temp_mean_cond_var + temp_var_cond_exp)
    )
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_cartesian(xlim = c(-55, 20), ylim = c(-75, -58)) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Temperature Conditional SD", fill = "SD"
  )




ggplot() +
  scale_fill_viridis_b(
    limits = c(-1.8, 1), breaks = c(-1, -.5, 0, .125, .25, .375, .5, .8),
    values = scales::rescale(c(-1, -.5, 0, .125, .25, .375, .5, .8))
  ) +
  geom_tile(
    data = filter(plot_df_sum, !is.na(temp_mean_cond_exp)),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = (temp_mean_cond_exp)
    )
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_cartesian(xlim = c(-55, 20), ylim = c(-75, -58)) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Temperature Conditional Mean Average", fill = "SD"
  )



p <- world_map +
  scale_fill_viridis_b(limits = c(-1.8, 1), breaks = c(-1.8, -1.5, -1, -.5, 0, .5)) +
  geom_tile(
    data = filter(plot_df, !is.na(temp_NW), sample %in% 1:3),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = temp_NW
    )
  ) +
  labs(title = "Temperature Mean, 150 dbar", fill = "°C") +
  facet_wrap(~sample)
ggsave(
  paste0(
    "temp/mean_est_LR_",
    pressure_val,
    "_var_",
    h_space,
    "_km_", type,
    ".png"
  ),
  p,
  width = 8,
  height = 3
)
save(plot_df, data_used, date, nn_neigh_prediction, file = file_name)

################## salinity

a <- Sys.time()
sp_mods_samples <- mclapply(c("mean", "linterp", 1:20),
  fit_spatial_model,
  plot_df_use = plot_df, profile_data_use = profile_data,
  variable = "psal_m0",
  mc.cores = mc.cores, mc.preschedule = F
)
names(sp_mods_samples) <- c("mean", "linterp", 1:20)
b <- Sys.time()
b - a

file_name <- paste("temp/spatial_models", pressure_val, h_space, "psal", "time.RData",
  sep = "_"
)
save(sp_mods_samples, plot_df, file = file_name)
# predictions
###
data_used <- lapply(sp_mods_samples, function(x) x[[2]])
library(dplyr)
library(GPvecchia)

date <- "2015-08-01"
preds <- mclapply(c("mean", "linterp", 1:20), predict_fun,
  models = sp_mods_samples,
  date = date, plot_df = plot_df, nn = nn_neigh_prediction,
  mc.cores = mc.cores, mc.preschedule = F
)

cond_exp_psal <- unlist(lapply(preds, function(x) as.double(x$mu.pred)))
cond_var_psal <- unlist(lapply(preds, function(x) as.double(x$var.pred)))

plot_df <- plot_df %>%
  mutate(cond_exp = cond_exp_psal, cond_var = cond_var_psal)

file_name <- paste0(
  "temp/spatial_pred_", pressure_val, "_",
  h_space, "_time_psal_", date, ".RData"
)
save(plot_df, data_used, date, nn_neigh_prediction, file = file_name)

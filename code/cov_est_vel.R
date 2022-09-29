source("code/src_R/cov_est_funs.R")
pressure_range <- 5 # range above/below level to accept measurements
pressure_val <- 150
h_space <- 400 # in km, how far to look for data
grid_size <- 1 # 1 is the plot for the presentations, 6 is original grid, 2 and 3 should also work
n_samples <- 20 # Number of samples to read from the estimation
nn_neigh_prediction <- 100
mc.cores <- 5
library(tidyverse)
library(fields)
library(Matrix)
library(parallel)
library(ncdf4)
library(rhdf5)

# Step 1: create a grid on which to compute estimates
grid <- create_grid(grid_size, long_range = c(-60, 30), lat_range = c(-80, -50))


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

traj_data <- readRDS("temp/data/traj_pressure_data_07_11_2020.RDS") %>%
  mutate(float = factor(float), cycle = factor(cycle)) %>%
  group_by(float, cycle) %>%
  summarise(park_pressure = mean(pressure, na.rm = T))
# reduce by parking pressure
profile_data_1000 <- profile_data %>%
  left_join(traj_data, by = c("float", "cycle")) %>%
  filter(!is.na(park_pressure)) %>%
  filter(park_pressure > 950 & park_pressure < 1050)
profile_data <- profile_data_1000
vel_df <- profile_data %>%
  arrange(sample, float, day) %>%
  group_by(sample, float) %>%
  mutate(
    dist_deg_long = fields::rdist.earth.vec(cbind(long, lat),
      cbind(long + 1, lat),
      miles = F
    ),
    dist_deg_lat = fields::rdist.earth.vec(cbind(long, lat),
      cbind(long, lat + 1),
      miles = F
    ),
    v_long_deg = v_long, v_lat_deg = v_lat,
    v_long = v_long * dist_deg_long * 1000 / 86400,
    v_lat = v_lat * dist_deg_lat * 1000 / 86400
  ) %>%
  filter(!is.na(v_long)) %>%
  filter(abs(v_long) < .2, abs(v_lat) < .2)
runs <- c("mean", "linterp", "removed", 1:20)

vel_df <- filter(vel_df, sample == "linterp") %>%
  filter(pos_qc != 8) %>%
  mutate(sample = "removed") %>%
  bind_rows(vel_df) %>%
  filter(sample %in% runs) %>%
  mutate(sample = factor(sample, levels = runs))

vel_df <- filter(vel_df, float != 5901735)
grid_fine <- create_grid(6, long_range = c(-60, 30), lat_range = c(-80, -50))

continent_outlines <-
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group), inherit.aes = F,
    color = "black", fill = "white"
  )
bathymetry <- geom_contour(
  data = grid_fine, aes(x = long, y = lat, z = depth), inherit.aes = F,
  breaks = c(.2, 1000, 2000), color = "black"
)

ggplot(
  data = filter(vel_df, sample %in% c("mean", "linterp", "removed", "1")) %>%
    left_join(data.frame(
      "sample" = c("mean", "linterp", "removed", "1"),
      "label" = factor(c(
        "ArgoSSM mean locations", "Linear interpolation",
        "Interpolated locations removed",
        "One ArgoSSM sample"
      ), levels = c(
        "ArgoSSM mean locations", "Linear interpolation",
        "Interpolated locations removed",
        "One ArgoSSM sample"
      ))
    )),
  aes(x = long_est, y = lat_est, color = park_pressure)
) +
  geom_point(size = .1) +
  facet_wrap(~label, ncol = 2) +
  coord_quickmap(xlim = c(-60, 20), ylim = c(-78, -57)) +
  scale_color_viridis_c() +
  labs(x = "Longitude", y = "Latitude", color = "Parking\nDepth\n(dbar)") +
  theme_bw() +
  bathymetry +
  continent_outlines
ggsave("temp/velocity_meas.png", width = 6.83, height = 4.86)

if (file.exists(paste0("temp/velocity_mean_1000_", h_space, ".RDS"))) {
  load(file = paste0("temp/velocity_mean_1000_", h_space, ".RDS"))
} else {
  plot_df <- plyr::ddply(vel_df,
    .variables = "sample", mean_est_grid, grid = grid, h_space = h_space,
    vars = c("v_long", "v_lat"),
    mc.cores = mc.cores
  ) %>%
    as_tibble() %>%
    mutate(
      dist_deg_long = fields::rdist.earth.vec(cbind(long, lat),
        cbind(long + 1, lat),
        miles = F
      ),
      dist_deg_lat = fields::rdist.earth.vec(cbind(long, lat),
        cbind(long, lat + 1),
        miles = F
      ),
      v_long_deg = v_long_NW / dist_deg_long / 1000 * 86400,
      v_lat_deg = v_lat_NW / dist_deg_lat / 1000 * 86400,
      cond_exp_u = v_long_NW, cond_exp_v = v_lat_NW,
      cond_exp_u_deg = v_long_deg, cond_exp_v_deg = v_lat_deg
    )

  save(plot_df, vel_df, file = paste0("temp/velocity_mean_1000_", h_space, ".RDS"))
}

if (file.exists(paste0("temp/spatial_models_velocity_1000_", h_space, ".RDS"))) {
  load(file = paste0("temp/spatial_models_velocity_1000_", h_space, ".RDS"))
} else {
  u_mod <- mclapply(runs,
    fit_spatial_model_velocity,
    plot_df_use = plot_df,
    profile_data_use = vel_df, variable = "v_long_m0", m_seq = c(10, 20, 40)
  )
  v_mod <- mclapply(runs,
    fit_spatial_model_velocity,
    plot_df_use = plot_df,
    profile_data_use = vel_df, variable = "v_lat_m0", m_seq = c(10, 20, 40)
  )
  names(u_mod) <- names(v_mod) <- runs
  save(plot_df, vel_df, u_mod, v_mod, file = paste0("temp/spatial_models_velocity_1000_", h_space, ".RDS"))
}
library(dplyr)
library(GPvecchia)
u_preds <- list()
u_preds <- mclapply(runs,
  predict_fun_velocity,
  plot_df = plot_df, models = u_mod,
  nn = nn_neigh_prediction, mc.cores = mc.cores, mc.preschedule = F
)
u_cond_exp <- as.double(unlist(lapply(u_preds, function(x) as.double(x$mu.pred))))
u_cond_var <- as.double(unlist(lapply(u_preds, function(x) as.double(x$var.pred))))


plot_df <- plot_df %>%
  mutate(
    cond_exp_u = u_cond_exp + v_long_NW,
    cond_var_u = u_cond_var
  )
file_name <- paste0(
  "temp/spatial_pred_velocity_prelim_1000_", h_space, ".RData"
)
save(plot_df, nn_neigh_prediction, file = file_name)

v_preds <- mclapply(runs,
  predict_fun_velocity,
  plot_df = plot_df, models = v_mod,
  nn = nn_neigh_prediction, mc.cores = mc.cores, mc.preschedule = F
)
v_cond_exp <- as.double(unlist(lapply(v_preds, function(x) as.double(x$mu.pred))))
v_cond_var <- as.double(unlist(lapply(v_preds, function(x) as.double(x$var.pred))))

plot_df <- plot_df %>%
  mutate(
    cond_exp_v = v_cond_exp + v_lat_NW,
    cond_var_v = v_cond_var
  )
file_name <- paste0(
  "temp/spatial_pred_velocity_1000_", h_space, ".RData"
)
save(plot_df, nn_neigh_prediction, file = file_name)

q()

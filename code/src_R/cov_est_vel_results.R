library(tidyverse)
library(fields)
library(Matrix)
library(ncdf4)
load("temp/spatial_pred_velocity_1000_400.RData")
plot_only_df <- plot_df %>%
  filter(n_prof > 5)
label_df <- data.frame(
  sample = c("mean", "linterp", "removed", "total"),
  label = factor(c(
    "ArgoSSM mean", "Linear interpolation",
    "Interpolated locations removed",
    "ArgoSSM"
  ),
  levels = c(
    "ArgoSSM mean", "Linear interpolation",
    "Interpolated locations removed",
    "ArgoSSM"
  )
  ),
  label_mean = factor(c(
    "ArgoSSM mean", "Linear interpolation",
    "Interpolated locations removed",
    "ArgoSSM"
  ), levels = c(
    "ArgoSSM mean", "Linear interpolation",
    "Interpolated locations removed",
    "ArgoSSM"
  ))
)

source("code/src_R/cov_est_funs.R")

grid_fine <- create_grid(6, long_range = c(-60, 30), lat_range = c(-80, -50))

ssm_avg <- plot_df %>%
  filter(sample %in% c(1:20)) %>%
  group_by(long, lat) %>%
  summarise(
    n_prof = mean(n_prof),
    cond_var_u = var(cond_exp_u, na.rm = T) + mean(cond_var_u, na.rm = T),
    cond_var_v = var(cond_exp_v, na.rm = T) + mean(cond_var_v, na.rm = T),
    cond_var_u_p1 = mean(cond_var_u, na.rm = T),
    cond_var_v_p1 = mean(cond_var_v, na.rm = T),
    cond_var_u_p2 = var(cond_exp_u, na.rm = T),
    cond_var_v_p2 = var(cond_exp_v, na.rm = T),
    cond_exp_u = mean(cond_exp_u, na.rm = T),
    cond_exp_v = mean(cond_exp_v, na.rm = T),
    width = mean(width), height = mean(height),
    depth = mean(depth), .groups = "drop"
  ) %>%
  mutate(sample = "total")

plot_df_avg <- rbind(
  select(
    plot_df, long, lat, n_prof, cond_exp_u, cond_exp_v,
    cond_var_u, cond_var_v, width,
    height, depth, sample
  ),
  select(
    ssm_avg, long, lat, n_prof, cond_exp_u, cond_exp_v,
    cond_var_u, cond_var_v, width,
    height, depth, sample
  )
)
plot_df_avg <- filter(
  plot_df_avg, n_prof > 30,
  sample %in% c("mean", "linterp", "removed", "total")
) %>%
  left_join(label_df)

base_plot <- ggplot() +
  scale_fill_viridis_c() +
  scale_color_viridis_c() +
  coord_quickmap(xlim = c(-60, 20), ylim = c(-78, -57)) +
  labs(x = "Longitude", y = "Latitude", fill = "Cond SD", color = "Cond SD") +
  theme_bw()
continent_outlines <-
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group), inherit.aes = F,
    color = "black", fill = "white"
  )
bathymetry <- geom_contour(
  data = grid_fine, aes(x = long, y = lat, z = depth), inherit.aes = F,
  breaks = c(.2, 1000, 2000), color = "black"
)
base_plot +
  geom_tile(
    data = pivot_longer(ssm_avg, starts_with("cond_var_u"),
      names_to = "type",
      values_to = "cond_var_u_decom"
    ),
    aes(
      x = long, y = lat, fill = sqrt(cond_var_u_decom), color = sqrt(cond_var_u_decom),
      width = width, height = height
    )
  ) +
  facet_wrap(~type) + continent_outlines + bathymetry


base_plot +
  geom_tile(
    data = filter(plot_df_avg, sample != "mean"),
    aes(
      x = long, y = lat, fill = sqrt(cond_var_u) / 1000 * 86400, color = sqrt(cond_var_u) / 1000 * 86400,
      width = width, height = height
    )
  ) +
  continent_outlines + bathymetry +
  facet_wrap(~label, ncol = 2) +
  labs(fill = "Cond SD\n(km/day)", color = "Cond SD\n(km/day)") +
  theme(legend.position = "bottom")
ggsave("temp/var_vel_u.png", width = 5, height = 6)

base_plot +
  geom_tile(
    data = filter(plot_df_avg, sample != "mean"),
    aes(
      x = long, y = lat,
      fill = cond_exp_u / 1000 * 86400,
      color = cond_exp_u / 1000 * 86400,
      width = width, height = height
    )
  ) +
  scale_fill_gradient2() +
  scale_color_gradient2() +
  facet_wrap(~label_mean, ncol = 2) +
  continent_outlines + bathymetry +
  theme_bw() +
  labs(fill = "Velocity\n(km/day)", color = "Velocity\n(km/day)") +
  theme(legend.position = "bottom")
ggsave("temp/exp_vel_u.png", width = 5, height = 6)

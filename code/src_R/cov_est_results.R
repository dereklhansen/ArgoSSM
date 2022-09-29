################# Results #############
library(dplyr)
library(ggplot2)
pressure_val <- 150
h_space <- 250
date <- "2015-08-01"
variable <- "psal"

file_name <- paste0(
  "temp/spatial_pred_", pressure_val, "_",
  h_space, "_time_", variable, "_", date, ".RData"
)

load(file_name)
locations <- bind_rows(data_used)
day_to_pred <- as.numeric(julian(as.Date(date, format = "%Y-%m-%d"),
  origin = as.Date("1950-01-01")
)) %% 365
constants <- 2 * pi * (1:6) / 365.25
covariates <- c(
  1, sin(constants[1] * day_to_pred), cos(constants[1] * day_to_pred),
  sin(constants[2] * day_to_pred), cos(constants[2] * day_to_pred)
)
if (variable == "temp") {
  plot_df <- plot_df %>%
    mutate(
      cond_sd = sqrt(cond_var),
      mean_est = temp_ll.1 +
        temp_ll.2 * covariates[2] +
        temp_ll.3 * covariates[3] +
        temp_ll.4 * covariates[4] +
        temp_ll.5 * covariates[5],
      cond_exp_true = cond_exp + mean_est,
      diff = cond_exp_true - mean_est,
      cond_exp = cond_exp_true
    ) %>%
    filter(n_prof > 20)
  limits_value <- c(-1.8, 1.1)
  breaks_value <- c(-1.25, -.5, 0, .5, 1)
  limits_sd <- c(.2, 1.3)
  limits_sd_mean <- c(0, 1.1)
  var_label <- "Temperature"
  var_units <- "°C"
} else {
  plot_df <- plot_df %>%
    mutate(
      cond_sd = sqrt(cond_var),
      mean_est = psal_ll.1 +
        psal_ll.2 * covariates[2] +
        psal_ll.2 * covariates[3] +
        psal_ll.2 * covariates[4] +
        psal_ll.2 * covariates[5],
      cond_exp_true = cond_exp + mean_est,
      diff = cond_exp_true - mean_est,
      cond_exp = cond_exp_true
    ) %>%
    filter(n_prof > 20)
  limits_value <- c(34.3, 34.7)
  breaks_value <- seq(34.3, 34.7, by = .07)
  limits_sd <- c(0, .18)
  limits_sd_mean <- c(0, .10)
  var_label <- "Salinity"
  var_units <- "PSU"
}

theme_set(theme_bw() + theme(text = element_text(size = 18)))
ggplot() +
  scale_fill_stepsn(
    colors = colorRamps::matlab.like(10),
    limits = limits_value, breaks = breaks_value,
  ) +
  geom_tile(
    data = filter(plot_df, !is.na(temp_NW), sample %in% 1:4),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = cond_exp
    ), size = .000001
  ) +
  geom_point(
    data = locations[locations$sample %in% 1:4, ] %>%
      filter(
        date > "2015-03-01", date < "2016-03-01",
        pos_qc %in% c(8), !is.na(long_est)
      ),
    aes(x = long, y = lat), color = "black", size = .8, shape = 17, stroke = .8,
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2
  ) +
  labs(
    title = paste(var_label, "Predictions"), fill = var_units,
    y = "Latitude", x = "Longitude"
  ) +
  coord_quickmap(
    expand = F, clip = T,
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  facet_wrap(~sample)
ggsave(paste0("temp/cond_exp_samples_", variable, ".png"),
  width = 6, height = 4, scale = 2
)

ggplot() +
  scale_color_viridis_c(limits = limits_sd) +
  scale_fill_viridis_c(limits = limits_sd) +
  geom_tile(
    data = filter(plot_df, !is.na(temp_NW), sample %in% 1:4),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = cond_sd, color = cond_sd
    )
  ) +
  geom_point(
    data = locations[locations$sample %in% 1:4, ] %>%
      filter(
        date > "2015-03-01", date < "2016-03-01",
        pos_qc %in% c(8), !is.na(long_est)
      ),
    aes(x = long, y = lat), color = "white", size = .4, shape = 17, stroke = .4,
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_quickmap(
    expand = F, clip = T,
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  labs(
    y = "Latitude", x = "Longitude",
    title = paste(var_label, "Predictive Standard Deviation"), fill = "SD",
    color = "SD"
  ) +
  facet_wrap(~sample, ncol = 2)
ggsave(paste0("temp/cond_sd_samples_", variable, ".png"), width = 6, height = 4, scale = 2)




plot_df_sum <- plot_df %>%
  filter(sample %in% 1:20) %>%
  group_by(long, lat) %>%
  summarise(
    height = head(height, 1), width = head(width, 1),
    var_cond_exp = var(cond_exp, na.rm = T),
    mean_cond_var = mean(cond_var, na.rm = T),
    mean_cond_exp = mean(cond_exp, na.rm = T)
  )

ggplot() +
  scale_fill_viridis_c(limits = limits_sd) +
  scale_color_viridis_c(limits = limits_sd) +
  geom_tile(
    data = filter(
      plot_df_sum, !is.na(mean_cond_var),
      !is.na(var_cond_exp)
    ),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = sqrt(mean_cond_var), color = sqrt(mean_cond_var)
    )
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_quickmap(
    expand = F, clip = "on",
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  labs(
    y = "Latitude", x = "Longitude",
    title = paste("Averaged", var_label, "Predictive Standard Deviation"), fill = "SD", color = "SD"
  ) +
  theme(text = element_text(size = 24))
ggsave(paste0("temp/cond_sd_avg_", variable, ".png"), width = 6, height = 4, scale = 2)

ggplot() +
  scale_fill_viridis_c(limits = limits_sd) +
  geom_tile(
    data = filter(
      plot_df_sum, !is.na(mean_cond_var),
      !is.na(var_cond_exp)
    ),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = ifelse(sqrt(mean_cond_var + var_cond_exp) > limits_sd[2],
        limits_sd[2], sqrt(mean_cond_var + var_cond_exp)
      )
    ), size = .000001,
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_quickmap(
    expand = F, clip = "on",
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Total Predictive Standard Deviation", fill = "SD", color = "SD"
  ) +
  theme(text = element_text(size = 24))
ggsave(paste0("temp/cond_sd_avg_total_", variable, ".png"), width = 6, height = 4, scale = 2)




ggplot() +
  scale_fill_stepsn(
    colors = colorRamps::matlab.like(10),
    limits = limits_value, breaks = breaks_value,
  ) +
  geom_tile(
    data = filter(
      plot_df_sum, !is.na(mean_cond_exp),
      !is.na(var_cond_exp)
    ),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = mean_cond_exp
    ), size = .000001
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_quickmap(
    expand = F, clip = "on",
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Averaged Predictions", fill = var_units, color = var_units
  ) +
  theme(text = element_text(size = 24))
ggsave(paste0("temp/cond_exp_avg_", variable, ".png"), width = 6, height = 4, scale = 2)

# results for linear interpolation
ggplot() +
  scale_fill_stepsn(
    colors = colorRamps::matlab.like(10),
    limits = limits_value, breaks = breaks_value,
  ) +
  geom_tile(
    data = filter(plot_df, !is.na(temp_NW), sample == "linterp"),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = cond_exp
    ), size = .000001
  ) +
  geom_point(
    data = filter(locations, sample %in% "linterp") %>%
      filter(
        date > "2015-03-01", date < "2016-03-01",
        pos_qc %in% c(8), !is.na(long_est)
      ),
    aes(x = long, y = lat), color = "black", size = 1.2, shape = 17, stroke = .8,
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2
  ) +
  labs(
    title = paste(var_label, "Predictions"), fill = var_units,
    y = "Latitude", x = "Longitude"
  ) +
  coord_quickmap(
    expand = F, clip = "on",
    xlim = c(-59, 20), ylim = c(-75, -58)
  )
ggsave(paste0("temp/cond_exp_linterp_", variable, ".png"), width = 6, height = 4, scale = 2)

ggplot() +
  scale_color_viridis_c(limits = limits_sd) +
  scale_fill_viridis_c(limits = limits_sd) +
  geom_tile(
    data = filter(plot_df, !is.na(temp_NW), sample == "linterp"),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = ifelse(cond_sd > limits_sd[2], limits_sd[2], cond_sd),
      color = ifelse(cond_sd > limits_sd[2], limits_sd[2], cond_sd)
    )
  ) +
  geom_point(
    data = locations %>%
      filter(
        sample == "linterp",
        date > "2015-03-01", date < "2016-03-01",
        pos_qc %in% c(8), !is.na(long_est)
      ),
    aes(x = long, y = lat), color = "white", size = 1.2, shape = 17, stroke = .4,
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_quickmap(
    expand = F, clip = "on",
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  labs(
    y = "Latitude", x = "Longitude",
    title = paste(var_label, "Predictive Standard Deviation"), fill = "SD",
    color = "SD"
  )
ggsave(paste0("temp/cond_sd_linterp_", variable, ".png"), width = 6, height = 4, scale = 2)


ggplot() +
  scale_fill_viridis_c(limits = limits_sd_mean) +
  geom_tile(
    data = filter(
      plot_df_sum, !is.na(mean_cond_var),
      !is.na(var_cond_exp)
    ),
    aes(
      x = long, y = lat, height = height + .01, width = width,
      fill = ifelse(sqrt(var_cond_exp) > limits_sd_mean[2],
        limits_sd_mean[2], sqrt(var_cond_exp)
      )
    ), size = .000001,
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  coord_quickmap(
    expand = F, clip = "on",
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  labs(
    y = "Latitude", x = "Longitude",
    title = "Standard Deviation of Predictive Mean", fill = "SD", color = "SD"
  ) +
  theme(text = element_text(size = 24))
ggsave(paste0("temp/cond_mean_var_", variable, ".png"), width = 6, height = 4, scale = 2)

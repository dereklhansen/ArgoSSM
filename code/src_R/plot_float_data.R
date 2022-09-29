float_data <- readRDS("temp/data/float_data_07_11_2020.RDS")

library(ggplot2)
library(mapproj)
library(dplyr)
# interpolated locations
# I focused on floats that could have been in the region focused on in Chamberlain et al
ggplot() +
  geom_point(data = float_data[float_data$pos_qc == 8, ], aes(x = long, y = lat), size = .2, color = "red") +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2
  ) +
  coord_map("ortho", orientation = c(-90, 0, 0), ylim = c(-90, -30)) +
  theme_bw()
float_data$pos_qc <- factor(float_data$pos_qc,
  levels = c(" ", "0", "1", "2", "3", "4", "8", "9")
)

float_missing_poster <- float_data %>%
  group_by(float) %>%
  mutate(missing = sum(pos_qc %in% c(8, 9)) > 0) %>%
  ungroup() %>%
  filter(long > -65, long < 30, lat < -45) %>%
  mutate(pos2 = ifelse(pos_qc %in% c(8, 9), "Linear\nInterpolation", ifelse(pos_qc %in% c(1, 2),
    "GPS", "bad"
  ))) %>%
  filter(pos2 != "bad") %>%
  filter(missing) %>%
  filter(!(float %in% 5904096))
ggplot() +
  geom_point(
    data = float_missing_poster, aes(x = long, y = lat, color = pos2),
    size = .1
  ) +
  geom_path(
    data = float_missing_poster, aes(
      x = long, y = lat, color = pos2,
      group = float
    ),
    size = .2
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2
  ) +
  scale_color_discrete(drop = FALSE) +
  coord_cartesian(xlim = c(-60, 30), ylim = c(-75, -50)) +
  theme_bw() +
  labs(
    color = "Position QC", x = "Longitude", y = "Latitude" # ,
  ) +
  theme(legend.position = "bottom")
ggsave(paste0("temp/float_missing_poster.png"), height = 5, width = 6)

ggplot() +
  geom_point(
    data = float_data, aes(x = long, y = lat, color = pos_qc),
    size = .1
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2
  ) +
  scale_color_discrete(drop = FALSE) +
  coord_cartesian(xlim = c(-60, 30), ylim = c(-80, -50)) +
  theme_bw() +
  labs(
    color = "Position QC", x = "Longitude", y = "Latitude",
    title = "Data from floats with interpolated positions"
  )
ggsave(paste0("temp/float_missing_", date, ".png"), height = 4, width = 6)


float_data_old <- readRDS("temp/data/float_data.RDS")
float_data_old$pos_qc <- factor(float_data_old$pos_qc,
  levels = c(" ", "0", "1", "2", "3", "4", "5", "8", "9")
)
ggplot() +
  geom_point(
    data = float_data_old, aes(x = long, y = lat, color = pos_qc),
    size = .1
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2
  ) +
  scale_color_discrete(drop = FALSE) +
  coord_cartesian(xlim = c(-60, 60), ylim = c(-80, -50)) +
  theme_bw() +
  labs(
    color = "Position QC", x = "Longitude", y = "Latitude",
    title = "Float positions as of February 2020"
  )
ggsave(paste0("temp/float_positions_previous.png"), height = 4, width = 6)

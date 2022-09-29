library(tidyverse)
library(lubridate)
library(rhdf5)

prof_data_in <- read_rds("temp/data/prof_data_06_2021.RDS")
prof_data <- prof_data_in %>%
  tbl_df() %>%
  mutate(date = as_date(as.integer(day), origin = "1950-01-01")) %>%
  select(float, date, day, lat, long, pos_qc, pos_sys) %>%
  distinct() %>%
  group_by(float, date) %>%
  summarize(
    lat = head(lat, 1), long = head(long, 1), pos_qc = head(pos_qc, 1), pos_sys = head(pos_sys, 1),
    day = head(day, 1)
  ) %>%
  ungroup()

linearinterp <- function(a, b, d_a, d_b, d_x) {
  (a * (d_b - d_x) + b * (d_x - d_a)) / (d_b - d_a)
}

make_intermediate_dates <- function(tbl_row, max_gap) {
  n_rows <- ceiling(tbl_row$date_diff / max_gap)
  tibble(
    float = tbl_row$float,
    date = seq(tbl_row$date_prev + max_gap, tbl_row$date - 1, max_gap),
    pos_qc = factor("9", levels = c("1", "2", "8", "9")),
    lat_oob = NA,
    long_oob = NA
  ) %>%
    mutate(long = linearinterp(tbl_row$long_prev, tbl_row$long, as.integer(tbl_row$date_prev), as.integer(tbl_row$date), as.integer(date))) %>%
    mutate(lat = linearinterp(tbl_row$lat_prev, tbl_row$lat, as.integer(tbl_row$date_prev), as.integer(tbl_row$date), as.integer(date)))
}

fill_in_gaps <- function(tbl, max_gap = 10) {
  my_tbl <- tbl %>%
    mutate(
      date_prev = lag(date), long_prev = lag(long),
      lat_prev = lag(lat), date_diff = date - date_prev
    )
  gaps_too_long <- filter(my_tbl, date_diff > max_gap)
  gaps_filled_in <- gaps_too_long %>%
    plyr::adply(.margins = 1, make_intermediate_dates, max_gap = max_gap) %>%
    as_tibble()

  out_tbl <- my_tbl %>%
    bind_rows(gaps_filled_in) %>%
    arrange(date) %>%
    mutate(t = 1:length(float)) %>%
    mutate(date_prev = c(NA_Date_, head(date, -1)), date_diff = date - date_prev)

  return(out_tbl)
}

float_tbl <- prof_data %>%
  arrange(float, date) %>%
  group_by(float) %>%
  fill_in_gaps() %>%
  arrange(float, date) %>%
  mutate(pos_qc = as.integer(as.character(pos_qc))) %>%
  filter(date >= min(date[pos_qc %in% c(1, 2)])) %>%
  group_by(float) %>%
  mutate(
    n_good = sum(pos_qc %in% c(1, 2)),
    n_pos_bad = sum((lat > -60) | (long < -60) | (long > 20))
  ) %>%
  ungroup() %>%
  filter(n_good > 25, n_pos_bad == 0) %>%
  arrange(float, date)

float_tbl <- float_tbl %>%
  mutate(
    missing = ifelse(pos_qc %in% c(4, 8, 9), "Missing", "GPS"),
    month = factor(month.name[as.numeric(substr(date, 6, 7))],
      levels = c("All months", month.name)
    )
  )

ggplot() +
  geom_path(
    data = float_tbl,
    aes(x = long, y = lat, color = missing, group = float), size = .2
  ) +
  geom_point(
    data = float_tbl,
    aes(x = long, y = lat, color = missing), size = .2
  ) +
  labs(x = "Longitude", y = "Latitude", color = "Measurement Type") +
  theme_bw() +
  theme(legend.position = "bottom") +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    color = "black", fill = "white", size = .2, inherit.aes = F
  ) +
  coord_quickmap(xlim = c(-60, 20), ylim = c(-75, -60)) # +
ggsave("temp/float_locations.png", height = 3.5, width = 6)


ggplot(data = float_tbl, aes(x = month, fill = missing)) +
  geom_bar() +
  labs(x = "Month", y = "Number of Profiles", fill = "Measurement Type") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 60, vjust = .96, hjust = 1), legend.position = "bottom",
    text = element_text(size = 20)
  )
ggsave("temp/float_bar.png", height = 7.3, width = 6)

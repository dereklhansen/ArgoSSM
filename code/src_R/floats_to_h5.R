library(tidyverse)
library(lubridate)
library(rhdf5)
library(stringi)
# https://archimer.ifremer.fr/doc/00228/33951/32470.pdf
# source('code/src/import_data_box.R')
main <- function(prof_data_file = "temp/data/prof_data_06_2021.RDS") {
  prof_data_in <- read_rds(prof_data_file)
  float_tbl <- make_float_tbl(prof_data_in)
  output_rds <- "output/float_tbl.rds"
  if (file.exists(output_rds)) {
    file.remove(output_rds)
  }
  write_rds(float_tbl, output_rds)
  message("Generated output/float_tbl.rds")

  ## Construct unique float objects
  unique_floatids <- as.character(unique(float_tbl$float))
  floats <- map(unique_floatids, function(id) {
    tbl <- filter(float_tbl, float == id)
    float_data <- list()
    float_data$id <- id
    float_data$X <- cbind(tbl$long, tbl$lat)
    float_data$days <- as.integer(tbl$date)
    float_data$pos_qc <- tbl$pos_qc
    return(float_data)
  })
  names(floats) <- unique_floatids

  ## Write to H5 file
  output_h5 <- "output/floats.h5"
  write_h5_from_list(floats, output_h5)
  message("Generated output/floats.h5")

  ## Holdout experiment
  holdouts_under_ice <- map(floats, function(float_data) {
    holdout_tbl <- mark_holdouts_under_ice(float_data$pos_qc)
    holdouts <- pmap(holdout_tbl, function(idx, type, holdout_id) {
      holdout_data <- float_data
      holdout_data$holdout_idx <- idx
      holdout_data$holdout_type <- type
      holdout_data$holdout_id <- holdout_id
      return(holdout_data)
    })
    return(holdouts)
  })
  output_rds <- "output/holdouts_under_ice.rds"
  if (file.exists(output_rds)) {
    file.remove(output_rds)
  }
  write_rds(holdouts_under_ice, output_rds)

  # output_h5 <- "output/holdouts_under_ice.h5"
  write_h5_from_list(holdouts_under_ice, "output/holdouts_under_ice.h5")
  message("Generated output/holdouts_under_ice.h5")

  holdouts_streaks <- map(floats, function(float_data) {
    holdout_tbl <- mark_holdouts_streaks(float_data$pos_qc)
    if (is.null(holdout_tbl)) {
      return(NULL)
    } else {
      out <- pmap(holdout_tbl, function(holdout_begin, holdout_end, holdout_id) {
        holdout_data <- float_data
        holdout_data$holdout_idx <- holdout_begin:holdout_end
        holdout_data$holdout_type <- "streak"
        holdout_data$holdout_id <- holdout_id
        return(holdout_data)
      })
      return(out)
    }
  })
  holdouts_streaks <- purrr::compact(holdouts_streaks)

  output_rds <- "output/holdouts_streaks.rds"
  if (file.exists(output_rds)) {
    file.remove(output_rds)
  }
  write_rds(holdouts_streaks, output_rds)

  write_h5_from_list(holdouts_streaks, "output/holdouts_streaks.h5")
  message("Generated output/holdouts_streaks.h5")
}

make_float_tbl <- function(prof_data_in) {
  prof_data <- prof_data_in %>%
    tibble::as_tibble() %>%
    mutate(date = as_date(as.integer(day), origin = "1950-01-01")) %>%
    select(float, date, day, lat, long, pos_qc, pos_sys) %>%
    distinct() %>%
    group_by(float, date) %>%
    summarize(
      lat = head(lat, 1),
      long = head(long, 1),
      pos_qc = head(pos_qc, 1),
      pos_sys = head(pos_sys, 1),
      day = head(day, 1)
    ) %>%
    ungroup()

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
  return(float_tbl)
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

make_intermediate_dates <- function(tbl_row, max_gap) {
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

linearinterp <- function(a, b, d_a, d_b, d_x) {
  (a * (d_b - d_x) + b * (d_x - d_a)) / (d_b - d_a)
}

write_h5_from_list <- function(in_list, h5_file) {
  if (file.exists(h5_file)) {
    file.remove(h5_file)
  }
  H5Fcreate(h5_file)
  write_h5group_from_list(in_list, h5_file, group_name = "")
}

write_h5group_from_list <- function(in_list, h5_file, group_name = "") {
  if (group_name != "") {
    h5f <- H5Fopen(h5_file)
    group <- H5Gcreate(h5f, group_name)
    H5Gclose(group)
    H5Fclose(h5f)
    h5closeAll()
  }

  for (i in seq_along(in_list)) {
    x <- in_list[[i]]
    if (is.null(names(in_list))) {
      name_x <- as.character(i)
    } else {
      name_x <- names(in_list)[i]
    }
    if ("list" %in% class(x)) {
      write_h5group_from_list(x, h5_file, group_name %s+% "/" %s+% name_x)
    } else {
      h5write(x, h5_file, group_name %s+% "/" %s+% name_x)
    }
  }
}

mark_holdouts_under_ice <- function(pos_qc) {
  A <- pos_qc %in% c(1, 2)
  streak_len <- streaklen(A)
  streak_len_reverse <- rev(streaklen(rev(A)))

  before_ice <- A & (lag(streak_len * (A)) > 5) & (lead(streak_len_reverse * (!A)) > 5)
  before_ice[c(1, length(before_ice))] <- FALSE
  before_ice <- which(before_ice)

  after_ice <- A & (lag(streak_len * (!A)) > 5) & (lead(streak_len_reverse * (A)) > 5)
  after_ice[c(1, length(after_ice))] <- FALSE
  after_ice <- which(after_ice)

  holdout_tbl <- bind_rows(tibble(idx = before_ice, type = "before_ice"), tibble(idx = after_ice, type = "after_ice"))
  holdout_tbl <- mutate(holdout_tbl, holdout_id = seq_along(idx))
}

mark_holdouts_streaks <- function(pos_qc, streak_length = 11, pad = (1 / 6)) {
  A <- pos_qc %in% c(1, 2)
  streak_len <- streaklen(A)
  streak_len_reverse <- rev(streaklen(rev(A)))

  ## Identify all streaks bigger than streak_length
  streak_end <- ((streak_len * A) >= streak_length) & lead(!A, default = TRUE)
  streak_end <- which(streak_end)
  streak_begin <- ((streak_len_reverse * A) >= streak_length) & lag(!A, default = TRUE)
  streak_begin <- which(streak_begin)

  # holdout_variants <- purrr::cross(
  #   list(location=c("before", "after"), op=c("shift", "add"))
  # )
  holdout_variants <- list(
    c(-1, -1),
    c(1, 1),
    c(-1, 0),
    c(0, 1)
  )

  ## Put big streaks into a table
  if (length(streak_begin) > 0) {
    streak_tbl <- tibble(streak_begin = streak_begin, streak_end = streak_end, streak_len = (streak_end - streak_begin + 1))
    holdout_tbl <- mutate(streak_tbl, holdout_begin = streak_begin + ceiling(streak_len * pad), holdout_end = streak_end - ceiling(streak_len * pad))
    holdout_tbl_variants <- map(holdout_variants, function(variant) {
      mutate(holdout_tbl, holdout_begin = holdout_begin + variant[1], holdout_end = holdout_end + variant[2])
    })
    holdout_tbl_final <- bind_rows(c(list(holdout_tbl), holdout_tbl_variants), .id = "holdout_id")
    holdout_tbl_final <- select(holdout_tbl_final, holdout_begin, holdout_end, holdout_id)
    return(holdout_tbl_final)
  } else {
    return(NULL)
  }
}

streaklen <- function(x) {
  r <- rle(x)
  map(r$lengths, seq, from = 1) %>%
    reduce(c)
}



main()

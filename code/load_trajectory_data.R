# Load trajectory files
# Note: run load_trajectory_data.R first
date <- "06_2021"
argo_data_location <- "~/Downloads/202106-ArgoData/dac/"
# package for reading netcdf files
library(ncdf4)
library(tidyverse)

traj_info <- read.delim(
  file = paste0(argo_data_location, "ar_index_global_traj.txt"),
  skip = 8, header = T,
  sep = ",", stringsAsFactors = FALSE
)

tech_info <- read.delim(
  file = paste0(argo_data_location, "ar_index_global_tech.txt"),
  skip = 8, header = T,
  sep = ",", stringsAsFactors = FALSE
)

traj_info_SO <- traj_info[(traj_info$latitude_min < -60 &
  traj_info$longitude_min < 20 &
  traj_info$longitude_max > -60) |
  is.na((traj_info$latitude_min < -60 &
    traj_info$longitude_min < 20 &
    traj_info$longitude_max > -60)), ]
floats <- sapply(strsplit(traj_info_SO$file, "\\/"), function(x) x[2])
floats_tech <- sapply(strsplit(tech_info$file, "\\/"), function(x) x[2])

tech_info_SO <- tech_info[floats_tech %in% floats, ]
floats_tech <- floats_tech[floats_tech %in% floats]
# Load information from each file
data_list <- list()
for (i in 1:nrow(traj_info_SO)) {
  file <- nc_open(paste0(argo_data_location, traj_info_SO$file[i]))
  float_name <- strsplit(traj_info_SO$file[i], "\\/")[[1]][2]
  if (float_name %in% floats_tech) {
    i_tech <- which(floats_tech == float_name)[1]
    tech_file <- nc_open(paste0(
      argo_data_location,
      tech_info_SO$file[i_tech]
    ))
    name_tech <- ncvar_get(tech_file, "TECHNICAL_PARAMETER_NAME")
    test <- grepl(name_tech, pattern = "FLAG_IceDe", ignore.case = T)
    ice_names <- name_tech[test]
    ice_values <- ncvar_get(tech_file, "TECHNICAL_PARAMETER_VALUE")[test]
    cycle <- ncvar_get(tech_file, "CYCLE_NUMBER")[test]
    nc_close(tech_file)
    if (length(ice_names) == 0) {
      traj_df <- data.frame(cycle = NA, ice_det = NA)
    } else {
      traj_df <- data.frame(i,
        index = 1:length(ice_names),
        cycle, ice_names, ice_values
      ) %>%
        mutate(
          ice_names_str = gsub(" ", "", ice_names, fixed = TRUE),
          ice_values_str = gsub(" ", "", ice_values, fixed = TRUE),
          ice_names_bit = ifelse(ice_names_str == "FLAG_IceDetected_NUMBER",
            F, T
          ),
          ice_det = ifelse(ice_names_bit,
            as.integer(ice_values_str) %% 2 == 1,
            ifelse(as.numeric(ice_values_str) > 0, 1, 0)
          )
        ) %>%
        dplyr::select(cycle, ice_det)
    }
  } else {
    traj_df <- data.frame(cycle = NA, ice_det = NA)
  }
  cycle <- ncvar_get(file, "CYCLE_NUMBER")

  if ("REPRESENTATIVE_PARK_PRESSURE" %in% names(file$var)) {
    park_pressure <- ncvar_get(file, "REPRESENTATIVE_PARK_PRESSURE")
    cycle_dim <- file$dim$N_CYCLE$vals
    park_pressure_df <- data.frame(cycle = cycle_dim, pressure = park_pressure)
    park_pressure <- left_join(data.frame(cycle), park_pressure_df, by = "cycle")
  } else {
    park_pressure <- data.frame(cycle = c(0), pressure = NA)
  }
  if (sum(is.na(park_pressure$pressure)) != length(park_pressure$pressure)) {

  } else if ("PRES" %in% names(file$var)) {
    mode <- strsplit(ncvar_get(file, "DATA_MODE"), split = "")[[1]]
    if (mode[1] == "R") {
      pressure <- ncvar_get(file, "PRES")
    } else {
      pressure <- ncvar_get(file, "PRES_ADJUSTED")
    }
    if (sum(is.na(pressure)) == length(pressure)) {
      pressure <- ncvar_get(file, "PRES")
    }
    juld_desc <- ncvar_get(file, "JULD_DESCENT_START")
    juld <- ncvar_get(file, "JULD")
    cycle <- ncvar_get(file, "CYCLE_NUMBER")
    if (length(juld_desc) == sum(is.na(juld_desc)) & "MEASUREMENT_CODE" %in% names(file$var)) {
      measurement_code <- ncvar_get(file, "MEASUREMENT_CODE")
      indexes <- measurement_code > 249 & measurement_code < 301
      park_pressure <- data.frame(pressure = pressure[indexes], cycle = cycle[indexes]) %>%
        left_join(data.frame(cycle), park_pressure, by = "cycle") %>%
        group_by(cycle) %>%
        summarise(pressure = mean(pressure, na.rm = T)) %>%
        filter(!is.na(cycle))
    } else {
      indexes_park <- sapply(juld, function(x) {
        min(abs((x - juld_desc)[(x - juld_desc) > .2]),
          na.rm = T
        ) < 8
      })
      park_pressure <- data.frame(pressure, cycle, indexes_park) %>%
        filter(indexes_park) %>%
        group_by(cycle) %>%
        summarise(pressure = mean(pressure, na.rm = T))
      park_pressure <- left_join(data.frame(cycle), park_pressure, by = "cycle") %>%
        filter(!is.na(cycle))
    }
  } else {
    park_pressure <- data.frame(cycle = 0, pressure = NA)
    print(i)
    print(park_pressure)
  }



  lat <- ncvar_get(file, "LATITUDE")
  long <- ncvar_get(file, "LONGITUDE")
  pos_qc <- strsplit(ncvar_get(file, "POSITION_QC"), "")[[1]]
  pos_sys <- ncvar_get(file, "POSITIONING_SYSTEM")
  day <- ncvar_get(file, "JULD")
  day_qc <- strsplit(ncvar_get(file, "JULD_QC"), "")[[1]] # quality control on day
  float <- as.numeric(ncvar_get(file, "PLATFORM_NUMBER")) # float number
  # mode <- strsplit(ncvar_get(file, 'DATA_MODE'), '')[[1]]
  pos_acc <- strsplit(ncvar_get(file, "POSITION_ACCURACY"), "")[[1]]
  reference <- ncvar_get(file, "REFERENCE_DATE_TIME")
  year <- substr(reference, 1, 4)
  month <- substr(reference, 5, 6)
  day_ref <- substr(reference, 7, 8)
  hours <- substr(reference, 9, 10)
  minutes <- substr(reference, 11, 12)
  seconds <- substr(reference, 13, 14)
  date_time <- as.POSIXct(day * 60 * 60 * 24,
    origin = as.POSIXct(ISOdatetime(
      year = year, month = month, day = day_ref,
      hour = hours, min = minutes,
      sec = seconds, tz = "GMT"
    )),
    tz = "GMT"
  )
  date_use <- as.Date(date_time)
  nc_close(file)

  df_return <- data.frame(float, cycle, lat, long, pos_qc, pos_sys, day, day_qc,
    pos_acc, reference,
    date = date_use, date_time
  ) %>%
    left_join(park_pressure, by = "cycle") %>%
    group_by(float, cycle) %>%
    summarise(
      pressure = mean(pressure, na.rm = T), lat = mean(lat, na.rm = T),
      long = mean(long, na.rm = T),
      pos_qc = pos_qc[1], day = mean(day, na.rm = T),
      day_qc = day_qc[1], pos_acc = pos_acc[1],
      reference = reference[1], date = date[1], date_time = date_time[1],
      .groups = "drop"
    ) %>%
    left_join(traj_df, by = "cycle")
  data_list[[i]] <- df_return
  print(i)
}
float_data <- do.call(rbind, data_list)
saveRDS(float_data, file = paste0("temp/data/traj_data_", date, ".RDS"))

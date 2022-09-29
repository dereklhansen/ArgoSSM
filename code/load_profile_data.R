
# Load profile data
argo_data_location <- "~/Downloads/202106-ArgoData/dac/"
# package for reading netcdf files
library(ncdf4)

date <- "06_2021"
## Load in list of all Argo floats and where they have been
# system2("gunzip", c("--force", "--keep", "input/ar_index_global_prof.txt.gz"))
prof_info <- read.delim(
  file = gzfile(paste0(argo_data_location, "ar_index_global_prof.txt.gz"), "r"), skip = 8, header = T,
  sep = ",", stringsAsFactors = FALSE
)

prof_info$float <- sapply(strsplit(prof_info$file, "/"), function(x) x[2])

float_data <- readRDS(paste0("temp/data/traj_data_", date, ".RDS"))
floats_of_interest <- as.character(float_data[["float"]])

# limit to area used in Chamberlain et al
prof_info_SO <- prof_info[prof_info$float %in% floats_of_interest, ]

# further reduce profiles
prof_info_SO <- prof_info_SO[(prof_info_SO$longitude > -80 &
  prof_info_SO$longitude < 80) | is.na(prof_info_SO$longitude), ]

# Load information from each file
data_list <- list()
for (i in 1:nrow(prof_info_SO)) {
  file_name <- paste0(argo_data_location, prof_info_SO$file[i])
  if (file.exists(file_name)) {
    file <- nc_open(file_name)
  } else {
    file_name2 <- gsub("R", "D", file_name)
    file <- nc_open(file_name2)
  }

  lat <- ncvar_get(file, "LATITUDE")[1]
  long <- ncvar_get(file, "LONGITUDE")[1]
  pos_qc <- strsplit(ncvar_get(file, "POSITION_QC"), "")[[1]][1]
  pos_sys <- ncvar_get(file, "POSITIONING_SYSTEM")[1]
  day <- ncvar_get(file, "JULD")[1]
  day_ref <- ncvar_get(file, "REFERENCE_DATE_TIME")[1]
  day_qc <- strsplit(ncvar_get(file, "JULD_QC"), "")[[1]][1] # quality control on day
  float <- as.numeric(ncvar_get(file, "PLATFORM_NUMBER"))[1] # float number
  cycle <- ncvar_get(file, "CYCLE_NUMBER")[1] # cycle number increments by 1 for each profile float collects
  mode <- substr(ncvar_get(file, "DATA_MODE")[1], 1, 1)
  if (mode %in% c("D", "A")) {
    if (length(dim(ncvar_get(file, "PRES_ADJUSTED"))) == 2) {
      pressure <- ncvar_get(file, "PRES_ADJUSTED")[, 1]
      temperature <- ncvar_get(file, "TEMP_ADJUSTED")[, 1]
      salinity <- ncvar_get(file, "PSAL_ADJUSTED")[, 1]
      pressure_qc <- strsplit(ncvar_get(file, "PRES_ADJUSTED_QC")[1], "")[[1]]
      temperature_qc <- strsplit(ncvar_get(file, "TEMP_ADJUSTED_QC")[1], "")[[1]]
      salinity_qc <- strsplit(ncvar_get(file, "PSAL_ADJUSTED_QC")[1], "")[[1]]
      pres_adj_error <- ifelse(is.na(ncvar_get(file, "PRES_ADJUSTED_ERROR")[, 1]),
        Inf, ncvar_get(file, "PRES_ADJUSTED_ERROR")[, 1]
      )
    } else {
      pressure <- ncvar_get(file, "PRES_ADJUSTED")
      temperature <- ncvar_get(file, "TEMP_ADJUSTED")
      salinity <- ncvar_get(file, "PSAL_ADJUSTED")
      pressure_qc <- strsplit(ncvar_get(file, "PRES_ADJUSTED_QC"), "")[[1]]
      temperature_qc <- strsplit(ncvar_get(file, "TEMP_ADJUSTED_QC"), "")[[1]]
      salinity_qc <- strsplit(ncvar_get(file, "PSAL_ADJUSTED_QC"), "")[[1]]
      pres_adj_error <- ifelse(is.na(ncvar_get(file, "PRES_ADJUSTED_ERROR")[1]),
        Inf, ncvar_get(file, "PRES_ADJUSTED_ERROR")[1]
      )
    }
  } else {
    if (length(dim(ncvar_get(file, "PRES"))) == 2) {
      pressure <- ncvar_get(file, "PRES")[, 1]
      temperature <- ncvar_get(file, "TEMP_ADJUSTED")[, 1]
      salinity <- ncvar_get(file, "PSAL")[, 1]
      pressure_qc <- strsplit(ncvar_get(file, "PRES_QC")[1], "")[[1]]
      temperature_qc <- strsplit(ncvar_get(file, "TEMP_QC")[1], "")[[1]]
      salinity_qc <- strsplit(ncvar_get(file, "PSAL_QC")[1], "")[[1]]
      pres_adj_error <- ifelse(is.na(ncvar_get(file, "PRES_ADJUSTED_ERROR")[, 1]),
        Inf, ncvar_get(file, "PRES_ADJUSTED_ERROR")[, 1]
      )
    } else {
      pressure <- ncvar_get(file, "PRES")
      temperature <- ncvar_get(file, "TEMP")
      salinity <- ncvar_get(file, "PSAL")
      pressure_qc <- strsplit(ncvar_get(file, "PRES_QC"), "")[[1]]
      temperature_qc <- strsplit(ncvar_get(file, "TEMP_QC"), "")[[1]]
      salinity_qc <- strsplit(ncvar_get(file, "PSAL_QC"), "")[[1]]
      pres_adj_error <- ifelse(is.na(ncvar_get(file, "PRES_ADJUSTED_ERROR")[1]),
        Inf, ncvar_get(file, "PRES_ADJUSTED_ERROR")[1]
      )
    }
  }
  nc_close(file)

  # if (cycle == 0 | sum(pres_adj_error >= 20) > 0) {
  #   data_list[[i]] <- NULL
  # }
  year <- substr(day_ref, 1, 4)
  month <- substr(day_ref, 5, 6)
  day_ref_final <- substr(day_ref, 7, 8)
  hours <- substr(day_ref, 9, 10)
  minutes <- substr(day_ref, 11, 12)
  seconds <- substr(day_ref, 13, 14)
  date_time <- as.POSIXct(day * 60 * 60 * 24,
    origin = as.POSIXct(ISOdatetime(
      year = year, month = month, day = day_ref_final,
      hour = hours, min = minutes,
      sec = seconds, tz = "GMT"
    )),
    tz = "GMT"
  )
  date_use <- as.Date(date_time)

  df_return <- data.frame(pressure, temperature, salinity,
    float, cycle, lat, long, pos_qc, pos_sys, day, day_qc,
    pressure_qc, temperature_qc, salinity_qc, pres_adj_error,
    reference = day_ref, date = date_use, date_time
  )
  data_list[[i]] <- df_return[!is.na(df_return$lat) & df_return$day_qc != 4, ]
  # df_return <- df_return[pressure_qc %in% c(1,2) & temperature_qc %in% c(1,2) &
  #                          salinity_qc %in% c(1,2),]
  # data_list[[i]] <- df_return
  if (i %% 50 == 0) {
    print(i)
  }
}
save(data_list, file = paste0("temp/data/prof_data_prelim_", date, ".RData"))
load(paste0("temp/data/prof_data_prelim_", date, ".RData"))
prof_data <- dplyr::bind_rows(data_list[1:10000])
prof_data2 <- dplyr::bind_rows(data_list[10001:20000])
prof_data3 <- dplyr::bind_rows(data_list[20001:30000])
prof_data4 <- dplyr::bind_rows(data_list[30001:40000])
prof_data5 <- dplyr::bind_rows(data_list[40001:length(data_list)])
prof_data_test <- dplyr::bind_rows(
  prof_data, prof_data2, prof_data3, prof_data4,
  prof_data5
)
prof_data <- prof_data_test
dim(prof_data)
prof_data$pos_sys <- as.factor(prof_data$pos_sys)
prof_data$float <- as.factor(prof_data$float)
prof_data$cycle <- as.factor(prof_data$cycle)
prof_data$pos_qc <- as.factor(prof_data$pos_qc)
prof_data$day_qc <- as.factor(prof_data$day_qc)
prof_data$pressure_qc <- as.factor(prof_data$pressure_qc)
prof_data$temperature_qc <- as.factor(prof_data$temperature_qc)
prof_data$salinity_qc <- as.factor(prof_data$salinity_qc)

prof_data <- prof_data[(prof_data$lat < -55 & prof_data$long < 35 & # limit to certain area, otherwise too large of files
  prof_data$long > -65) | is.na(prof_data$lat), ]

saveRDS(prof_data, file = paste0("temp/data/prof_data_", date, ".RDS"))


date <- "06_2021"
library(tidyverse)

float_data <- readRDS(paste0("temp/data/traj_data_", date, ".RDS"))
prof_data <- readRDS(paste0("temp/data/prof_data_", date, ".RDS"))

prof_data_unique <- prof_data %>%
  mutate(profile_unique = paste(float, cycle, sep = "_")) %>%
  filter(!duplicated(profile_unique)) %>%
  mutate(
    float = as.double(as.character(float)),
    cycle = as.double(as.character(cycle)),
    lat = as.double(as.character(lat)),
    long = as.double(as.character(long)),
    day = as.double(as.character(day))
  )
data_combined <- right_join(float_data, prof_data_unique,
  by = c("float", "cycle")
)

data_combined %>%
  filter(!is.na(ice_det)) %>%
  dplyr::select(float, cycle, long.y, lat.y, ice_det) %>%
  summary(ice_det)

ggplot(data = data_combined, aes(x = long.y, y = lat.y, color = ice_det)) +
  geom_point(size = .2)

ggplot(data = data_combined, aes(x = long.y, y = lat.y, color = pos_qc.y == 8 & !ice_det)) +
  geom_point(size = .2) +
  labs(color = "Missing\nLocation,\nNo Ice\nDetected") +
  theme(legend.position = "bottom")

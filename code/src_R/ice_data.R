library(tidyverse)
library(raster)
library(rgdal)
library(lubridate)
library(parallel)
library(rhdf5)

get_file_name <- function(date, verbose=FALSE) {
  year <- substr(date, 1, 4)
  month <- substr(date, 6, 7)
  day <- substr(date, 9, 10)
  month_name <- month.abb[as.numeric(month)]
  file_start <- "https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02135/south/daily/geotiff/"
  file_name <- paste0(
    file_start, year, "/", month, "_", month_name, "/",
    "S_", year, month, day, "_concentration_v3.0.tif"
  )
  return(file_name)
}



return_ice <- function(file, date, out_folder, long_step=0.25, lat_step=0.125) {
  imported_raster <- raster(file)
  projection(imported_raster) <- "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs "
  new_xy <- raster::projectRaster(
    from = imported_raster, res = c(long_step, lat_step),
    crs = "+proj=longlat +datum=WGS84 +no_defs ", method = "ngb"
  )
  new_xy <- as.data.frame(new_xy, xy = T)
  colnames(new_xy) <- c("long", "lat", "concentration")
  df <- new_xy %>%
    filter(long > -80, long < 30, lat < -50, lat > -75) %>%
    mutate(date = date) %>%
    mutate(days = as.integer(date))
  
  h5file <- sprintf("%s/%s.h5", out_folder, date)
  if (file.exists(h5file)) {
    file.remove(h5file)
  }
  for (nm in c("long", "lat", "concentration", "days")) {
      h5write(df[[nm]], h5file, nm)
  }
  return(NULL)
}

grid_dates <- c(
  seq(ymd("2002-02-22"), ymd("2020-02-19"), by = "1 day"),
  seq(ymd("2020-02-22"), ymd("2021-06-10"), by = "1 day")
)

file_names <- sapply(grid_dates, get_file_name)
make_args <- sprintf("URL=%s FILE_NAME=%s", file_names, str_sub(file_names, -33))
write_lines(make_args, "temp/data/ice_data_tiffs/file_list")

## Load all available dates
tif_files <- list.files("temp/data/ice_data_tiffs", "\\.tif$")
dates <- ymd(str_sub(tif_files, 3, 10))
tif_paths <- str_c("temp/data/ice_data_tiffs/", tif_files)

out_folder <- "temp/data/ice_data_concentration_h5"
dir.create(out_folder)
return_ice(tif_paths[1], dates[1], out_folder)

ice_data <- parallel::mcMap(return_ice, file=tif_paths, date=dates, out_folder=out_folder, mc.cores=20) %>% 
  bind_rows() %>%
  as_tibble()


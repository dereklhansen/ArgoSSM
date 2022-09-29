library(tidyverse)
library(rhdf5)
daily_ice_data <- read_rds(file = "temp/data/daily_ice_data.RDS") %>%
    as_tibble

unique_long <- unique(daily_ice_data$long)
unique_lat <- unique(daily_ice_data$lat)
unique_days <- as.integer(unique(daily_ice_data$date))

dlong <- diff(unique_long)
dlat <- diff(unique_lat)

n_long <- length(unique_long)
n_lat <- length(unique_lat)
n_days <- length(unique_days)

as.integer

extent <- ifelse(is.na(daily_ice_data$extent), 99, daily_ice_data$extent)
days <- as.integer(daily_ice_data$date)

for (nm in c("long", "lat", "extent", "days")) {
    h5write(environment()[[nm]], "temp/data/daily_ice_data.h5", nm)
}


# plyr::daply(daily_ice_data, .variables = c("long", "lat", "date"), .parallel = TRUE)
 

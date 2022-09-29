library(tidyverse)
library(lubridate)
library(rhdf5)
library(foreach)
library(doParallel)
library(parallel)
registerDoParallel()

models <- c("linearinterp", "kalman", "smc2", "smc2_ice", "smc2_pv")
# models <- c("linearinterp", "kalman", "smc2")
models_nicenames <- c("RW", "AR", "AR+Ice+PV", "AR+Ice", "AR+PV")
# models_nicenames <- c("RW", "AR", "ArgoSSM")
float_chamberlain_tbl <- read_rds("output/float_chamberlain_tbl.rds")
holdouts_under_ice <- read_rds("output/holdouts_under_ice.rds")

holdouts_all <- bind_rows(map(holdouts_under_ice, bind_rows, .id="holdout"), .id = "float") %>%
  #filter(float %in% floats_of_interest) %>%
  select(float, holdout) %>%
  unique %>%
  mutate(models = list(c("smc2", "smc2_ice", "smc2_pv", "kalman", "linearinterp", "pvinterp1", "pvinterp2"))) %>%
  unnest(cols = models)

holdouts_tbl_for_msekm <- holdouts_all

holdout_mse_tbl <- plyr::adply(holdouts_tbl_for_msekm, 1, function(x) {
  mse_km <- try(h5read(sprintf("output/floats_results/%s/%s/%s.h5", x$models, x$float, x$holdout), "holdout_mse_km"))
  if (is.character((mse_km))) {
    mse_km <- NA_real_
  }
  return(tibble(mse_km=mse_km))
}) %>% 
  as_tibble
holdout_mse_tbl %>% 
  #filter(models %in% c("smc2", "kalman")) %>%
  filter(models %in% c("smc2", "smc2_ice", "kalman", "linearinterp", "pvinterp1", "pvinterp2")) %>%
  group_by(float, holdout) %>%
  #filter(all(!is.na(mse_km) || (models %in% c("pvinterp1", "pvinterp2", "linearinterp")))) %>%
  filter(all(!is.na(mse_km))) %>%
  #filter(holdout %in% c("1")) %>%
  ungroup() %>%
  group_by(models) %>% 
  summarize(rmse = sqrt(mean(mse_km)), med = sqrt(median(mse_km, na.rm=T))) %>% arrange(-rmse)

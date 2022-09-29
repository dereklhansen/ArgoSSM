library(tidyverse)
library(lubridate)
library(rhdf5)
library(foreach)
library(doParallel)
library(parallel)
library(gridExtra)
registerDoParallel()

models <- c("linearinterp", "pvinterp2", "kalman", "smc2", "smc2_ice", "smc2_pv")
# models <- c("linearinterp", "kalman", "smc2")
models_nicenames <- c("RW", "PV Interp.", "AR", "AR+Ice+PV", "AR+Ice", "AR+PV")
# models_nicenames <- c("RW", "AR", "ArgoSSM")
float_chamberlain_tbl <- read_rds("output/float_chamberlain_tbl.rds")
holdouts_under_ice <- read_rds("output/holdouts_under_ice.rds")
chamberlain_holdouts_under_ice <- holdouts_under_ice$`5901717`
holdouts <- 1:5

float_chamberlain_holdout_tbls <- map(
  holdouts, 
  ~mutate(
    float_chamberlain_tbl, 
    t = 1:n(),
    heldout = 1:n() %in% chamberlain_holdouts_under_ice[[.x]]$holdout_idx,
    near_heldout = 1:n() %in% (min(chamberlain_holdouts_under_ice[[.x]]$holdout_idx) - 5):(max(chamberlain_holdouts_under_ice[[.x]]$holdout_idx) + 5)
    )
  )
chamberlain_holdout_tbl <- bind_rows(float_chamberlain_holdout_tbls, .id="holdout") %>%
  group_by(holdout, float) 

holdouts_tbl <- cross_df(list(
  model = discard(models, ~str_detect(.x, "pvinterp")),
  holdout = holdouts
))

pv_interp_tbl <- cross_df(list(
  model = c("pvinterp2"),
  holdout = holdouts
))

pv_predictions <- plyr::adply(pv_interp_tbl, 1, function(x) {
  pred <- h5read(sprintf("output/float_chamberlain_holdouts/%s/5901717/%d.h5", x$model, x$holdout), "pred")
  dimnames(pred) <- list(c("long", "lat"), unique(float_chamberlain_tbl$t))
  plyr::adply(pred, c(2), .id = "t")
}) %>%
  as_tibble()


path_clusters1 <- plyr::adply(holdouts_tbl, 1, function(x) {
  cluster_sizes <- h5read(sprintf("output/float_chamberlain_holdouts/%s/5901717/%d.h5", x$model, x$holdout), "cluster_sizes")
  cluster_sizes_tbl <- tibble(cluster = as.factor(seq_along(cluster_sizes)), size = cluster_sizes) %>%
    mutate(prop = size/sum(size))
  
  clusters <- h5read(sprintf("output/float_chamberlain_holdouts/%s/5901717/%d.h5", x$model, x$holdout), "clusters") %>%
    aperm(c(2, 1, 3))
  dimnames(clusters) <- list(c("long", "lat"), 1:27, unique(float_chamberlain_tbl$t))
  clusters_tbl <- plyr::adply(clusters, c(2, 3), .id = c("cluster", "t"))
  
  return(inner_join(clusters_tbl, cluster_sizes_tbl, by = c("cluster")))
}) %>% 
  as_tibble

path_clusters <- path_clusters1 %>%
  bind_rows(mutate(pv_predictions, cluster = "1", prop=1, size=1)) %>%
  mutate(holdout = as.factor(holdout)) %>%
  mutate(t = as.integer(t)) %>%
  mutate(model = factor(model, models, models_nicenames)) %>%
  inner_join(
    select(chamberlain_holdout_tbl, holdout, t, heldout, near_heldout),
    c("holdout", "t"))



model_colors <- c(
  `AR+Ice+PV` = "#619CFF",
  `AR+Ice` = "#619CFF",
  `AR+PV` = "#619CFF",
  `AR` = "#00BA38",
  `RW` = "#F8766D",
  `PV Interp.` = "#F8766D",
  `PV Interp. (2)` = "#F8766D"
)
path_means <- path_clusters %>% 
  group_by(model, holdout, t, heldout, near_heldout) %>% 
  summarize(long = sum(long*prop), lat = sum(lat*prop))

PV_300 <- read.csv("temp/data/PV300_w_est.csv", header = F)
colnames(PV_300) <- c("long", "lat", "gradlong", "gradlat", "PV", "depth")
pv_300 <- PV_300
make_pv_for_lims <- function(plot) {
  plot_build <- ggplot_build(plot)
  x_scale <- plot_build$layout$panel_params[[1]]$x.range
  y_scale <- plot_build$layout$panel_params[[1]]$y.range
  pv_300_trunc <- pv_300[pv_300$long > x_scale[1] - .3 & pv_300$long < x_scale[2] +.3 &
                                     pv_300$lat > y_scale[1] - 0.2 & pv_300$lat < y_scale[2] + 0.2, ]
  set_bins <- ggplot_build(ggplot(path_clusters, aes(x = long, y = lat)) +
                             geom_contour(
                               data = pv_300_trunc[pv_300_trunc$PV > quantile(pv_300$PV, .05, na.rm = T), ], aes(x = long, y = lat, z = PV),
                               alpha = .2, size = .25, bins = 20, color = "black"
                             ))
  levels <- unique(set_bins$data[[1]]$level)
  gc <- geom_contour(
    data = pv_300_trunc, aes(x = long, y = lat, z = PV),
    alpha = .4, size = .25, breaks = levels, color = "gray50"
  )
  return(plot + gc)
}

## Make different plots
plot_float_chamberlain_mean <- map(holdouts, function(h) {
  g <- ggplot(filter(path_means, near_heldout, holdout==h), aes(x = long, y = lat)) +
    geom_path(aes(color = model)) +
    geom_point(
      data = filter(path_clusters, heldout, holdout==h),
      aes(group = cluster, color = model, alpha = prop, shape=heldout)) +
    geom_point(
      data = filter(chamberlain_holdout_tbl, holdout==h, near_heldout, pos_qc %in% c(1,2)),
      aes(shape = heldout)) +
    coord_map() +
    scale_color_manual(values = model_colors) +
    facet_wrap(.~model) +
    #ggtitle("Predicted trajectories of Argo float") +
    xlab("Longitude") +
    ylab("Latitute") +
    theme_bw() +
    guides(color = FALSE, alpha = FALSE, shape=FALSE)
  make_pv_for_lims(g)
}) 

iwalk(
  plot_float_chamberlain_mean, 
  ~ggsave(
    sprintf("output/plots/chamberlain_holdouts/chamberlain_holdout_mean_%02d.png", .y),
    .x,
    height = 3,
    width = 9,
  )
)

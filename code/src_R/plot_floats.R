## Generate plots for section 3.2 ("Analysis of Float Trajectories in Southern Ocean)
## TODO: Move chamber
library(rhdf5)
library(tidyverse)
library(stringi)

library(foreach)
library(doParallel)
library(parallel)
library(latex2exp)
registerDoParallel()

OUTDIR <- "output/floats_results_predict/smc2/"

float_tbl <- read_rds("output/float_tbl.rds") %>%
  group_by(float) %>%
  mutate(t_new = 1:n()) %>%
  ungroup()

read_smc2_parameters <- function(float_id) {
  file <- sprintf("%s/%s.h5", OUTDIR, float_id)
  params <- h5read(file, "param_mat")
  paramnames <- h5read(file, "param_names")
  rownames(params) <- paramnames
  return(params)
}

read_clusters <- function(idx, file = "output/floats_results.h5") {
  model_names <- c("kalman_lininterp", "kalman", "smc2")
  map(model_names, function(model) {
    x <- h5read(file, sprintf("%s/%s/clusters", idx, model)) %>%
      aperm(c(2, 1, 3))
    dimnames(x) <- list(c("long", "lat"), 1:27, NULL)
    x
  }) %>%
    set_names(model_names) %>%
    plyr::ldply(plyr::adply, .margins = c(2, 3), .parallel = TRUE) %>%
    rename(model = .id) %>%
    as_tibble() %>%
    rename(cluster = X1) %>%
    mutate(cluster = as.integer(as.character(cluster))) %>%
    # left_join(path_cluster_sizes, by = c("model", "cluster")) %>%
    mutate(model = factor(model, model_names, c("RW", "AR", "ArgoSSM"))) %>%
    filter(model == "ArgoSSM") %>%
    mutate(t_new = 1:n())
}

read_uncertainties <- function(idx, file = "output/floats_results.h5") {
  mean_dist <- h5read(file, sprintf("%s/smc2/distance_mean_t", idx)) %>%
    t() %>%
    as.data.frame() %>%
    set_names("mean_dist") %>%
    as_tibble()
  quants <- h5read(file, sprintf("%s/smc2/distance_qs_t_quantiles", idx))
  quant_dist <- h5read(file, sprintf("%s/smc2/distance_qs_t", idx)) %>%
    t()
  colnames(quant_dist) <- quants
  quant_dist <- as.data.frame(quant_dist) %>%
    as_tibble()
  tbl <- bind_cols(mean_dist, quant_dist) %>%
    mutate(t_new = 1:n()) %>%
    select(t_new, everything())
  return(tbl)
}

float_tbl <- read_rds("output/float_tbl.rds") %>%
  group_by(float) %>%
  mutate(t_new = 1:n())
unique_floats <- unique(float_tbl$float)

params <- map(unique_floats, read_smc2_parameters) %>%
  set_names(unique_floats) %>%
  map(~ as_tibble(t(.x))) %>%
  map(~ mutate(.x, idx = 1:n())) %>%
  bind_rows(.id = "float") %>%
  mutate(across(starts_with("σ"), exp)) %>%
  mutate(across(starts_with("ice_"), exp)) %>%
  mutate(γ = exp(γ)) %>%
  mutate(α = exp(α))
params



####
# GAMMA PLOT
###
gamma_tbl <- params %>%
  group_by(float) %>%
  summarize(sd = sd(1/γ), sigma2_gamma = mean(1/γ)) %>%
  arrange(-sigma2_gamma) %>%
  mutate(float = factor(float, levels(alpha_tbl$float)))
gamma_tbl

gamma_dist_plot <- ggplot(gamma_tbl, aes(y = float, x = sigma2_gamma)) +
  geom_point() +
  geom_segment(aes(x = sigma2_gamma - sd, xend = sigma2_gamma + sd, yend=float)) +
  theme_bw() +
  xlab(TeX("$\\sigma^2_{{PV}}")) +
  ylab("Float")
gamma_dist_plot

ggsave("output/gamma_posterior.png", gamma_dist_plot, width = 4, height = 5)

###
# ALPHA PLOT
# ###
alpha_tbl <- params %>%
  group_by(float) %>%
  summarize(sd = sd(α), α = mean(α)) %>%
  arrange(-α) %>%
  #filter(float %in% floats_2015[c(2, 4, 6, 8, 10)]) %>%
  mutate(float = factor(float, float))
alpha_tbl

alpha_dist_plot <- ggplot(alpha_tbl, aes(y = float, x = α)) +
  geom_point() +
  geom_segment(aes(x = α - sd, xend = α + sd, yend=float)) +
  theme_bw() +
  xlab(TeX("$\\alpha")) +
  ylab("Float")
alpha_dist_plot

ggsave("output/alpha_posterior.png", alpha_dist_plot, width = 4, height = 5)

confusion_tbl <- params %>%
  select(float, ice_tpr, ice_tnr) %>%
  gather(-float, key="measure", value="prob") %>%
  group_by(float, measure) %>%
  summarize(sd = sd(prob), prob = mean(prob))
confusion_tbl

tpr_tbl <- filter(confusion_tbl, measure == "ice_tpr") %>%
  arrange(prob)

ice_prob_tpr_plot <- confusion_tbl %>%
  filter(measure == "ice_tpr") %>%
  mutate(float = factor(float, levels=(tpr_tbl$float))) %>%
  ggplot(aes(y = float, x=prob)) +
  geom_point() +
  geom_segment(aes(x = prob - sd, xend=prob+sd, yend=float)) +
  theme_bw() +
  xlab("Ice TPR") +
  ylab("Float")
  #facet_wrap(~measure)
ggsave("output/ice_prob_tpr_posterior.png", ice_prob_tpr_plot, width = 4, height = 5)

ice_prob_tnr_plot <- confusion_tbl %>%
  filter(measure == "ice_tnr") %>%
  mutate(float = factor(float, levels=(tpr_tbl$float))) %>%
  ggplot(aes(y = float, x=prob)) +
  geom_point() +
  geom_segment(aes(x = prob - sd, xend=prob+sd, yend=float)) +
  theme_bw() +
  xlab("Ice TNR") +
  ylab("Float")
#facet_wrap(~measure)
ggsave("output/ice_prob_tnr_posterior.png", ice_prob_tnr_plot, width = 4, height = 5)
###
# Plot of all float clusters
###

cluster_data <- map(unique_floats, read_clusters)

floats_2015 <- float_tbl %>%
  filter(date > "2015-03-01", date < "2015-06-01") %>%
  {
    unique(.$float)
  }

highlight_floats <- c(
  "5903614"
)

cluster_tbl <- set_names(cluster_data, unique_floats) %>%
  bind_rows(.id = "float") %>%
  filter(model == "ArgoSSM") %>%
  unite("float_cluster", float, cluster, remove = FALSE) %>%
  inner_join(gamma_tbl, by = "float") # %>%
# filter(float %in% floats_2015[c(2, 4, 6, 8, 10)])

# filter(float == "5901716")

all_floats <- ggplot(
  filter(cluster_tbl, float %in% floats_2015[c(2, 4, 6, 8, 10)]),
  aes(x = long, y = lat, group = float_cluster, color = float)
) +
  geom_point(
    inherit.aes = FALSE,
    size = 0.3,
    alpha = 0.25,
    data = filter(
      float_tbl, pos_qc %in% c(1, 2), float %in% floats_2015,
    ),
    aes(x = long, y = lat)
  ) +
  # geom_path(
  #   inherit.aes=FALSE,
  #   linetype = "dashed",
  #   alpha=0.05,
  #   data = filter(cluster_tbl, float %in% floats_2015),
  #   aes(x=long, y=lat, group=float_cluster)
  # ) +
  geom_path(alpha = 0.4) +
  geom_point(
    inherit.aes = FALSE,
    size = 0.3,
    data = filter(
      float_tbl, pos_qc %in% c(1, 2), float %in% floats_2015[c(2, 4, 6, 8, 10)]
    ),
    aes(x = long, y = lat, color = float)
  ) +
  theme_bw() +
  guides(color = FALSE) +
  coord_quickmap(
    expand = F,
    xlim = c(-59, 20), ylim = c(-75, -58)
  ) +
  geom_polygon(
    data = map_data("world"), aes(x = long, y = lat, group = group),
    fill = "white", color = "black", size = .2,
  ) +
  xlab("Longitude") +
  ylab("Latitude")

ggsave("output/all_floats_trajectories.png", all_floats, width = 5, height = 3)

uncertainty_data <- map(unique_floats, read_uncertainties) %>%
  set_names(unique_floats) %>%
  bind_rows(.id = "float") %>%
  left_join(float_tbl, by = c("float", "t_new"))

g <- ggplot_build(all_floats)
colors <- set_names(unique(g$data[[2]][["colour"]]), sort(floats_2015[c(2, 4, 6, 8, 10)]))

uncertainty_data_2015 <- uncertainty_data %>%
  filter(float %in% floats_2015[c(2, 4, 6, 8, 10)])
# gather(mean_dist, `0.005`, `0.01`, `0.05`, `0.1`, `0.25`, `0.5`, `0.75`, `0.9`, `0.95`, `0.99`, `0.995`, key="quantile", value="uncertainty_km") %>%
# filter(quantile==c("0.5", "0.95", "0.99", "0.995"))

uncertainty_2015_plot <- ggplot(uncertainty_data_2015, aes(x = date, group = float)) +
  facet_grid(float ~ .) +
  theme_bw() +
  guides(color = FALSE) +
  scale_color_manual(values = colors) +
  guides(color = FALSE, fill = FALSE) +
  geom_line(aes(y = `0.5`, color = float)) +
  geom_ribbon(aes(ymin = `0.05`, ymax = `0.95`, fill = float), alpha = 0.25) +
  ylab("Uncertainty (km)") +
  xlab("Date")

ggsave("output/uncertainty_2015_plot.png", uncertainty_2015_plot, width = 4, height = 6)

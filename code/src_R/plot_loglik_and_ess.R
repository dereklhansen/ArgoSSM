library(rhdf5)
library(tidyverse)
library(stringi)
library(fields)
map <- purrr::map

read_loglik_comparison <- function(est_method, idx, file = "output/floats_results/compare_ess.h5") {
  h5read(file, sprintf("%03d/1/%s/logliks", idx, est_method))
}

read_ess_comparison <- function(est_method, idx, file = "output/compare_ess.h5") {
  h5read(file, sprintf("%03d/1/%s/filtering_ess", idx, est_method))
}


models = c("smc2", "smc2_pv")
# models = c("smc2", "smc2_naive")

holdouts_under_ice <- read_rds("output/holdouts_under_ice.rds")
holdouts_cross <- cross_df(list(
  model = models,
  float = names(holdouts_under_ice),
  holdout = c(1)
))


holdout_mse_tbl <- plyr::adply(holdouts_cross, 1, function(x) {
  lls <- 
  tryCatch(
    {
      h5read(
        "output/floats_results/compare_ess.h5",
        sprintf("%s/%d/%s/logliks", x$float, x$holdout, x$model)
      )
    },
    error= function(cond) {
      NA
    })
  return(tibble(lls = lls))
}) %>%
  as_tibble

x = holdout_mse_tbl %>%
# filter(float == "5901717") %>%
  group_by(model, float, holdout) %>%
  summarize(m = mean(lls), s = sd(lls))# %>%
  #filter(model == "smc2_pv")

### Loglik comparison
models = c("smc2")
# models = c("smc2", "smc2_naive")
loglik_est_tbl <- purrr::cross_df(list(est_method = c("smc2"), idxs = 1:210)) %>%
  plyr::adply(.margins = 1, function(x) {
    logliks <- read_loglik_comparison(x$est_method, x$idxs)
    tibble(loglik = logliks) %>% mutate(i = 1:n())
  }) %>%
  as_tibble()

loglik_var_tbl <- loglik_est_tbl %>%
  group_by(est_method, idxs) %>%
  summarize(sd_ll = sd(loglik))

loglik_var_tbl %>%
  mutate(est_method = factor(est_method,
    levels = c("argonaive", "argossm"),
    labels = c("Bootstrap Filter", "ArgoSSM")
  )) %>%
  ggplot(aes(x = idxs, y = sd_ll, color = est_method)) +
  geom_line() +
  geom_ribbon(aes(ymin = 1e-5, ymax = sd_ll, fill = est_method)) +
  scale_y_log10()

loglik_boxplot <- x %>%
  mutate(model = factor(
    model, levels = c("smc2", "smc2_pv"),
    labels = c("AR+Ice+PV", "AR+PV"),
  )) %>%
  # mutate(est_method = factor(est_method,
  #   levels = c("argonaive", "argossm"),
  #   labels = c("Bootstrap Filter", "ArgoSSM")
  # )) %>%
  ggplot(aes(y = s, x = model)) +
  geom_boxplot() +
  scale_y_log10() +
  geom_hline(yintercept = 1.0, linetype = "dotted") +
  xlab("SMC Proposal") +
  ylab("Standard Deviation of log-likelihood") +
  theme_bw()
# theme(text = element_text(family = "IBM Plex Sans"))

ggsave("output/loglik_boxplot.png", loglik_boxplot, width = 3, height = 4)

holdouts <- read_rds("output/holdouts/holdouts_06_2021.rds")
truth_tbl <- plyr::ldply(holdouts, function(x) {
  long <- x$X[, 1]
  lat <- x$X[, 2]
  tibble(
    t = 1:length(long), long, lat,
    pos_qc = x$my_tbl$pos_qc, heldout = x$my_tbl$heldout, days = x$days, float = x$my_tbl$float
  ) %>%
    mutate(holdout_begin = min(which(heldout))) %>%
    mutate(holdout_end = max(which(heldout))) %>%
    mutate(days_from_start = days - days[holdout_begin - 1]) %>%
    mutate(days_from_end = days - days[holdout_end + 1])
}) %>%
  as_tibble() %>%
  rename(idxs = .id) %>%
  mutate(idxs = as.integer(idxs)) %>%
  rename(lat_truth = lat) %>%
  rename(long_truth = long)


last_obs <- truth_tbl %>%
  group_by(float, idxs) %>%
  filter(pos_qc %in% c("1", "2")) %>%
  summarize(t_min = min(t), t_max = max(t)) %>%
  select(float, idxs, t_min, t_max)

ess_tbl <- purrr::cross_df(list(est_method = c("argossm", "argonaive"), idxs = 1:210)) %>%
  left_join(last_obs, by = "idxs") %>%
  plyr::adply(.margins = 1, function(x) {
    ess <- read_ess_comparison(x$est_method, x$idxs)
    tibble(ess = ess[x$t_max - x$t_min + 1])
  }) %>%
  as_tibble() %>%
  group_by(float, est_method) %>%
  summarize(ess = mean(ess)) %>%
  mutate(est_method = factor(est_method,
    levels = c("argonaive", "argossm"),
    labels = c("Bootstrap Filter", "ArgoSSM")
  )) %>%
  mutate(float_id = as.factor(as.integer(float))) %>%
  mutate(float_id = fct_rev(float_id))



ggplot(ess_tbl, aes(x = float, y = ess, fill = est_method)) +
  geom_col(position = "dodge2")
ggplot(ess_tbl, aes(x = est_method, y = ess, color = est_method, fill = est_method)) +
  geom_violin() +
  scale_y_log10()
ggplot(ess_tbl, aes(x = est_method, y = ess, color = est_method)) +
  geom_boxplot() +
  scale_y_log10()
ggplot(ess_tbl, aes(x = ess, fill = est_method)) + #+ geom_histogram(binwidth=10)
  geom_density(bw = 10.0)

ess_tbl_test <- ess_tbl %>%
  plyr::adply(.margins = 1, function(x) {
    n_samples <- floor(x$ess)
    tbl <- tibble(i = seq(1, 1000, 1)) %>%
      mutate(effective_draw = ifelse(i < n_samples, TRUE, FALSE))
  }) %>%
  as_tibble()

p <- ggplot(ess_tbl_test, aes(x = float, y = i, alpha = effective_draw)) +
  facet_wrap(~est_method) +
  geom_point() +
  xlab(NULL) +
  ylab(NULL) +
  scale_x_discrete(labels = NULL, breaks = NULL) +
  guides(alpha = FALSE)

p <- ggplot(ess_tbl_test, aes(y = float, x = i, alpha = effective_draw)) +
  facet_wrap(~est_method) +
  geom_point() +
  xlab(NULL) +
  ylab(NULL) +
  scale_y_discrete() +
  guides(alpha = FALSE) +
  ggtitle("Effective Sample Size at final observation of each float", subtitle = "Each filled dot represents 10 effective samples.")


p2 <- ggplot(ess_tbl[with(ess_tbl, float %in% unique(float)[1:10]), ], aes(y = float_id, x = ess)) +
  facet_wrap(~est_method) +
  geom_col() +
  xlab("Number of SMC samples") +
  ylab(NULL) +
  # scale_y_discrete(labels = NULL) +
  guides(fill = FALSE) +
  theme_bw()
# theme(text = element_text(family = "IBM Plex Sans"))

ggsave("output/ess.png", p2, width = 6, height = 3)

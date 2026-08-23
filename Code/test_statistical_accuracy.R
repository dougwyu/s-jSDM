# Statistical accuracy validation: Mojo backend vs pure PyTorch
#
# Fits identical sjSDM models on synthetic communities under both compute
# backends and measures how accurately each recovers the known simulation
# ground truth. Two scenarios:
#
#   1. Environmental niche + species associations: communities simulated
#      via simulate_SDM() with known env coefficients and a known species
#      association matrix (the residual co-occurrence structure). Tests
#      recovery of env coefficients and of Sigma.
#   2. Spatial dispersal: communities driven by an autocorrelated latent
#      geographic field with known per-species loadings ("dispersal
#      limitation" signal), fitted with sjSDM's `spatial` term. Tests
#      whether fitted spatial coefficients track the true loadings.
#
# The two backends draw different Monte Carlo noise, so fits are not
# comparable estimate-by-estimate; the question is whether they recover
# the truth EQUALLY WELL.
#
# Usage:
#   Rscript Code/test_statistical_accuracy.R [n_rep]
#   default n_rep = 3 (each rep refits both backends in both scenarios;
#   expect roughly 10 minutes total)

stopifnot(requireNamespace("here", quietly = TRUE))
Sys.setenv(RETICULATE_PYTHON = here::here(
  "work/mojo-backend/reticulate-venv/bin/python"
))
Sys.setenv(TQDM_DISABLE = "1")  # silence python-side progress bars

devtools::load_all(here::here("sjSDM"), quiet = TRUE)
stopifnot(is_torch_available())

n_rep <- if (length(commandArgs(TRUE))) as.integer(commandArgs(TRUE)[[1]]) else 3L
set.seed(2026)

set_backend <- function(backend) {
  # Sys.setenv() is invisible to Python's os.environ (snapshotted at
  # interpreter startup), so switch the backend through reticulate.
  reticulate::py_run_string(sprintf(
    "import os; os.environ['SJSDM_MOJO_BACKEND'] = '%s'", backend
  ))
}

fit_common <- list(
  family = binomial("logit"),
  sampling = 400L,
  iter = 100L,
  step_size = 10L,
  learning_rate = 0.1,
  se = FALSE,
  verbose = FALSE
)

# ---- scenario 1: env coefficients + species associations -------------------
sim_niche <- function(n = 2000L, sp = 40L, k = 3L) {
  comm <- simulate_SDM(sites = n, species = sp, env = k, link = "logit")
  list(Y = comm$response,
       X = as.matrix(comm$env_weights),
       beta = t(comm$species_weights),   # species x env (no intercepts)
       Corr = comm$correlation)
}

fit_niche <- function(sim) {
  do.call(sjSDM, c(list(Y = sim$Y, env = sim$X), fit_common))
}

recover_niche <- function(fit, sim) {
  w <- getWeights(fit)$env[[1]]        # species x (env+1); col 1 = intercept
  s <- stats::cov2cor(getCov(fit))     # compare on the correlation scale
  ut <- upper.tri(sim$Corr)
  k <- ncol(sim$X)
  # per-covariate: correlation, attenuation slope of raw truth on raw
  # estimate (1 = unbiased spread), and standardized RMSE
  per_cov <- vapply(seq_len(k), function(j) {
    e <- scale(w[, j + 1]); tr <- scale(sim$beta[, j])
    c(cor = cor(e, tr),
      slope = coef(lm(sim$beta[, j] ~ w[, j + 1]))[2],
      rmse = sqrt(mean((e - tr)^2)))
  }, numeric(3))
  list(
    coef_cor = mean(per_cov[1, ]),
    coef_slope = mean(per_cov[2, ]),
    coef_rmse = mean(per_cov[3, ]),
    assoc_cor = cor(s[ut], sim$Corr[ut]),
    assoc_rmse = sqrt(mean((s[ut] - sim$Corr[ut])^2)),
    r2 = Rsquared(fit),
    loss = fit$history[[length(fit$history)]]
  )
}

# ---- scenario 2: spatial dispersal -----------------------------------------
sim_spatial <- function(n = 1200L, sp = 30L, k = 3L) {
  # One identifiable spatial axis: an autocorrelated latent field over the
  # study area (exponential distance decay), used directly as the single
  # `spatial` covariate. Each species has a known loading on that field.
  d <- as.matrix(stats::dist(matrix(runif(n * 2L), n)))
  K <- exp(-d / 0.3)
  x1 <- drop(chol(K) %*% rnorm(n))
  beta <- matrix(rnorm(sp * (k + 1L)), sp)
  b <- rnorm(sp)                                     # per-species spatial loadings
  S <- matrix(rnorm(sp * sp), sp)
  Sigma <- S %*% t(S) / sp + diag(0.5, sp)
  L <- t(chol(stats::cov2cor(Sigma)))
  Z <- matrix(rnorm(n * sp), n) %*% t(L)
  X <- matrix(rnorm(n * k), n)
  lp <- cbind(1, X) %*% t(beta) + 2 * outer(x1, b) + Z
  Y <- matrix(rbinom(length(lp), 1L, plogis(lp)), n)
  list(Y = Y, X = X, x1 = x1, beta = beta, b = b,
       Corr = stats::cov2cor(Sigma))
}

fit_spatial <- function(sim) {
  do.call(sjSDM, c(list(
    Y = sim$Y, env = sim$X,
    spatial = linear(data = data.frame(X1 = sim$x1), formula = ~ 0 + X1),
    biotic = bioticStruct(df = 4L)
  ), fit_common))
}

recover_spatial <- function(fit, sim) {
  w_env <- getWeights(fit)$env[[1]]
  w_sp <- getWeights(fit)$spatial[[1]][, 1]  # per-species loadings on the field
  s <- stats::cov2cor(getCov(fit))           # compare on the correlation scale
  ut <- upper.tri(sim$Corr)
  k <- ncol(sim$X)
  per_cov <- vapply(seq_len(k), function(j) {
    e <- scale(w_env[, j + 1]); tr <- scale(sim$beta[, j + 1])
    c(cor = cor(e, tr),
      slope = coef(lm(sim$beta[, j + 1] ~ w_env[, j + 1]))[2],
      rmse = sqrt(mean((e - tr)^2)))
  }, numeric(3))
  list(
    coef_cor = mean(per_cov[1, ]),
    coef_slope = mean(per_cov[2, ]),
    coef_rmse = mean(per_cov[3, ]),
    disp_cor = suppressWarnings(cor(scale(w_sp), scale(sim$b))),
    disp_slope = suppressWarnings(coef(lm(scale(sim$b) ~ scale(w_sp)))[2]),
    assoc_cor = cor(s[ut], sim$Corr[ut]),
    r2 = Rsquared(fit),
    loss = fit$history[[length(fit$history)]]
  )
}

# ---- run --------------------------------------------------------------------
res <- list(); i <- 0L
for (r in seq_len(n_rep)) {
  for (backend in c("0", "auto")) {          # "0" = torch, "auto" = Mojo
    set_backend(backend)
    tag <- if (backend == "0") "torch" else "mojo"
    message(sprintf("rep %d/%d [%s] niche ...", r, n_rep, tag))
    s1 <- sim_niche()
    m1 <- recover_niche(fit_niche(s1), s1)
    message(sprintf("rep %d/%d [%s] spatial ...", r, n_rep, tag))
    s2 <- sim_spatial()
    m2 <- recover_spatial(fit_spatial(s2), s2)
    i <- i + 1L
    res[[i]] <- list(tag = tag, rep = r, niche = m1, spatial = m2)
  }
}
set_backend("auto")

# ---- summarize --------------------------------------------------------------
tags <- vapply(res, `[[`, character(1), "tag")
as_df <- function(scen, keys) {
  do.call(rbind, lapply(keys, function(key) {
    vals <- vapply(res, \(x) x[[scen]][[key]], numeric(1))
    tv <- vals[tags == "torch"]; mv <- vals[tags == "mojo"]
    data.frame(metric = paste(scen, key, sep = "."),
               torch_mean = mean(tv), mojo_mean = mean(mv),
               worst_gap = max(abs(tv - mv)))
  }))
}

summary_tbl <- rbind(
  as_df("niche", c("coef_cor", "coef_slope", "coef_rmse",
                   "assoc_cor", "assoc_rmse", "r2")),
  as_df("spatial", c("coef_cor", "disp_cor", "disp_slope", "assoc_cor"))
)
print(summary_tbl, digits = 4)

# ---- pass/fail ---------------------------------------------------------------
gap_of <- function(scen, key) {
  vals <- vapply(res, \(x) x[[scen]][[key]], numeric(1))
  tv <- vals[tags == "torch"]; mv <- vals[tags == "mojo"]
  max(abs(tv - mv))
}
floor_of <- function(scen, key) {
  min(vapply(res, \(x) x[[scen]][[key]], numeric(1)))
}
ok <-
  gap_of("niche", "coef_cor") < 0.05 &&
  gap_of("niche", "assoc_cor") < 0.10 &&
  gap_of("spatial", "coef_cor") < 0.05 &&
  gap_of("spatial", "disp_cor") < 0.10 &&
  gap_of("spatial", "assoc_cor") < 0.10 &&
  floor_of("niche", "coef_cor") > 0.8 &&
  floor_of("niche", "assoc_cor") > 0.6 &&
  floor_of("spatial", "coef_cor") > 0.8 &&
  floor_of("spatial", "disp_cor") > 0.6

verdict <- if (ok)
  "PASS: backends recover environmental coefficients, species associations,\nand spatial dispersal equally well.\n" else
  "FAIL: backend accuracy gap exceeds tolerance -- investigate.\n"
cat(verdict)
flush(stdout())
quit(status = if (ok) 0 else 1)

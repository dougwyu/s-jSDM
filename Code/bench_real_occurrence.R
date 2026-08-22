# End-to-end backend benchmark on a bundled REAL occurrence dataset.
#
# Usage: Rscript Code/bench_real_occurrence.R <torch|auto> <butterflies|eucalypts> <epochs> <seed> <out.rds>

args <- commandArgs(trailingOnly = TRUE)
backend <- args[1]; ds <- args[2]
epochs <- as.integer(args[3]); seed <- as.integer(args[4]); outfile <- args[5]

Sys.setenv(RETICULATE_PYTHON = here::here("work/port-feasibility/.pixi/envs/default/bin/python"))
if (backend == "mojo") {
  Sys.setenv(SJSDM_MOJO_BACKEND = "1")
} else if (backend == "torch") {
  Sys.setenv(SJSDM_MOJO_BACKEND = "0")
} else {
  Sys.unsetenv("SJSDM_MOJO_BACKEND") # auto
}

devtools::load_all(here::here("sjSDM"), quiet = TRUE)
options(sjSDM.device = "cpu")

set.seed(seed)
if (ds == "butterflies") {
  data(butterflies, package = "sjSDM")
  Y <- as.matrix(butterflies$PA); X <- as.matrix(butterflies$env)
} else if (ds == "eucalypts") {
  data(eucalypts, package = "sjSDM")
  Y <- as.matrix(eucalypts$PA); X <- as.matrix(eucalypts$env)
} else stop("unknown dataset")

t0 <- Sys.time()
fit <- sjSDM(
  Y = Y, env = X,
  biotic = bioticStruct(df = 5L),
  family = stats::binomial("logit"),
  iter = epochs, step_size = 10L,
  learning_rate = 0.1, sampling = 400L, se = FALSE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
history <- as.numeric(fit$model$history)

saveRDS(
  list(backend = backend, dataset = ds, sites = nrow(Y), species = ncol(Y),
       epochs = epochs, seed = seed, elapsed = elapsed, history = history),
  outfile
)
cat(sprintf("%s %s (%dx%d): %.2fs, final loss %.6f\n",
            backend, ds, nrow(Y), ncol(Y), elapsed, history[length(history)]))

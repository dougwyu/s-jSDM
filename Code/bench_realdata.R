# End-to-end backend benchmark: simulate a community of given size, fit
# sjSDM under the selected backend, save per-epoch history and timing.
#
# Usage: Rscript Code/bench_realdata.R <torch|mojo> <sites> <species> <epochs> <seed> <out.rds>

args <- commandArgs(trailingOnly = TRUE)
backend <- args[1]; sites <- as.integer(args[2]); species <- as.integer(args[3])
epochs <- as.integer(args[4]); seed <- as.integer(args[5]); outfile <- args[6]

Sys.setenv(RETICULATE_PYTHON = here::here("work/port-feasibility/.pixi/envs/default/bin/python"))
if (backend == "mojo") Sys.setenv(SJSDM_MOJO_BACKEND = "1") else Sys.unsetenv("SJSDM_MOJO_BACKEND")

devtools::load_all(here::here("sjSDM"), quiet = TRUE)
options(sjSDM.device = "cpu")

set.seed(seed)
n_env <- 5L
X <- matrix(stats::rnorm(sites * n_env), sites, n_env)
beta <- matrix(stats::rnorm(species * (n_env + 1L)), species)
Sigma <- cov(matrix(stats::rnorm(species * species), species, species)) |> (\(m) m + diag(0.5, species))()
Z <- matrix(stats::rnorm(sites * species), sites, species) |> (\(m) m %*% t(Sigma))()
logit_p <- cbind(1, X) %*% t(beta) + Z
p <- 1 / (1 + exp(-logit_p))
Y <- matrix(stats::rbinom(length(p), 1, p), sites, species)

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
  list(backend = backend, sites = sites, species = species,
       epochs = epochs, seed = seed, elapsed = elapsed,
       history = history),
  outfile
)
cat(sprintf("%s %dx%d: %.2fs, final loss %.6f\n",
            backend, sites, species, elapsed, history[length(history)]))

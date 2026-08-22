# Trace torch RNG usage during a fixed-seed sjSDM fit under either backend.
# Usage: Rscript Code/trace_rng.R <torch|mojo> <out.log>

args <- commandArgs(trailingOnly = TRUE)
backend <- args[1]; logfile <- args[2]

Sys.setenv(RETICULATE_PYTHON = here::here("work/port-feasibility/.pixi/envs/default/bin/python"))
if (backend == "mojo") Sys.setenv(SJSDM_MOJO_BACKEND = "1") else Sys.unsetenv("SJSDM_MOJO_BACKEND")

devtools::load_all(here::here("sjSDM"), quiet = TRUE)
options(sjSDM.device = "cpu")

reticulate::py_run_string(sprintf("
import torch, hashlib, json, traceback
_logf = open('%s', 'w')
_orig_randn = torch.randn
_orig_seed = torch.manual_seed
def _seed_traced(seed):
    _logf.write(json.dumps({'op': 'manual_seed', 'seed': seed}) + chr(10)); _logf.flush()
    return _orig_seed(seed)
def _randn_traced(*a, **k):
    t = _orig_randn(*a, **k)
    stack = ''.join(traceback.format_stack(limit=6)[:-1]).replace(chr(10), ' | ')
    _logf.write(json.dumps({'op': 'randn', 'shape': list(t.shape),
                            'hash': hashlib.sha1(t.detach().numpy().tobytes()).hexdigest()[:12],
                            'stack': stack}) + chr(10))
    _logf.flush()
    return t
torch.randn = _randn_traced
torch.manual_seed = _seed_traced
", logfile))

set.seed(101)
sites <- 800L; species <- 30L; n_env <- 5L
X <- matrix(stats::rnorm(sites * n_env), sites)
beta <- matrix(stats::rnorm(species * (n_env + 1L)), species)
Sigma <- cov(matrix(stats::rnorm(species * species), species)) |> (\(m) m + diag(0.5, species))()
Z <- matrix(stats::rnorm(sites * species), sites) %*% t(Sigma)
p <- plogis(cbind(1, X) %*% t(beta) + Z)
Y <- matrix(stats::rbinom(length(p), 1, p), sites)

fit <- sjSDM(
  Y = Y, env = X, biotic = bioticStruct(df = 4L),
  family = stats::binomial("logit"),
  iter = 2L, step_size = 10L,
  learning_rate = 0.1, sampling = 50L, se = FALSE
)

cat(backend, "done; log:", logfile, "\n")

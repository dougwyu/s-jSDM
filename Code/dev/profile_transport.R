# Profile R<->Python transport overhead during an sjSDM fit.
#
# Decomposes end-to-end time into: Python-side fit() time, cumulative
# per-batch loss-call time (Mojo pipe round-trip or torch compute),
# post-fit logLik evaluation, and the residual R-side/dispatch overhead.
#
# Usage: Rscript Code/profile_transport.R <torch|mojo> <sites> <species> <epochs> <tag>

args <- commandArgs(trailingOnly = TRUE)
backend <- args[1]; sites <- as.integer(args[2])
species <- as.integer(args[3]); epochs <- as.integer(args[4]); tag <- args[5]

Sys.setenv(RETICULATE_PYTHON = here::here("work/mojo-backend/reticulate-venv/bin/python"))
if (backend == "mojo") Sys.setenv(SJSDM_MOJO_BACKEND = "1") else Sys.unsetenv("SJSDM_MOJO_BACKEND")

devtools::load_all(here::here("sjSDM"), quiet = TRUE)
options(sjSDM.device = "cpu")

reticulate::py_run_string(sprintf("
import time
from sjSDM_py import model_sjSDM as _M
_M._prof_stats = {'n_loss_calls': 0, 'loss_time': 0.0,
                  'fit_time': 0.0, 'n_fits': 0}
_orig_fit = _M.Model_sjSDM.fit
_orig_blf = _M.Model_sjSDM._build_loss_function
def _blf_traced(self, *a, **k):
    f = _orig_blf(self, *a, **k)
    def f_traced(*aa, **kk):
        t0 = time.perf_counter()
        r = f(*aa, **kk)
        _M._prof_stats['loss_time'] += time.perf_counter() - t0
        _M._prof_stats['n_loss_calls'] += 1
        return r
    return f_traced
def _fit_traced(self, *a, **k):
    t0 = time.perf_counter()
    r = _orig_fit(self, *a, **k)
    _M._prof_stats['fit_time'] += time.perf_counter() - t0
    _M._prof_stats['n_fits'] += 1
    return r
_M.Model_sjSDM.fit = _fit_traced
_M.Model_sjSDM._build_loss_function = _blf_traced
"))

set.seed(101)
n_env <- 5L
X <- matrix(stats::rnorm(sites * n_env), sites)
beta <- matrix(stats::rnorm(species * (n_env + 1L)), species)
Sigma <- cov(matrix(stats::rnorm(species * species), species)) |> (\(m) m + diag(0.5, species))()
Z <- matrix(stats::rnorm(sites * species), sites) %*% t(Sigma)
p <- plogis(cbind(1, X) %*% t(beta) + Z)
Y <- matrix(stats::rbinom(length(p), 1, p), sites)

tt <- system.time(
  fit <- sjSDM(
    Y = Y, env = X, biotic = bioticStruct(df = 5L),
    family = stats::binomial("logit"),
    iter = epochs, step_size = 10L,
    learning_rate = 0.1, sampling = 400L, se = FALSE
  )
)

t_ll <- system.time(ll <- fit$model$logLik(fit$data$X, Y))

stats <- reticulate::py_to_r(
  reticulate::py_run_string(
    "import sys\nprof_stats = sys.modules['sjSDM_py.model_sjSDM']._prof_stats"
  )
)$prof_stats

res <- list(
  backend = backend, sites = sites, species = species, epochs = epochs,
  r_total = tt[["elapsed"]],
  r_loglik = t_ll[["elapsed"]],
  py_fit = stats$fit_time, n_fits = stats$n_fits,
  py_loss_total = stats$loss_time, n_loss_calls = stats$n_loss_calls
)
res$r_overhead <- res$r_total - res$py_fit
res$per_batch_loss_ms <- 1000 * res$py_loss_total / max(res$n_loss_calls, 1)
res$overhead_frac <- res$r_overhead / res$r_total

saveRDS(res, sprintf("/tmp/prof_%s_%s.rds", tag, backend))
cat(sprintf("%s %dx%d ep=%d: r_total=%.2fs py_fit=%.2fs loss=%.2fs (%d calls, %.1fms/call) logLik=%.2fs overhead=%.2fs (%.0f%%)\n",
            backend, sites, species, epochs, res$r_total, res$py_fit,
            res$py_loss_total, res$n_loss_calls, res$per_batch_loss_ms,
            res$r_loglik, res$r_overhead, 100 * res$overhead_frac))

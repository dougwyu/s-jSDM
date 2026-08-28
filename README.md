
<!-- README.md is generated from README.Rmd. Please edit that file -->

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# s-jSDM (Mojo fork) - Joint Species Distribution Modeling accelerated on Apple Silicon CPUs

## What this repository is

This is a fork of
[TheoreticalEcology/s-jSDM](https://github.com/TheoreticalEcology/s-jSDM)
that replaces the inner Monte Carlo likelihood computation of sjSDM with
a hand-written **Mojo/MAX kernel**, giving roughly **1.3–3x faster model
fits on Apple Silicon (M-series) CPUs** while producing statistically
equivalent results.

**This repository is only for users who want the Mojo-accelerated
version of sjSDM on an Apple Silicon Mac.** It currently requires:

- An Apple Silicon Mac (arm64 macOS)
- A clone of this repository (the Mojo worker binary is built locally)

If you are not in that audience – Intel Macs, Windows, or Linux – please
use the **original package instead**:

``` r
remotes::install_github("TheoreticalEcology/s-jSDM/sjSDM")
library(sjSDM)
install_sjSDM()
```

**On Apple Silicon but don’t want Mojo?** Stay here anyway. Installing
sjSDM from the upstream repository is currently unreliable on Apple
Silicon, whereas this fork contains the compatibility fixes (released as
[v0.1.0](https://github.com/dougwyu/s-jSDM/releases/tag/v0.1.0)). You
can use plain PyTorch-mode sjSDM from this fork without any of the
pixi/Mojo setup:

``` r
remotes::install_github("dougwyu/s-jSDM/sjSDM", ref = "v0.1.0")
library(sjSDM)
install_sjSDM()
```

Everything about fitting and interpreting models is identical between
all of these; only the compute backend differs.

## Why Mojo? The tradeoff

The Monte Carlo likelihood kernel is being moved from PyTorch to
[Mojo/MAX](https://www.modular.com/mojo) incrementally, and that choice
involves a real tradeoff:

**The long-term upside is hardware portability.** Mojo/MAX compiles the
same kernel source to native code for multiple chip architectures – CPU
SIMD across vendors today, with NVIDIA and other accelerators as
explicit MAX targets going forward. A hand-written CUDA or Metal kernel
locks you into one vendor; a Mojo kernel is intended to follow new
hardware without a rewrite.

**The short-term cost is maturity.** Mojo and MAX are young and changing
quickly compared to PyTorch: the language and APIs evolve between
releases, the ecosystem of libraries, debuggers, and community knowledge
is much smaller, and this repository therefore pins and validates
against specific versions rather than tracking latest. Bugs and
performance cliffs are more likely to come from the toolchain than from
our code.

**In the meantime, the speedup pays for itself on Apple Silicon CPUs:**
roughly 1.3–3x faster end-to-end fits than the PyTorch CPU path,
depending on problem size. To contain the maturity risk, only the inner
Monte Carlo loop is ported; optimizers, predictor layers, prediction,
and standard errors/Hessians remain in PyTorch, so the R API and
statistical behavior are unchanged and falling back to pure PyTorch is
always one environment variable away (`SJSDM_MOJO_BACKEND = "0"`).

**Why CPU only?** The Mojo kernel targets Apple Silicon CPUs, not the
GPU (MPS). Three reasons:

1.  **MAX does not yet expose Apple GPUs.** MAX’s accelerator support
    today centers on NVIDIA GPUs; there is no stable path to compile
    this kernel for Apple’s GPU via MAX. A native Metal port would be a
    separate, vendor-locked effort – exactly what using Mojo is meant to
    avoid.
2.  **The CPU already beats MPS.** Benchmarks in this repository show
    PyTorch-on-MPS reaching only ~1.5–1.7x over the PyTorch CPU path on
    large workloads (and losing on small ones), while the Mojo CPU
    kernel is ~1.3–3x end-to-end – and it avoids MPS’s tendency to
    materialize very large `sampling x sites x species` intermediate
    tensors.
3.  **Scope discipline.** Each backend must pass parity tests and
    benchmarks before it ships; validating one target at a time keeps
    that feasible.

If MAX gains Apple GPU support, the same kernel source should be
portable to it – which is precisely the hardware-portability argument
above.

## What sjSDM does

sjSDM is an R package for estimating joint species distribution models
(jSDMs): GLMMs that model a many-species response to the environment,
space, and a covariance term capturing conditional correlations between
species. Unlike latent-variable approximations, sjSDM fits a full
(low-rank) covariance matrix in the likelihood, numerically approximated
via simulations. The method is described in Pichler & Hartig (2021), *A
new joint species distribution model for faster and more accurate
inference of species associations from big community data*,
<https://www.doi.org/10.1111/2041-210X.13687>.

The core computations run in Python/PyTorch wrapped into R; this fork
adds a native Mojo kernel alongside PyTorch for the hot Monte Carlo
loop. Citation info: `citation("sjSDM")`.

## Installation (Mojo backend)

One-time setup – install [pixi](https://pixi.sh), create the pinned
Python environment, build the Mojo worker, and create a thin venv
wrapper that lets recent versions of `reticulate` use the pixi
environment:

``` bash
git clone https://github.com/dougwyu/s-jSDM.git
cd s-jSDM/work/mojo-backend
pixi install
pixi run mojo build -O 3 mojo/mc_logit_grad_server.mojo -o mc_grad_server_bin
.pixi/envs/default/bin/python -m venv reticulate-venv --system-site-packages
```

The `-O 3` flag matters: unoptimized builds are ~50x slower. Binaries
embed absolute paths, so redo these commands after moving or re-cloning
the repository. The final command is needed because `reticulate` \>=
1.45 refuses conda-style environments without a `conda` binary and
silently falls back to an environment without PyTorch; the wrapper venv
reuses the pixi environment’s packages via `--system-site-packages`.

In R, load the package from the checkout (this is the supported way to
use the fork – see the note below on why not `library(sjSDM)`):

``` r
Sys.setenv(RETICULATE_PYTHON = here::here(
  "work/mojo-backend/reticulate-venv/bin/python"
))
devtools::load_all(here::here("sjSDM"), quiet = TRUE)
is_torch_available()
```

> **Why not `library(sjSDM)`?** If you install this fork as a regular
> package, the installed copy looks for the Mojo worker *inside your R
> library*, won’t find it, and silently falls back to PyTorch – no
> error, just no acceleration. Just use `devtools::load_all()` on the
> checkout as shown above.

That’s it – the Mojo backend is used **automatically** whenever the
worker binary exists and the fit runs on CPU float32 logit/probit.
Control it with one environment variable:

``` r
Sys.setenv(SJSDM_MOJO_BACKEND = "auto")  # default: Mojo when available
Sys.setenv(SJSDM_MOJO_BACKEND = "0")     # force pure PyTorch
Sys.setenv(SJSDM_MOJO_BACKEND = "1")     # force Mojo (error if missing)
```

Auto mode falls back to PyTorch for the entire operation when the dtype
is float64 or any response is missing.

> **v0.2.0 safety note:** The v0.2.0 tag has a 262,144-value
> seed-generation ceiling and collision-prone mixed-shape buffer reuse.
> Both are fixed on the current development branch. Tagged-release users
> should keep `sampling * step_size * biotic_rank <= 262144`, use
> PyTorch, or disable seed transport until a corrected release is
> available.

Standard errors / Hessians always use PyTorch regardless of the setting.

For a complete walkthrough of every feature – fitting, species
associations, prediction, variation partitioning, cross-validation,
spatial models, plots, benchmarking torch vs. Mojo, and troubleshooting
– see [Code/sjSDM_mojo_tutorial.Rmd](Code/sjSDM_mojo_tutorial.Rmd).

## Basic workflow

``` r
# after the setup chunk above (load_all + RETICULATE_PYTHON):
set.seed(42)
community <- simulate_SDM(sites = 100, species = 10, env = 3)
Env <- community$env_weights
Occ <- community$response

model <- sjSDM(Y = Occ,
               env = linear(data = Env, formula = ~X1+X2+X3),
               se = TRUE, family = binomial("probit"),
               sampling = 100L, verbose = FALSE)

summary(model)
plot(model)          # niche estimates with error bars
image(getCor(model)) # species association matrix
anova(model)         # variation partitioning
```

## Validating statistical accuracy

The Mojo kernel uses different float32 arithmetic than PyTorch, so
fitted models are statistically equivalent rather than identical. To
check that equivalence quantitatively, the repository ships a validation
script that fits both backends to synthetic communities with known
ground truth and compares how well each recovers it, across three
targets: environmental coefficients, the species association matrix, and
per-species loadings on an autocorrelated spatial field (dispersal
limitation):

``` r
Rscript Code/test_statistical_accuracy.R          # ~10 min, 3 replicates
```

It prints a table of recovery metrics per backend (correlation with
truth, attenuation slope, RMSE) and exits non-zero if the backends ever
disagree beyond tolerance. Reference result from the current build:

| metric                         | torch | mojo  | worst gap |
|--------------------------------|-------|-------|-----------|
| env coefficient correlation    | 0.964 | 0.966 | 0.008     |
| association matrix correlation | 0.807 | 0.812 | 0.018     |
| spatial loading correlation    | 0.988 | 0.986 | 0.010     |

## Notes and caveats

- Cross-backend results are statistically equivalent, not bit-identical:
  float32 reduction-order differences perturb trajectories slightly.
- Keep predictors standardized (`scale(env)`); fits can diverge on
  raw-scale covariates under either backend.
- The Mojo backend is CPU-only. GPU users can still set
  `options(sjSDM.device = "mps")`, which routes fits through PyTorch
  MPS.
- Releases: [v0.1.0 “Apple Silicon
  support”](https://github.com/dougwyu/s-jSDM/releases/tag/v0.1.0)
  contains only the PyTorch compatibility fixes – convenient if you want
  plain PyTorch-mode sjSDM on Apple Silicon without the pixi/Mojo setup.
  [v0.2.0 “Mojo CPU
  acceleration”](https://github.com/dougwyu/s-jSDM/releases/tag/v0.2.0)
  adds the Mojo backend and is the current release of the fork.

## License

GPL-3, matching the upstream project. The Mojo kernel sources live under
`work/mojo-backend/mojo/`.

## Acknowledgments

This port was carried out with the assistance of **Ox Alpha**, an AI
coding and data-analysis assistant developed by an undisclosed
organization. Ox Alpha implemented and validated the Mojo/MAX kernels,
the persistent worker bridge, the SIMD optimizations, and much of the
benchmarking and parity-testing infrastructure described in this
repository, working in collaboration with Douglas Yu. All AI-generated
changes were reviewed, tested, and committed by the repository
maintainer.

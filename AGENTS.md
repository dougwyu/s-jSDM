# AGENTS.md

## Repository overview

This repository contains `sjSDM`, an R package for scalable joint species distribution models. The public API and most post-fitting analysis are implemented in R; the computational model-fitting backend is embedded Python using PyTorch and is located at `sjSDM/inst/python/sjSDM_py/`.

The current performance-porting objective is to evaluate an incremental MAX/Mojo hybrid implementation. Preserve the R API and Python orchestration initially. Treat the Monte Carlo likelihood and its gradients as the primary candidate for acceleration. Keep standard-error/Hessian functionality in PyTorch until a validated second-order autodiff implementation exists.

## Repository structure

- `sjSDM/`: R package source.
- `sjSDM/R/`: R API, model configuration, fitting wrappers, prediction, CV, ANOVA, plotting, simulation, and utilities.
- `sjSDM/inst/python/sjSDM_py/`: Python/PyTorch backend.
- `sjSDM/inst/python/tests/`: Python tests.
- `sjSDM/tests/`: R tests.
- `Code/`: experiments, benchmarks, and working scripts.
- `work/port-feasibility/`: project-local pixi environment for Mojo, MAX, PyTorch, and Python dependencies.
- `Feasibility_study.md`: MAX/Mojo feasibility assessment.

## Runtime environment

Use the project-local pixi Python rather than the deleted legacy `r-sjsdm` conda environment:

```text
work/port-feasibility/.pixi/envs/default/bin/python
```

The validated environment currently contains Mojo 1.0.0, MAX 26.5.0, PyTorch 2.5.1, and the sjSDM Python test dependencies. PyTorch reports Apple MPS built and available on the target Mac.

In R, set `RETICULATE_PYTHON` before importing sjSDM or any Python-dependent package:

```r
Sys.setenv(RETICULATE_PYTHON = here::here(
  "work/port-feasibility/.pixi/envs/default/bin/python"
))
```

For repository development, load the package with:

```r
devtools::load_all(here::here("sjSDM"), quiet = TRUE)
```

Do not assume that an installed copy of the package reflects repository edits. A fresh R session is often needed after modifying embedded Python modules because reticulate and TorchScript cache imported modules.

## Testing

Run the Python suite from the package's Python directory using the pixi interpreter:

```bash
cd sjSDM/inst/python
../../../../work/port-feasibility/.pixi/envs/default/bin/python -m pytest tests -q
```

For a quick R walkthrough, use:

```bash
Rscript Code/run_sjSDM.R
```

The walkthrough uses `devtools::load_all()`, fits simulated models, runs ANOVA and cross-validation, fits a spatial model, creates plots, and compares synthetic communities.

`sjSDM_cv()` accepts raw inputs, not a fitted model. Use `Y = ...`, `env = ...`, and related model arguments. Its tuning argument is `tune_steps`, not `IT`.

When testing parallel CV, remember that worker processes call `library(sjSDM)` and may load the installed package copy. Reinstall the local package after backend edits if parallel workers must see those edits:

```bash
R CMD INSTALL --no-multiarch sjSDM
```

## Coding conventions

- Use base R's `|>` pipe, not magrittr's `%>%`.
- Use `here::here()` for paths anchored at the repository root.
- Set `RETICULATE_PYTHON` before loading sjSDM.
- Do not guess column names or data schemas; inspect them first.
- Keep R and Python changes focused and maintainable.
- Preserve existing R API behavior unless a deliberate compatibility change is documented.
- Add or update tests for behavior changes.
- Avoid unnecessary package-wide refactors while evaluating performance.

## PyTorch compatibility notes

The backend has required compatibility repairs for current PyTorch usage:

- Tensor dimensions passed to `torch.randn()` must be integer-valued. R numerics can arrive in Python as floats, so functions receiving dimensions from R should coerce values such as `sampling` and `batch_size` with `int()`.
- Prefer positional tuple dimensions, for example `torch.randn((sampling, batch_size, df), ...)`, because this is compatible with TorchScript paths.
- Newer PyTorch versions may reject the obsolete `verbose=True` argument to `ReduceLROnPlateau`.
- `torch.cholesky` is deprecated; use `torch.linalg.cholesky` when making forward-compatibility changes.
- `torch.cuda.amp.autocast` is deprecated in favor of the current `torch.amp` API.

These repairs should be tested both through a direct Python fit and through R/reticulate, including the CV/log-likelihood path.

## MAX/Mojo porting guidance

Prioritize validation over broad feature coverage:

1. Establish CPU and MPS PyTorch baselines on representative workloads.
2. Implement a standalone binary logit/probit Monte Carlo forward kernel.
3. Pass externally generated noise during parity tests to eliminate RNG differences.
4. Compare loss and predictions with numerical/statistical tolerances.
5. Add gradients for `mu` and `sigma` only after forward parity.
6. Benchmark end-to-end runtime and memory, not only kernel throughput.
7. Retain PyTorch optimizers, predictor layers, conditional prediction, and standard errors initially.

Important dimensions for benchmarks are Monte Carlo samples, batch size, species count, covariance rank, and predictor count. A port is worthwhile only if it provides a reproducible improvement on realistic workloads, not merely on small examples dominated by Python or reticulate overhead.

## Generated and local artifacts

Avoid committing local caches, generated plots, R workspace files, Python `__pycache__` directories, or temporary benchmark output unless explicitly required. Generated walkthrough plots belong under `Code/plots/` when they are intentionally retained.

## Licensing

The project is GPL-3 licensed. Any derivative backend or distributed port must preserve compatible licensing and notices.

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

## Current port status (2026-08-21)

The Apple Silicon compatibility work is complete and was released as `v0.1.0`, titled `Apple Silicon support for sjSDM`: https://github.com/dougwyu/s-jSDM/releases/tag/v0.1.0. The release was cut at commit `e7ee582`, before the Phase 1/2 Mojo work below.

The following PyTorch/R changes are implemented and committed:

- MPS device construction uses `str(device)` rather than `device.type + ":" + str(device.index)`, avoiding the invalid `mps:None` device string.
- `ReduceLROnPlateau(verbose=True)` was removed for current PyTorch compatibility.
- `sjSDM()` and `sjSDM_cv()` use `getOption("sjSDM.device", "cpu")`; set `options(sjSDM.device = "mps")` to make MPS the default.
- Parallel `sjSDM_cv()` supports `options(sjSDM.dev = "/path/to/sjSDM")`. With this option, PSOCK workers load the source checkout using `library(devtools)` and `devtools::load_all()` instead of the installed package. Without it, workers retain the installed-package behavior.
- The Python suite passes with `35 passed, 7 xfailed, 2 xpassed`.

The Phase 0 CPU/MPS benchmark is in `Code/benchmarks/bench_baseline.py`. MPS is slower or comparable on small workloads but reached approximately 1.5–1.7x speedup on larger fits. The benchmark showed that the existing PyTorch/MPS path can materialize a very large `sampling x sites x species` intermediate.

## Phase 1/2 Mojo port status

Standalone Mojo kernels are in `work/port-feasibility/mojo/` and are not integrated into the R or Python training path yet:

- `mc_logit_loss.mojo`: binary logit Monte Carlo likelihood forward pass, using externally supplied float32 noise and parallelized across sites with `max.algorithm.parallelize`.
- `mc_logit_grad.mojo`: analytic gradients for `mu` and `sigma`, using 16 site chunks with private sigma-gradient buffers merged after the parallel work.
- `parity_check.py`: forward parity, randomized-grid parity, and finite-difference gradient validation harness.

Validation results:

- Forward parity against the PyTorch reference is at float32 rounding level; representative maximum absolute error is at most about `8e-6`.
- A randomized grid of 12 shape combinations passed; the largest observed absolute error was about `3.8e-5`.
- Gradient validation passed with worst relative error about `3.5e-3` using central differences with step `h = 1e-2`.
- On `20,000 sites x 200 species x rank 5 x 400 samples`, the parallel Mojo forward kernel took about `6.5s`, compared with about `29s` for the PyTorch CPU pipeline and `124s` for the PyTorch MPS pipeline. This is standalone kernel timing, not end-to-end model-fitting timing.

Important implementation lessons: externally supplied noise is required for deterministic parity; site-parallel scratch buffers must be private; `alloc` memory is not zero-initialized; and the derivative of `log(mean(exp(ll)))` uses ordinary softmax weights without an additional `1 / samples` factor.

## Python autograd integration status (2026-08-21)

The narrow autograd bridge is implemented:

- `sjSDM/inst/python/sjSDM_py/mojo_bridge.py`: `torch.autograd.Function` that shells out to the prebuilt binary `work/port-feasibility/mc_grad_bin` (forward + analytic `mu`/`sigma` gradients). Noise is generated with `torch.randn` inside the bridge so RNG streams match the PyTorch path exactly under a fixed seed.
- `mc_logit_grad.mojo` now accepts an optional trailing `alpha` argument (argv[12], default 1.0) so probit's 1.7012 scaling is supported; gradient parity re-passed after the change.
- `Model_sjSDM._build_loss_function` uses the bridge for logit/probit links when `SJSDM_MOJO_BACKEND=1`. Restrictions: CPU float32 only, no NaN responses, and backward requires uniform grad_output (holds for `loss.mean()` in `fit()`).
- Validation: `Code/benchmarks/train_parity_mojo.py` shows bitwise-identical training trajectories over 30 epochs vs the PyTorch loss, plus end-to-end timing.
- End-to-end timing result (one-shot transport): subprocess + file-I/O overhead (~60ms per batch call) dominated; mojo ~5x slower at 3000x60 and slightly slower at 10000x200.
- Persistent-process bridge (2026-08-21): `mojo/mc_logit_grad_server.mojo` is a long-lived worker reading a 36-byte header + raw float32 tensors from stdin and writing loss/gmu/gsigma to stdout; built as `work/port-feasibility/mc_grad_server_bin`. `mojo_bridge.py` uses it by default (`SJSDM_MOJO_PERSISTENT=0` falls back to the one-shot file transport). Mojo pipe API notes: `std.io.FileDescriptor`, `stdin.read_bytes(Span)`/`stdout.write_bytes(Span)`; `read_bytes` needs a `mut` parameter and short reads must be looped; globals are unsupported; `std.memory` has no `free`/`dealloc` for raw `alloc` pointers, so the server reuses buffers across requests (reallocates on shape change without freeing).
- Persistent-bridge results: trajectory parity still bitwise exact (30 epochs); small-workload overhead eliminated (3000x60 ties torch at ~0.5s/5 epochs). Remaining gap is kernel compute, not transport: per 500x200xrank5xK400 batch with backward, Mojo ~365ms vs PyTorch ~241ms — the gradient kernel's three passes over the noise dot products lose to fused autograd at these shapes.
- Kernel z-caching optimization (2026-08-21): `mc_logit_grad_server.mojo` now caches `z_ks = alpha*(noise_k . sigma_s + mu_i)` and per-sample `ll_ik` in per-chunk scratch buffers, computing each noise dot product exactly once instead of three times. Parity vs same-noise PyTorch reference re-passed (loss <=1.5e-5, gmu <=8e-6, gsigma <=2e-4 across three shape/alpha combos). Results: per-batch 500x200xrank5xK400 Mojo ~233ms vs PyTorch ~261ms; end-to-end 10000 sites x 200 species x 2 epochs 9.4s vs 12.4s (~1.3x); small workload 3000x60 now ~1.7x faster. Trajectory parity remains bitwise exact and the Python suite passes unchanged. The Mojo persistent path is now end-to-end faster than the PyTorch CPU path on tested workloads, but is still opt-in via `SJSDM_MOJO_BACKEND=1`.
- Shape-grid benchmark (2026-08-21/22, `Code/benchmarks/bench_grid_mojo.py`, sites=500, fwd+bwd per batch): the Mojo win is broad but not universal. Post-SIMD re-runs agree: Mojo faster in ~23/27 cells; strongest at larger species x K regardless of rank (best 1.77x at sp=200, d=2, K=400 in the latest run; 1.36-1.77x across sp=200 cells), with only tiny workloads (sp=20 or K=25 cells) losing, where absolute times are ~1-3 ms and fixed overhead dominates. Earlier interpretation (pre-SIMD) that scalar rank loops lost to BLAS-backed einsum at high rank is resolved: high-rank cells now win consistently.
- SIMD vectorization (2026-08-22): the server kernel's pass-1 noise dot products and pass-2 gsigma accumulation now use width-4 SIMD loads/stores with a scalar remainder for non-multiple ranks (`(ptr+off).load[W]()` / `ptr.store(off, val)` API; note `load[W]()` reads from the pointer as given — offset must be applied explicitly). Parity re-passed across six shape/alpha combos including odd rank 7 (loss <=9.2e-5, gmu <=2.1e-5, gsigma <=2.2e-4). Grid re-run: Mojo faster in 23/27 cells (was 9/27); high-rank cells flipped to wins (d=8, K=400: 1.34x; d=5, K=400: 1.38x at sp=200); best 1.54x (sp=200, d=2, K=400); only tiny workloads (sp=20, K=25) still lose, where absolute times are ~1-3 ms and fixed overhead dominates. End-to-end 10000x200xrank8xK400 x 2 epochs: 7.7s vs 11.7s torch (~1.5x). Trajectory parity bitwise exact; Python suite passes unchanged.
- SIMD width-8 test (2026-08-22): a W=8 build showed no meaningful difference from W=4 on rank 5/8 cells (all deltas within run-to-run noise; Apple NEON is 128-bit, so 8-wide float32 lowers to two 4-wide ops). W=4 retained. The test surfaced and fixed a latent buffer-reuse bug: input/output/scratch buffers were keyed on single dimensions (e.g. `sites`), so a shape change that kept sites constant (e.g. species 60 -> 200 at the same sites) reused stale undersized buffers and produced garbage gradients. All buffers are now keyed on the full shape fingerprint (`sites/species/rank/samples` packed); mixed-shape same-process parity re-passed for both widths. Any future per-shape caching in the server must key on the full shape.
- R-level integration check (2026-08-22): `Code/run_sjSDM.R` with `SJSDM_MOJO_BACKEND=1` (plus `options(sjSDM.dev = ...)` for CV workers) completes fitting, summary/getCov, predict, Rsquared, ANOVA refits, parallel `sjSDM_cv()` (n_cores=2), the spatial model, its ANOVA, and ALL plots including `plot(fit)`/`getSe()` (exit 0 after the se() fix below). Pure-torch parallel CV workers still hit the float-dimension `torch.randn()` compatibility error (the Mojo run passes CV because the bridge coerces dims with `int()`); see `Code/test_cv.R`.
- se()/plot(fit) fix (2026-08-22): with the Mojo backend enabled, `getSe()`/`plot(fit)` failed because `Model_sjSDM.se()` builds its loss via `_build_loss_function(train=True)`, which returned the Mojo bridge function — and the Hessian needs double backprop (`create_graph=True`), which the bridge's constant-backward custom Function cannot support; the failure was silently swallowed by `try()` in `getSe()`. Earlier diagnosis of a "pre-existing Hessian shape bug" was WRONG — it was an artifact of calling `model$se()` with raw X lacking the intercept column (`model.matrix` adds it; `data$X` already includes it, so `se()` itself is shape-correct). Fix: `_build_loss_function` gained an `allow_mojo` flag (default True) and `se()` passes `allow_mojo=False`, always using the differentiable PyTorch loss for Hessians. SEs are numerically identical across backends on a fixed-seed fit; full walkthrough passes under both backends. Note: `sjSDM(se=FALSE)` is the default, so `fit$se` is NULL unless requested or `plot(fit)` triggers `getSe()`.
- Memory audit (2026-08-21): insufficient RAM ruled out as a cause of the grid losses — worker peak RSS ~1.9 GB vs 32 GB total, system 72% free; the worst cell (sp=200, d=8, K=25) also has the smallest footprint. The audit did uncover a real leak: output and z/ll scratch buffers were allocated per request (only inputs were reused), growing ~160 MB per heavy request. Fixed by hoisting all buffers into shape-keyed reuse; worker RSS is now flat at ~0.18 GB across repeated requests, timing unchanged (~233 ms/req), trajectory parity bitwise exact, Python suite passes unchanged.
- Pure-torch float-dimension `torch.randn()` fix (2026-08-22): all `torch.randn` call sites in `model_sjSDM.py` (`_build_loss_function` simulate/train/individual paths) and `dist_mvp.py` (including `self.sampling`/`self.df` in `MVP_probit`) now coerce dimensions with `int()`. `Code/test_cv.R` passes with the pure-PyTorch backend (no Mojo bridge), so parallel `sjSDM_cv()` no longer depends on the Mojo path for this fix.
- The Python suite passes unchanged (`35 passed, 7 xfailed, 2 xpassed`).

Keep optimizers, predictor layers, prediction, and `se()`/Hessian functionality in PyTorch initially.

Relevant commits after the release include `5d9f66f` (parallel forward kernel), `ad6d66f` (gradient kernel), and `2f44f94` (ignore Mojo build/temp artifacts). Build artifacts such as `mc_loss_bin` and `mojo/tmp/` are ignored.

## Generated and local artifacts

Avoid committing local caches, generated plots, R workspace files, Python `__pycache__` directories, or temporary benchmark output unless explicitly required. Generated walkthrough plots belong under `Code/plots/` when they are intentionally retained.

`Code/plots/`, `Code/run_sjSDM.R`, and other exploratory working scripts may remain untracked unless explicitly requested. The pixi lockfile `work/port-feasibility/pixi.lock` was intentionally left local; `pixi.toml` is committed for the environment specification.

Use `pixi run mojo ...` from `work/port-feasibility/` so the Mojo standard library and MAX modules resolve correctly. Directly invoking the environment's `mojo` binary without pixi activation may fail to locate `std`.

The Mojo parity harness writes temporary raw float32 files under `work/port-feasibility/mojo/tmp/` and invokes the kernel via `pixi run mojo run`; use a prebuilt binary with `pixi run mojo build` for timing rather than including compilation time.

## Release history

- `v0.1.0` — `Apple Silicon support for sjSDM`, based on `e7ee582`; excludes the later Mojo Phase 1/2 commits.
- Phase 1/2 Mojo commits are currently post-release development and do not change the released R/Python API.

## GitHub CLI

The `gh` token lacks the `workflow` scope, so `gh release create` refuses with
a scope error even for plain releases; work around it with the REST API
(`gh api -X POST repos/<owner>/<repo>/releases --input body.json`), or run
`gh auth refresh -h github.com -s workflow` once to fix permanently.

## Licensing

The project is GPL-3 licensed. Any derivative backend or distributed port must preserve compatible licensing and notices.

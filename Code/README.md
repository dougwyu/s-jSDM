# Code/ layout

Note: all R scripts here expect `RETICULATE_PYTHON` to point at
`work/mojo-backend/reticulate-venv/bin/python` -- a thin venv wrapper over
the pixi environment required by `reticulate` >= 1.45. Create it once with:

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m venv \
  work/mojo-backend/reticulate-venv --system-site-packages
```

## User-facing

Start here:

- `sjSDM_mojo_tutorial.Rmd` -- complete walkthrough of the package with the
  Mojo backend: fitting, species associations, prediction, variation
  partitioning, cross-validation, spatial models, plots, and benchmarking.
- `run_sjSDM.R` -- quick end-to-end smoke test of every major feature
  (fits simulated models, ANOVA, CV, spatial model, plots). Run with
  `Rscript Code/run_sjSDM.R`.
- `test_statistical_accuracy.R` -- validates that the Mojo backend recovers
  environmental coefficients, species associations, and spatial dispersal
  loadings from synthetic communities as accurately as pure PyTorch.
  Run with `Rscript Code/test_statistical_accuracy.R [n_rep]`
  (~10 minutes at the default of 3 replicates; exits non-zero on failure).

## Development / internal (`dev/`)

Scripts used during development of the Mojo port. Not needed to use the
package.

- `benchmarks/` -- PyTorch CPU/MPS baselines and Mojo shape-grid benchmarks.
- `dev/benchmarks/validate_mojo_release.py` -- target-machine release checks
  for cross-binary response bytes, explicit-noise numerical accuracy, and
  worker RSS. Run from the repository root with:

  ```bash
  work/mojo-backend/.pixi/envs/default/bin/python \
    Code/dev/benchmarks/validate_mojo_release.py \
    --baseline-server /private/tmp/sjsdm-mojo-v0.2.0-server \
    --candidate-server work/mojo-backend/mc_grad_server_bin
  ```

  The absolute RSS gate is calibrated for the target Apple Silicon machine;
  it is a target-machine release check, not a portable unit test.
- `bench_realdata.R`, `bench_real_occurrence.R` -- end-to-end R-level
  timing on simulated and bundled real datasets.
- `test_cv.R`, `test_plots.R`, `test_randn.R`, `trace_rng.R`,
  `profile_transport.R` -- targeted regression/diagnostic scripts.

## Maintaining the Mojo/MAX pin

The pixi environment pins specific toolchain versions (currently Mojo 1.0.0
and MAX 26.5.0). Mojo/MAX are young and change quickly, so treat upgrades as
deliberate events, not automatic ones:

1. **Bump the pin** in `work/mojo-backend/pixi.toml`, then `pixi install`.
2. **Rebuild** the worker: `pixi run mojo build -O 3 mojo/mc_logit_grad_server.mojo -o mc_grad_server_bin` (from `work/mojo-backend/`; `-O 3` matters -- unoptimized builds are ~50x slower).
3. **Run parity tests**: the Python bridge suite (`cd sjSDM/inst/python && <pixi python> -m pytest tests -q`) and the kernel-level harness (`work/mojo-backend/mojo/parity_check.py`).
4. **Re-benchmark**: `dev/benchmarks/` shape-grid plus one end-to-end R-level run (`bench_realdata.R`).
5. If parity or performance regresses, revert the pin; do not merge a partial upgrade.

Keep version-sensitive code isolated: the kernels should touch only a small
Mojo API surface (SIMD loads/stores, `max.algorithm.parallelize`, pipe I/O in
`mc_logit_grad_server.mojo`). Watch Modular's changelogs specifically for
changes to those APIs.

## Legacy upstream experiments

Directories inherited from the upstream s-jSDM project (simulation studies
and theory notes for the original paper); kept for reference only:

`SimulationExperiments/`, `sLVM/`, `Theory/`, `utils/`,
`sjSDM with random effects/`.

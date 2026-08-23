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
- `bench_realdata.R`, `bench_real_occurrence.R` -- end-to-end R-level
  timing on simulated and bundled real datasets.
- `test_cv.R`, `test_plots.R`, `test_randn.R`, `trace_rng.R`,
  `profile_transport.R` -- targeted regression/diagnostic scripts.

## Legacy upstream experiments

Directories inherited from the upstream s-jSDM project (simulation studies
and theory notes for the original paper); kept for reference only:

`SimulationExperiments/`, `sLVM/`, `Theory/`, `utils/`,
`sjSDM with random effects/`.

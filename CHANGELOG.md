# Changelog

All notable changes to the sjSDM package are documented here.
Format loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **Phase 1 Mojo forward kernel** (`work/port-feasibility/mojo/`): binary
  logit Monte Carlo likelihood forward pass, parallelized across sites on
  the CPU thread pool via `max.algorithm.parallelize`. Externally supplied
  noise (no RNG in the kernel) enables exact parity testing.
  - Parity vs PyTorch reference: max abs error ≤ 8e-6 at float32.
  - Speed on large workload (20k sites × 200 species × 400 MC samples):
    ~6.5s vs ~29s PyTorch CPU and ~124s PyTorch MPS; also avoids
    materializing the sampling×sites×species intermediate tensor.
  - `parity_check.py` generates inputs, runs both implementations, and
    compares per-site losses.

### Fixed

- **MPS device support**: replaced `device.type + ":" + str(device.index)`
  device-string construction with `str(device)` at 5 sites
  (`sjSDM_py/model_sjSDM.py`, `sjSDM_py/dist_mvp.py`). Previously any MPS
  fit failed with `RuntimeError: Invalid device string: 'mps:None'`.
- Removed obsolete `verbose=True` argument from `ReduceLROnPlateau`
  (`sjSDM_py/model_sjSDM.py`), deprecated in PyTorch 2.x.
- Parallel CV workers (`sjSDM_cv`, `n_cores > 1`) now export `dev_path`
  from the calling frame (`envir = environment()`), fixing a lookup failure
  introduced during development of the feature below.

### Added

- **`options(sjSDM.device = ...)`**: global option selecting the default
  compute device for `sjSDM()` and `sjSDM_cv()`. Defaults to `"cpu"`
  (unchanged behavior); set to `"mps"` on Apple Silicon for GPU fitting.
  Downstream methods follow the device stored on fitted objects.
- **`options(sjSDM.dev = ...)`**: development escape hatch for parallel
  cross-validation. When set to a source checkout path, PSOCK workers load
  that source via `devtools::load_all()` instead of the installed package,
  so local backend/R edits apply without reinstalling. Unset by default;
  workers require `devtools` when used.
- **Phase 0 benchmark harness** (`Code/benchmarks/bench_baseline.py`):
  synthetic binary community fits comparing CPU vs MPS across sites,
  species count, and Monte Carlo samples, with per-device warm-up.
  Baseline result: MPS wins ~1.5–1.7× only on large workloads
  (≥200 species or ≥20k sites × sampling ≥100); CPU is faster on small ones.

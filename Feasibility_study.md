---
title: "Feasibility_study"
output: html_document
---

# s-jSDM → Mojo/MAX feasibility study

Date: 2026-08-21  
Scope: an evidence-based feasibility spike; no production port was created.

## Executive recommendation

**Go, but only as an incremental hybrid port.** Do not attempt a direct full rewrite from R/PyTorch to Mojo/MAX. Start by preserving the R API and Python orchestration, then replace one validated Monte-Carlo likelihood kernel with Mojo. Continue only if that kernel meets numerical and performance gates on the target Mac.

The project already has a credible macOS route through R + Python/PyTorch. A Mojo/MAX port is justified only when measurable Apple-silicon performance, distribution simplicity, or long-term control is worth the engineering and scientific-validation cost.

## Evidence gathered

I inspected a fresh shallow clone of `TheoreticalEcology/s-jSDM` at commit `855e785863e96a44940776410c76dbdc6d4b0ab3` (2026-06-11). The package is GPL-3 and the current R package version is 1.0.7.

The user-facing R package is substantial: roughly 5,900 lines across R and Python sources. The computational Python core is `sjSDM/inst/python/sjSDM_py/`:

| Component | Lines | Role |
|---|---:|---|
| `model_sjSDM.py` | 1,134 | model construction, training, prediction, likelihoods, standard errors |
| `dist_mvp.py` | 182 | Monte-Carlo multivariate probability/log-likelihood and conditional prediction |
| `optimizer.py` | 73 | Adamax, RMSprop, SGD and optional third-party optimizers |
| `utils_fa.py` | 123 | seeds, covariance and importance helpers |

The R package calls this Python core through `reticulate`; it also supplies configuration, CV, ANOVA/variation partitioning, internal structure, importance, plotting, simulation, and the package interface. Replacing the R layer is out of scope for an initial port.

## What must be reproduced

The critical training expression samples noise of shape `[samples, batch, latent-df]`, multiplies by a species-by-latent covariance factor, adds the predictor output, transforms it by the response family, and uses a numerically stabilized log-mean-exp reduction. The primary hot operations are random generation, einsum/matmul, elementwise transforms, reduction, and gradient propagation.

Supported pathways include:

- environmental and optional spatial linear/DNN predictors;
- links/families: probit/logit, linear, Poisson count, negative binomial, and normal;
- low-rank covariance factor `sigma` and covariance/precision regularization;
- minibatch fitting, Adamax/RMSprop/SGD, LR scheduling and early stopping;
- conditional predictions using the separate Monte-Carlo likelihood;
- standard errors computed from first and second derivatives (Hessian per species).

The package depends on considerably more than tensors: `torch.nn`, `torch.optim`, `torch.autograd`, `torch.utils.data`, distributions, TorchScript, Pyro, `torch_optimizer`, and `madgrad`. The packaged Python `setup.py` declares only NumPy, so the R installer supplies the actual runtime dependencies.

## Local feasibility result

This Mac is an Apple M4 (ARM64) running macOS 26.6.2 with Xcode 26.6. The feasibility environment is project-local at `work/port-feasibility/.venv`: Mojo 1.1.0.dev2026082105, MAX 26.6.0.dev2026082105, PyTorch 2.13.0, Pyro 1.9.1, NumPy, PyTest, `torch-optimizer`, `madgrad`, and the remaining Python test dependencies are installed. The Metal Toolchain 17F109 is also installed and `metallib` is available. PyTorch reports both MPS built and MPS available.

Mojo compiled and ran a disposable smoke program successfully. Its current nightly syntax uses `def main()`; the older `fn main()` syntax is rejected.

The upstream Python suite did not establish a passing baseline: **35 failed, 9 xfailed**. Every non-expected failure reaches `model_sjSDM.py:385`, where the source calls `torch.optim.lr_scheduler.ReduceLROnPlateau(..., verbose=True)`; PyTorch 2.13 rejects the removed `verbose` argument. This is a reproducible framework-compatibility issue, not a Mojo/MAX issue. A deliberately narrow smoke run with that scheduler disabled succeeds on CPU, producing a `(20, 4)` prediction matrix. The equivalent MPS run fails before model computation because the code derives the device string as `mps:None`; PyTorch rejects that invalid device string. No upstream source was modified during this study.

Mojo supports Apple silicon on macOS. Mojo can interoperate with Python, which makes a kernel-first migration technically plausible. However, MAX is not a drop-in replacement for PyTorch's eager autograd/training ecosystem. The exact standard-error path—nested autodiff to form a Hessian—is the strongest reason not to replace all of PyTorch in the first milestone.

## Feasible architecture

```text
R sjSDM API (unchanged)
  → reticulate
    → Python compatibility/orchestration layer
      → PyTorch model, optimizer, autograd, DataLoader (initially retained)
      → Mojo extension: sampled covariance likelihood forward/backward kernel
      → MAX/Metal execution where supported and benchmarked
```

The first implementation should expose a narrow function such as `mc_binary_loss(mu, sigma, y, samples, seed)`. Its inputs, output, missing-data behavior, precision, RNG algorithm, and gradient contract must be specified before coding. The Python layer owns batching and optimizer state; the R layer is unchanged.

## Risk register

| Risk | Severity | Mitigation / gate |
|---|---|---|
| Monte-Carlo RNG differs, invalidating bitwise comparisons | High | Fixed seed and shared/noise-input test mode; compare distributional estimates and gradients with tolerances |
| MAX lacks an easy full training/autodiff equivalent | High | Keep PyTorch optimizer/autograd initially; implement a custom backward only after forward parity |
| `se()` requires second-order derivatives | High | Explicitly defer; retain PyTorch `se()` until a validated Hessian path exists |
| GPU memory grows as samples × batch × species | High | Benchmark realistic datasets; fuse operations and avoid materializing intermediates |
| Current code contains CUDA-specific APIs and old TorchScript idioms | Medium | Establish a current PyTorch MPS baseline before porting; treat baseline repairs as separate work |
| R users expect downstream objects and analyses | Medium | Preserve R API and model serialization in v1 |
| GPL-3 derivative requirements | Medium | Keep derivative source GPL-3-compatible and preserve notices; seek legal advice for distribution questions |

## Staged plan and decision gates

### Phase 0 — baseline and test harness (1–2 weeks)

Install current PyTorch with MPS support and the existing runtime dependencies. Run the upstream tests; add deterministic fixtures for the butterfly/eucalypt or synthetic data paths. Record CPU and MPS timings, peak memory, likelihood, predictions, covariance, and finite-difference/analytic gradient checks.

**Gate:** the maintained PyTorch implementation runs on the target Mac, and representative workloads plus tolerances are agreed.

### Phase 1 — throwaway Mojo kernel proof (2–4 weeks)

Implement only the binary logit/probit sampled likelihood forward pass. Pass random noise as an input so it matches the PyTorch reference. Exercise CPU first, then Apple GPU only if the installed toolchain supports the required path.

**Gate:** forward loss agrees within an agreed statistical/numerical tolerance and is at least 1.3–1.5× faster end-to-end on a representative workload. If it is not, stop: the existing PyTorch MPS route is preferable.

### Phase 2 — differentiable hybrid kernel (3–6 weeks)

Add validated gradients for `mu` and `sigma`; integrate with the Python wrapper while retaining PyTorch for neural layers, optimizer, schedules, conditional prediction, and standard errors.

**Gate:** training trajectories and fitted inference agree across multiple seeded simulations, with a worthwhile wall-clock and memory improvement.

### Phase 3 — broaden coverage selectively (6–12+ weeks)

Add count/NB/normal families, missing-data and conditional-prediction paths, then decide whether to port predictor layers. Do not port the R analyses unless a standalone non-R product is explicitly desired. Keep `se()` in PyTorch unless a second-order derivative design is validated.

## Estimated effort

| Target | Team/effort estimate |
|---|---|
| Baseline + Phase 1 proof | 3–6 engineer-weeks |
| Hybrid, binary-model production path | 2–4 engineer-months |
| Feature-complete PyTorch-core replacement including all families, conditional inference, and SE | 6–12+ engineer-months, with material framework risk |

These are engineering estimates, not commitments; scientific validation and available MAX GPU features dominate the uncertainty.

## Next action

The controlled Phase 0 toolchain setup is complete. Before benchmarking or writing a Mojo likelihood kernel, make the two small, independently tested upstream-compatibility repairs in a fork/working copy: remove the obsolete scheduler argument and represent Apple MPS as `mps` rather than `mps:None`. Then rerun the Python suite on CPU and MPS and add fixed-seed numerical fixtures; that produces the first hard go/no-go result before any port code is kept.

## Sources

- https://github.com/TheoreticalEcology/s-jSDM
- https://docs.modular.com/mojo/requirements/
- https://docs.modular.com/mojo/std/python/
- https://docs.modular.com/max/faq

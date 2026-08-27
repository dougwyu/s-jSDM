# Mojo Release Safety Design

**Date:** 2026-08-27

**Status:** Proposed for implementation

**Roadmap position:** Subproject 1 of 3. This design covers Mojo correctness and resource safety. Installation/validation reproducibility and performance optimisation remain separate follow-on subprojects.

## Objective

Make the v0.2.0 Mojo backend safe for supported CPU float32 logit and probit fits across valid request sizes, repeated fits, and backend transport modes. The corrected backend must preserve the existing Python-facing loss interface and persistent binary protocol while eliminating partially initialised random noise, unsafe buffer reuse, shape-change memory growth, incorrect automatic dtype selection, and the broken one-shot fallback.

## Scope

This subproject changes:

- `work/mojo-backend/mojo/mc_logit_grad_server.mojo`
- `sjSDM/inst/python/sjSDM_py/mojo_bridge.py`
- `sjSDM/inst/python/sjSDM_py/model_sjSDM.py`
- `sjSDM/inst/python/tests/test_mojo_bridge.py`
- The relevant backend-selection tests in `sjSDM/inst/python/tests/test_sjSDM.py`
- Mojo safety notes in `Code/sjSDM_mojo_tutorial.Rmd`, `README.Rmd`, `README.md`, and `CHANGELOG.md` after the implementation is verified

The persistent request and response byte layouts remain unchanged:

- Request header: four little-endian `Int64` dimensions, `Float32` alpha bits, and a `UInt32` transport mode
- Request body: `mu`, `sigma`, `y`, then either explicit float32 noise or one `UInt64` seed
- Response body: per-site loss, `gmu`, and `gsigma` as contiguous float32 values

## Non-goals

- Changing the Monte Carlo estimator or its smoothing constants
- Matching PyTorch's random-number stream bit-for-bit in seed-transport mode
- GPU or Apple MPS support
- General weighted-loss gradients; the existing uniform-upstream-gradient restriction remains
- Performance tuning beyond eliminating pathological allocation churn
- Installer, pixi-lock, statistical-validation, or benchmark-reporting changes; those belong to later subprojects

## Required invariants

1. Every requested noise element in `[0, samples * sites * rank)` is initialised exactly once before the likelihood reads it.
2. Request correctness cannot depend on prior requests or on allocator contents.
3. Buffer selection cannot depend on a lossy packed shape fingerprint.
4. Repeated shape changes cannot cause unbounded live allocation growth.
5. Automatic backend selection uses Mojo only for CPU float32 tensors with a supported link, available binary, and response data without missing values.
6. Forced Mojo mode remains strict: unsupported dtype, device, missing responses, or missing binary produces an explicit error.
7. `SJSDM_MOJO_PERSISTENT=0` executes the one-shot binary and returns tensors through the public autograd path.
8. Existing explicit-noise forward and analytic-gradient tolerances do not regress.

## Design

### 1. Complete server-side noise generation

Keep the compiled worker count at `N_GEN_BLOCKS = 64`, because Mojo's `parallelize` worker count is a compile-time parameter in the current implementation. Change each worker callback from processing one block to processing a strided sequence of blocks:

```text
start = worker_id * BLOCK
stride = N_GEN_BLOCKS * BLOCK
while start < n:
    fill [start, min(start + BLOCK, n))
    start += stride
```

The Marsaglia polar calculation and the mapping from `(seed, element index, rejection attempt)` to random values remain unchanged. This preserves deterministic results for requests at or below the old 262,144-element boundary while extending generation to arbitrary valid `n`.

The regression suite will exercise `n` immediately below, at, and above the old boundary. A large seeded request will be evaluated, the same worker buffer will then be overwritten through explicit-noise transport, and the original seeded request will be repeated. All three returned arrays must be bit-identical between the two seeded calls.

### 2. Collision-free, capacity-based buffers

Remove `cap_shape` and the base-8192 fingerprint. Compute the required element count for each buffer from the current request:

- `mu`, `y`, and `gmu`: `sites * species`
- `sigma` and `gsigma`: `species * rank`
- `noise`: `samples * sites * rank`
- `out`: `sites`
- `gsigma_buf`: `N_CHUNKS * species * rank`
- `zbuf_all`: `N_CHUNKS * ceil(sites / N_CHUNKS) * samples * species`
- `llbuf_all`: `N_CHUNKS * ceil(sites / N_CHUNKS) * samples`

Track a capacity integer beside each pointer. For each buffer independently:

1. Reuse the pointer when `required <= capacity`.
2. When growth is required, deallocate the old pointer if its capacity is non-zero, allocate exactly `required` elements, and set the new capacity.
3. Do not shrink on smaller requests; this makes alternating shapes allocation-stable.
4. Deallocate every non-empty buffer once after the request loop exits.

The current per-request logical dimensions still control reads, writes, indexing, and response lengths. Capacity is only an allocation bound, never a logical shape.

This design removes both the collision and the leak without changing the binary protocol or introducing a separate allocator abstraction that would be difficult to express safely in the current Mojo version.

### 3. Automatic dtype eligibility

In `model_sjSDM.py`, automatic mode will include `self.dtype == torch.float32` in the Mojo eligibility predicate. A CPU float64 fit in auto mode therefore builds and uses `torch_tmp`.

Forced mode (`SJSDM_MOJO_BACKEND=1`) will continue selecting the Mojo function and will retain the bridge's explicit CPU-float32 guard. This distinction keeps auto mode convenient and forced mode diagnostic.

Tests will patch or select an available worker, build otherwise equivalent float32 and float64 models, and assert:

- auto + CPU float32 selects Mojo when the binary is available;
- auto + CPU float64 completes through PyTorch;
- forced Mojo + CPU float64 raises the documented `CPU float32` error.

### 4. One-shot transport execution

Replace the ambiguous conditional lambda with an explicit branch:

```python
if persistent:
    runner = lambda: _WORKER.run(mu, sigma, y, noise, alpha)
else:
    runner = lambda: _run_oneshot(mu, sigma, y, noise, alpha)
```

The one-shot integration test must set `SJSDM_MOJO_PERSISTENT=0` and call `mojo_logit_loss`, not `_run_oneshot` directly. Before the call it will save Torch's RNG state; afterwards it will restore that state and recreate the exact noise tensor that the public call consumed. It will then compare loss, `mu.grad`, and `sigma.grad` with the persistent explicit-noise path using the existing tolerances. Environment changes will be isolated with pytest's `monkeypatch`, and the global worker will be closed by the existing fixture.

### 5. Failure behaviour

No protocol-level error response is added in this subproject. Python remains responsible for tensor dtype, device, contiguity, and missing-response guards before a request is sent.

The bridge will validate all three model tensors, not only `mu`:

- `mu`, `sigma`, and `Ys` must be CPU float32 tensors;
- `mu`, `sigma`, and `Ys` dimensions must agree;
- `sampling` must be a positive integer;
- `Ys` must not contain NaNs in forced/direct Mojo calls.

Invalid requests fail in Python with a `RuntimeError` before any bytes are written, preventing protocol desynchronisation. The internal model path normally satisfies these conditions, but the public bridge function must defend its own boundary.

If the persistent worker dies mid-request, the existing one-restart policy remains. A validation failure is not retryable and must not restart the worker.

## Testing strategy

### Focused regression tests

Add tests covering:

- seed transport at 262,143, 262,144, and 262,145 requested values;
- a request substantially above the boundary;
- seeded output independence from an intervening explicit-noise buffer overwrite;
- the historical `(rank=2, samples=1)` / `(rank=1, samples=8193)` fingerprint collision sequence;
- repeated growth and shrinkage across sites, species, rank, and samples;
- public one-shot forward and backward execution;
- auto float64 fallback and forced float64 rejection;
- mismatched `mu`, `sigma`, and `Y` shapes failing before transport.

### Memory stability gate

Use an integration check on Apple Silicon with `psutil`:

1. Start one persistent worker.
2. Warm it with both shapes used by the alternating test.
3. Record worker RSS.
4. Alternate the shapes at least 20 additional times.
5. Require RSS growth after warm-up to remain below 16 MiB.

This is a release-validation gate rather than a universally strict unit assertion, because operating-system RSS accounting can vary. The functional test still exercises the same shape sequence in the standard pytest suite.

### Numerical gate

For an explicit-noise request larger than the former seed limit:

- maximum absolute loss error versus PyTorch: `<= 1e-4`;
- maximum absolute `gmu` error: `<= 1e-4`;
- maximum absolute `gsigma` error: `<= 5e-4`.

Seed transport is checked for determinism, finiteness, and independence from prior buffer contents rather than equality with PyTorch RNG values.

### Suite gate

The subproject is not complete until:

- focused Mojo tests pass with the compiled binaries;
- the complete Python suite passes with only documented xfails;
- the source-aware R suite has zero failed expectations;
- `git diff --check` is clean;
- the memory and large-request numerical gates pass.

## Documentation and benchmark handling

Until all gates pass, retain the tutorial's v0.2.0 safety warning and do not publish shape-grid results above the old seed limit. Once the corrected implementation is verified:

1. Replace the workaround language with a statement that v0.2.0 contains the limitation and the next release fixes it.
2. Rerun all shape-grid and end-to-end benchmarks from clean worker processes.
3. Keep historical v0.2.0 numbers labelled by release rather than silently replacing them.

Performance optimisation begins only after this corrected build supplies the new baseline.

## Acceptance criteria

- A seeded request of any tested valid size is independent of prior worker contents.
- The historical shape-key collision sequence matches a fresh worker within numerical tolerance.
- Alternating warmed shapes does not exhibit monotonic per-request RSS growth.
- Auto float64 fits use PyTorch and complete; forced float64 fits fail clearly.
- Public one-shot mode completes forward and backward.
- Large explicit-noise loss and gradients satisfy the numerical gate.
- Existing persistent transport, restart, NaN, probit-alpha, and autograd tests remain green.
- No wire-format change is required by existing Python or Mojo clients.

# Mojo Release Safety Design

**Date:** 2026-08-27

**Status:** Implemented and merged. Delivered on `master` in `068f4ba..5a9671c` (2026-08-28), summarised in the `[Unreleased]` section of `CHANGELOG.md`. Five items were deliberately not implemented; they are listed with their reasons under "Review outcomes" at the end, and the implementation plan recorded three of them as explicit non-goals. The plan itself, `docs/superpowers/plans/2026-08-27-mojo-release-safety.md`, is complete and superseded.

**Revision:** 2026-08-27, revised after design review. Changes from the first draft are listed under "Review outcomes" at the end.

**Roadmap position:** Subproject 1 of 3. This design covers Mojo correctness and resource safety. Installation/validation reproducibility and performance optimisation remain separate follow-on subprojects.

## Objective

Make the v0.2.0 Mojo backend safe for supported CPU float32 logit and probit fits across valid request sizes, repeated fits, and backend transport modes. The corrected backend must preserve the existing Python-facing loss interface and persistent binary protocol while eliminating partially initialised random noise, unsafe buffer reuse, shape-change memory growth, a dead scratch dimension that dominates worker memory, incorrect automatic dtype selection, and the broken one-shot fallback.

## Scope

This subproject changes:

- `work/mojo-backend/mojo/mc_logit_grad_server.mojo`
- `sjSDM/inst/python/sjSDM_py/mojo_bridge.py`
- `sjSDM/inst/python/sjSDM_py/model_sjSDM.py`
- `sjSDM/inst/python/tests/test_mojo_bridge.py`
- The relevant backend-selection tests in `sjSDM/inst/python/tests/test_sjSDM.py`
- `AGENTS.md`, whose shape-fingerprint guidance is superseded by this design
- Mojo safety notes in `Code/sjSDM_mojo_tutorial.Rmd`, `README.Rmd`, `README.md`, and `CHANGELOG.md` after the implementation is verified

The persistent request and response byte layouts remain unchanged:

- Request header: four little-endian `Int64` dimensions, `Float32` alpha bits, and a `UInt32` transport mode
- Request body: `mu`, `sigma`, `y`, then either explicit float32 noise or one `UInt64` seed
- Response body: per-site loss, `gmu`, and `gsigma` as contiguous float32 values

`work/mojo-backend/mojo/mc_logit_grad.mojo`, the one-shot kernel, is deliberately not changed. It reads noise from a file and allocates per invocation, so it carries none of the defects fixed here. Note that it is also a different implementation of the same maths: it has no z/ll caching and recomputes the noise dot products three times per site. Section 4 revives a call path that has been dead since the conditional-lambda defect landed, so comparisons between it and the persistent worker are cross-implementation agreement checks, not regression checks.

## Non-goals

- Changing the Monte Carlo estimator or its smoothing constants
- Matching PyTorch's random-number stream bit-for-bit in seed-transport mode
- GPU or Apple MPS support
- General weighted-loss gradients; the existing uniform-upstream-gradient restriction remains
- Throughput tuning. Removing pathological worker allocation churn and dead scratch capacity is in scope. Python pipe-copy reduction, cache-line experiments, and kernel layout changes belong to the performance subproject because they are not required for correctness or bounded worker memory.
- Installer, pixi-lock, statistical-validation, or benchmark-reporting changes; those belong to later subprojects

## Required invariants

1. Every requested noise element in `[0, samples * sites * rank)` is initialised exactly once before the likelihood reads it.
2. Request correctness cannot depend on prior requests or on allocator contents.
3. Buffer selection cannot depend on a lossy packed shape fingerprint.
4. Repeated shape changes cannot cause unbounded live allocation growth.
5. Each buffer's required capacity is derived from the largest index the kernel can access; retained high-water capacity may be larger than the current request.
6. The byte count written for each request section equals the byte count the server computes from the header, for every entry point that writes to the pipe.
7. The server refuses to allocate from header dimensions it has not sanity-checked.
8. Automatic backend selection uses Mojo only for CPU float32 tensors with a supported link, available binary, and response data without missing values.
9. Forced Mojo mode remains strict: unsupported dtype, device, missing responses, or missing binary produces an explicit error.
10. `SJSDM_MOJO_PERSISTENT=0` executes the one-shot binary and returns tensors through the public autograd path.
11. Existing explicit-noise forward and analytic-gradient tolerances do not regress.

## Design

### 1. Complete server-side noise generation

Change each `gen_noise` worker callback from processing one block to processing a strided sequence of blocks:

```text
start = worker_id * BLOCK
stride = n_workers * BLOCK
while start < n:
    fill [start, min(start + BLOCK, n))
    start += stride
```

The Marsaglia polar calculation and the mapping from `(seed, element index, rejection attempt)` to random values remain unchanged, so each element's value depends only on its own index. Block starts stay even under the new stride, so the polar pairing stays aligned and the boundary guard on `idx + 1` keeps working.

Set the worker count at the call site rather than always dispatching the maximum: `n_workers = min(N_GEN_BLOCKS, (n + BLOCK - 1) // BLOCK)` followed by `parallelize[fill](n_workers)`. A compiler probe against the pinned Mojo 1.0.0 toolchain confirms that `parallelize` accepts a runtime `Int` count. This avoids dispatching up to 63 immediately exiting workers on the small requests that dominate a normal fit. Determinism is unaffected because values are indexed, not sequenced.

**Index mapping is pinned for this subproject only.** Preserving the current index-to-value mapping keeps results at or below the old 262,144-element boundary bit-identical to v0.2.0, which makes the fix reviewable against existing benchmark trajectories. That property is not an acceptance criterion and is not a long-term commitment. Non-goal 2 already disclaims matching PyTorch's stream, and the acceptance criteria below require only within-process determinism. The performance subproject is free to renumber, and should consider it: generating noise site-major as `[sites][samples][rank]` would make the dominant loop's reads contiguous, where today `nz_base = k * sites * rank + site * rank` strides by `sites * rank` across `k`. Record that as an open option there rather than foreclosing it here.

### 2. Collision-free, capacity-based buffers

Remove `cap_shape` and the base-8192 fingerprint. Compute the required element count for each buffer from the current request:

- `mu`, `y`, and `gmu`: `sites * species`
- `sigma` and `gsigma`: `species * rank`
- `noise`: `samples * sites * rank`
- `out`: `sites`
- `gsigma_buf`: `N_CHUNKS * species * rank`
- `zbuf_all`: `N_CHUNKS * samples * species`
- `llbuf_all`: `N_CHUNKS * samples`

Track a capacity integer beside each pointer. For each buffer independently:

1. Reuse the pointer when `required <= capacity`.
2. When growth is required, call the old pointer's `unsafe_free()`, allocate exactly `required` elements, and set the new capacity.
3. Do not shrink on smaller requests; this makes alternating shapes allocation-stable.
4. Call `unsafe_free()` on every buffer, including the header and seed buffers, once after the request loop exits.

Retain the current `alloc[T](0)` initialisation with capacity zero, but free that zero-length allocation before the first growth. Compiler probes against the pinned Mojo 1.0.0 toolchain show that `Pointer` is deliberately non-nullable and `dealloc(pointer)` does not compile; `pointer.unsafe_free()` is the supported operation for the pointer returned by the existing allocation API.

The current per-request logical dimensions still control reads, writes, indexing, and response lengths. Capacity is only an allocation bound, never a logical shape.

**Drop the per-site dimension from the scratch buffers.** `zbuf` and `llbuf` are per-site scratch, not per-chunk history: every read and write of both sits inside the `for site in range(start, stop)` body, and nothing outside that body touches either buffer. The current sizing therefore allocates `ceil(sites / N_CHUNKS)` times more scratch than the kernel ever indexes, and that dead space dominates worker memory. Set `zbase` and `llbase` to zero and size the slots at `samples * species` and `samples`. The `chunk` term disappears from the allocation block; the `chunk` used to partition sites in `chunk_loss` is unchanged.

The reviewer's A/B result was independently reproduced against the current binary at `sites=4096, species=200, rank=8, samples=25`:

- worker RSS: 103.2 MiB before, 25.1 MiB after, a 76% reduction
- `zbuf_all` alone: 81.9 MB before, 0.32 MB after
- response bytes bit-identical, maximum absolute difference 0.0

The reviewer also measured no material throughput change; performance will be remeasured during implementation rather than treated as a design premise. This is the single largest resource defect in the backend. `AGENTS.md` records "\~160 MB per heavy request" of growth and a "worker peak RSS \~1.9 GB" during the shape-grid failures; the oversized scratch buffer accounts for most of the retained per-shape allocation. Fixing it also makes the memory gate below meaningful, because the remaining shape-dependent footprint is dominated by `noise` and the `sites * species` arrays, all of which are genuinely required.

Cache-line padding is deferred to the performance subproject. Rounding each stride to 128 bytes does not by itself guarantee separation unless the base allocation is also aligned appropriately, and no benchmark in the review isolates padding's effect from the scratch-size correction.

**Sanity-check the header before allocating.** The fingerprint comment asserted that each dimension was below 2^13, but the implementation never enforced that assertion. Before the first allocation of a request, require `sites`, `species`, `rank`, and `samples` to be at least 1, require `mode` to be 0 or 1, and reject any dimension product that would overflow the server's `Int` element or byte counts. Do not introduce an arbitrary workload cap in this subproject: the server is a local child process, Python validates its requests, and a fixed cap would silently define a new supported-size limit without evidence. On violation, write a diagnostic to stderr and exit non-zero. This is a fail-fast, not a protocol error response, so it needs no wire-format change.

**High-water retention is deliberate but sticky.** Because buffers never shrink, a single oversized request pins its footprint for the life of the worker, which is the life of the R session. `_WORKER.close()` remains the escape hatch, and the tutorial should say so once the fix ships.

This design removes the collision, the leak, and the dead scratch dimension without changing the binary protocol or introducing a separate allocator abstraction that would be difficult to express safely in the current Mojo version.

### 3. Automatic dtype eligibility

In `model_sjSDM.py`, automatic mode will include `self.dtype == torch.float32` in the Mojo eligibility predicate. `self.dtype` is a real `torch.dtype` after `_device_and_dtype`, so the comparison is well defined. A CPU float64 fit in auto mode therefore builds and uses `torch_tmp`.

Forced mode (`SJSDM_MOJO_BACKEND=1`) will continue selecting the Mojo function and will retain the bridge's explicit CPU-float32 guard. This distinction keeps auto mode convenient and forced mode diagnostic.

Tests will patch or select an available worker, build otherwise equivalent float32 and float64 models, and assert:

- auto + CPU float32 selects Mojo when the binary is available;
- auto + CPU float64 completes through PyTorch;
- forced Mojo + CPU float64 raises the documented `CPU float32` error.

These tests must save and restore the global Torch default dtype around each model construction. `Model_base.__init__` calls `torch.set_default_dtype(self.dtype)` and then `torch.set_default_tensor_type('torch.FloatTensor')`; a direct probe confirms that constructing a float64 model leaves the process default at float32. Without an explicit `try/finally` fixture that restores the previous value, test behaviour can depend on collection order. Repairing the model's global-default mutation is a package-wide concern and is not folded into this Mojo safety change.

### 4. One-shot transport execution

Replace the ambiguous conditional lambda with an explicit branch:

```python
if persistent:
    runner = lambda: _WORKER.run(mu, sigma, y, noise, alpha)
else:
    runner = lambda: _run_oneshot(mu, sigma, y, noise, alpha)
```

The current expression parses as a single lambda whose body is a conditional returning either the result tuple or another lambda, so in one-shot mode the caller unpacks a function object and raises `TypeError`, which the surrounding `except` clause does not catch.

Do not expand `mojo_logit_loss` to accept an explicit-noise tensor. Its model-facing contract takes a sample count, while explicit noise is an internal parity/testing facility already reachable through `_MojoLogitMCLoss.apply` and `_WORKER.run`. The public one-shot test below intentionally uses an integer sample count and RNG-state replay, so exposing a second public input type is unnecessary.

The one-shot integration test must set `SJSDM_MOJO_PERSISTENT=0` and call `mojo_logit_loss`, not `_run_oneshot` directly. Before the call it will save Torch's RNG state; afterwards it will restore that state and recreate the exact noise tensor that the public call consumed, using the same shape, dtype, and device as the internal `torch.randn`. It will then compare loss, `mu.grad`, and `sigma.grad` with the persistent explicit-noise path using the existing tolerances. Environment changes will be isolated with pytest's `monkeypatch`, and the global worker will be closed by the existing fixture.

### 5. Failure behaviour

No protocol-level error response is added in this subproject. Python remains responsible for tensor dtype, device, shape, sampling, and missing-response guards before a request is sent, and the server's header sanity gate in section 2 is a fail-fast rather than a reply.

**Validate where the bytes are written, not only at the public boundary.** The first draft placed all checks in `mojo_logit_loss`, but the payload is assembled and written in `_PersistentWorker._request`, and both the test suite and any internal caller can reach `_request` through `_WORKER.run` or `_WORKER.run_seed` without passing through the bridge function. Put the byte-count assertions in `_request`, so that the guard the invariants depend on cannot be bypassed by the very tests meant to prove it.

In `_request`, before writing anything, derive the expected byte count for each section from the header dimensions and compare it against the array's actual `nbytes`:

- `mu` must be `sites * species * 4` bytes, `sigma` `species * rank * 4`, `y` `sites * species * 4`;
- in mode 0, `noise` must be `samples * sites * rank * 4` bytes;
- every array must be CPU float32.

All four payload arrays can desynchronise the pipe: `sites` and `species` come from `mu`, `rank` comes only from `sigma.shape[1]`, and `samples` comes only from `noise.shape[0]`. A mismatched first sigma dimension, mismatched `y` shape, wrong noise trailing dimensions, or non-float32 dtype therefore changes the number of bytes written without changing the corresponding server read count. The O(1) checks above close that hole regardless of caller.

The bridge will additionally validate at its own boundary, where the error messages can be user-facing:

- `mu`, `sigma`, and `Ys` must be CPU float32 tensors;
- `mu`, `sigma`, and `Ys` dimensions must agree;
- `sampling` must be positive and integer-valued;
- `Ys` must not contain NaNs in forced/direct Mojo calls.

Invalid requests fail in Python with a `RuntimeError` before any bytes are written, preventing protocol desynchronisation. The internal model path normally satisfies these conditions, but the public bridge function must defend its own boundary.

If the persistent worker dies mid-request, the existing one-restart policy remains. A validation failure is not retryable and must not restart the worker.

### 6. Whole-operation missing-data selection

Automatic mode must not choose a backend independently for each minibatch. The current wrapper sends complete batches through Mojo and batches containing missing responses through PyTorch; those paths consume Torch's global RNG differently, so one fit has neither pure-backend stochastic semantics.

At the start of `fit()` and `logLik()`, inspect the complete response matrix once. In auto mode, build the PyTorch loss for the entire operation when any response is missing and build the Mojo loss otherwise. Forced Mojo mode remains strict and raises before iteration when any response is missing. The bridge keeps its own NaN check because it is a boundary guard for direct calls.

The current request construction makes multiple full-size Python byte copies, especially in legacy explicit-noise mode. That is a valid performance opportunity but not a safety prerequisite under default seed transport; record and benchmark direct buffered writes and `readinto` in the performance subproject.

## Testing strategy

### Focused regression tests

Add tests covering:

- seed transport at 262,143, 262,144, and 262,145 requested values;
- a request substantially above the boundary;
- seeded output independence from an intervening explicit-noise buffer overwrite;
- the historical `(rank=2, samples=1)` / `(rank=1, samples=8193)` fingerprint collision sequence, whose packed keys both evaluate to 16385;
- repeated growth and shrinkage across sites, species, rank, and samples;
- public one-shot forward and backward execution;
- auto float64 fallback and forced float64 rejection, with the global default dtype restored;
- auto mode with any missing response uses PyTorch for the whole `fit()` or `logLik()` operation, while forced mode rejects it before iteration;
- mismatched `mu`, `sigma`, and `Y` shapes failing before transport;
- an explicit noise tensor of the wrong shape, and one of dtype float64, each failing in `_request` before any byte is written, with the worker still usable for a correct request afterwards.

The scratch-buffer resizing in section 2 must be proven non-behavioural, not merely tested for plausibility. As a release-validation check, run the pre-change and post-change binaries with identical explicit-noise payloads and compare the complete response bytes at one shape where `sites` greatly exceeds `N_CHUNKS` and one where it does not. Do not make the standard suite depend on an old platform-specific binary or a bitwise fixture; committed tests continue to compare numerically with the PyTorch reference.

### Memory stability gate

Use an integration check on Apple Silicon with `psutil`, reading the worker PID from `mojo_bridge._WORKER.proc.pid`:

1. Start one persistent worker.
2. Warm it with both shapes used by the alternating test.
3. Record worker RSS.
4. Alternate the shapes at least 20 additional times.
5. Require RSS growth after warm-up to remain below 4 MiB.

The first draft's 16 MiB threshold was calibrated against the old sizing. Under capacity semantics with the scratch fix, post-warm-up growth should be approximately zero, so a loose bound would pass even if the capacity logic regressed. 4 MiB still absorbs operating-system RSS accounting variance.

Add a second, absolute check that would have caught the dead scratch dimension, since a growth-only gate cannot: after five requests at `sites=4096, species=200, rank=8, samples=25`, require worker RSS below 40 MiB. The independent review measured 108.2 MB before and 26.3 MB after; reproduction measured 103.2 MiB before and 25.1 MiB after. The threshold is calibrated for the target Apple Silicon machine and should be recorded as such.

Both are release-validation gates rather than universally strict unit assertions, because operating-system RSS accounting can vary. The functional tests still exercise the same shape sequences in the standard pytest suite.

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
- the memory, byte-equality, and large-request numerical gates pass.

## Documentation and benchmark handling

Until all gates pass, retain the tutorial's v0.2.0 safety warning and do not publish shape-grid results above the old seed limit. Once the corrected implementation is verified:

1. Replace the workaround language with a statement that v0.2.0 contains the limitation and the next release fixes it.
2. Update `AGENTS.md`, whose 2026-08-22 entry instructs that any future per-shape caching must key on the full shape fingerprint. This design removes keying entirely in favour of per-buffer capacity, so that guidance is superseded and will mislead if left standing. Record the new rule in its place.
3. Rerun all shape-grid and end-to-end benchmarks from clean worker processes. Note that the scratch fix changes the memory profile substantially, so prior RSS observations in `AGENTS.md` are no longer comparable and should be labelled by build.
4. Keep historical v0.2.0 numbers labelled by release rather than silently replacing them.

Performance optimisation begins only after this corrected build supplies the new baseline.

## Acceptance criteria

- A seeded request of any tested valid size is independent of prior worker contents.
- The historical shape-key collision sequence matches a fresh worker within numerical tolerance.
- Alternating warmed shapes does not exhibit monotonic per-request RSS growth.
- Scratch-buffer resizing is byte-identical to the pre-change binary on the pinned reference shapes.
- Worker RSS at the heavy reference shape is below the absolute gate.
- A malformed noise tensor is rejected before transport and leaves the worker usable.
- Non-positive, invalid-mode, or overflow-producing header values cause a clean, diagnosable worker exit rather than an allocation attempt.
- Auto float64 fits use PyTorch and complete; forced float64 fits fail clearly, and the suite's global default dtype is unchanged afterwards.
- Public one-shot mode completes forward and backward through `mojo_logit_loss`.
- Large explicit-noise loss and gradients satisfy the numerical gate.
- Existing persistent transport, restart, NaN, probit-alpha, and autograd tests remain green.
- No wire-format change is required by existing Python or Mojo clients.

## Review outcomes

Changes accepted after review, with the evidence behind each:

1. **Scratch-buffer sizing corrected.** `zbuf_all` and `llbuf_all` no longer carry a per-site dimension. Static indexing analysis and an independent A/B build both show byte-identical output and a 76% worker-RSS reduction at the heavy reference shape.
2. **All payload validation added at `_request`.** `mu`, `sigma`, `y`, and explicit noise can each disagree with header-derived byte counts. Validating where bytes are written prevents protocol desynchronisation even when internal test helpers bypass `mojo_logit_loss`.
3. **Server header validation added.** Positive dimensions, valid mode, and checked size arithmetic are required before allocation. The old shape-fingerprint comment was not an enforced bound, so no arbitrary fixed workload cap is inferred from it.
4. **Pointer lifecycle corrected for pinned Mojo.** The implementation will free the initial zero-length allocations, each replaced allocation, and all final buffers using `unsafe_free()`. Direct compiler probes show that the review's nullable-`Pointer`/`dealloc(pointer)` formulation does not compile in Mojo 1.0.0.
5. **Global-dtype hazard covered by test isolation.** A probe confirms that float64 model construction leaves Torch's process default at float32 because `set_default_tensor_type` overrides the preceding call.
6. **Memory gates strengthened.** Post-warm-up growth is tightened to 4 MiB and a target-machine absolute 40 MiB gate is added, because a growth-only test cannot detect systematic scratch over-allocation.
7. **Runtime noise worker count accepted.** A pinned-toolchain compiler probe confirms `parallelize` accepts a runtime `Int`; the first draft's compile-time-only claim was incorrect.
8. **Missing-data backend choice hoisted.** Auto mode chooses PyTorch for the whole operation when any response is missing, avoiding per-minibatch backend and RNG-semantics mixing.
9. **Index compatibility time-boxed.** The safety change keeps the current mapping for reviewability, while site-major seed-noise layout remains open for measured performance work.
10. **`AGENTS.md` and one-shot divergence added to documentation scope.** Existing fingerprint guidance will be superseded, and the revived one-shot path is correctly treated as an independent implementation.

Suggestions deliberately deferred or rejected:

1. **Cache-line padding is deferred.** Rounding strides without proving base alignment does not guarantee isolation, and the supplied benchmark did not isolate padding from the much larger scratch-size correction.
2. **Public explicit-noise input is rejected.** It is an internal parity facility, and the public one-shot integration test works with integer sampling plus RNG-state replay. Expanding the model-facing contract is unnecessary.
3. **Python transport-copy tuning is deferred.** The copies are real, but default seed transport does not send the dominant noise tensor and repository history already says pipe payload is not the current bottleneck. It belongs in the performance subproject with benchmarks.
4. **A fixed server element cap is rejected for now.** Checked arithmetic and Python-side request validation address correctness without inventing an unsupported maximum workload. A resource-policy cap can be added later if deployment requirements justify one.
5. **A committed old-binary byte fixture is rejected.** Cross-binary byte equality remains a release gate; the normal suite should not depend on a historical platform-specific executable or brittle bitwise fixture.

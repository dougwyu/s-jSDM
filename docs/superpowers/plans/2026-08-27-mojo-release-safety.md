# Mojo Release Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Mojo backend correct and resource-safe for supported CPU float32 logit/probit fits, including large seeded requests, repeated shape changes, auto fallback, and one-shot transport.

**Architecture:** Keep the existing Python autograd interface and 40-byte persistent wire protocol. Repair the Mojo worker with indexed strided RNG generation, checked header arithmetic, independently capacity-managed buffers, and per-chunk reusable scratch; defend the protocol at the Python write boundary and select the backend once per fit/log-likelihood operation. Keep release-only byte/RSS checks in a dedicated validation script so portable pytest tests do not depend on historical binaries or platform-specific memory accounting.

**Tech Stack:** R, Python 3.12, PyTorch 2.5.1, NumPy, pytest 9, Mojo 1.0.0, MAX 26.5.0, pixi, psutil, reticulate, testthat, rmarkdown.

**Spec:** `docs/superpowers/specs/2026-08-27-mojo-release-safety-design.md`

## Global Constraints

- Preserve the existing R API, Python loss call signature, and persistent request/response byte layouts.
- Mojo remains limited to CPU float32 logit/probit loss evaluation; GPU/MPS, weighted upstream gradients, and Hessian/double-backprop support remain out of scope.
- Forced mode (`SJSDM_MOJO_BACKEND=1`) is strict; auto mode may fall back to PyTorch.
- Seed transport need not match PyTorch's random stream, but must be deterministic within a build and independent of prior worker contents.
- Do not change `work/mojo-backend/mojo/mc_logit_grad.mojo`; it is the independent one-shot implementation.
- Do not add a fixed workload-size cap. Reject non-positive dimensions, invalid mode values, and any element or byte-count multiplication that would overflow `Int`.
- Do not add cache-line padding, a public explicit-noise argument, or Python transport-copy optimisation in this subproject.
- Do not commit generated worker binaries, pixi lockfiles, rendered HTML, benchmark CSVs, or pytest caches.
- Build Mojo with `-O 3`; unoptimised binaries are not valid for performance or release checks.
- Keep historical v0.2.0 results labelled as historical; do not silently replace them with development-build results.

## File Map

- Modify `work/mojo-backend/mojo/mc_logit_grad_server.mojo`: RNG coverage, checked header sizes, buffer capacities, scratch sizing, and pointer cleanup.
- Modify `sjSDM/inst/python/sjSDM_py/mojo_bridge.py`: request validation, non-retryable validation errors, and one-shot runner selection.
- Modify `sjSDM/inst/python/sjSDM_py/model_sjSDM.py`: dtype eligibility and whole-operation missing-data selection.
- Modify `sjSDM/inst/python/tests/test_mojo_bridge.py`: worker, protocol, autograd, and one-shot regressions.
- Modify `sjSDM/inst/python/tests/test_sjSDM.py`: model-selection regressions and global-dtype isolation.
- Create `Code/dev/benchmarks/validate_mojo_release.py`: target-machine byte parity, numerical parity, and RSS gates.
- Modify `Code/README.md`: document the release-validation command.
- Modify `Code/sjSDM_mojo_tutorial.Rmd`: replace v0.2.0 workarounds after verification and document high-water worker memory.
- Modify `README.Rmd`, then regenerate `README.md`: distinguish the v0.2.0 limitation from the corrected development branch.
- Modify `CHANGELOG.md`: add an Unreleased safety-fix section.
- Modify `AGENTS.md`: replace the obsolete packed-shape guidance and label old memory observations by build.

---

### Task 1: Generate every seeded noise value

**Files:**
- Modify: `sjSDM/inst/python/tests/test_mojo_bridge.py:200-243`
- Modify: `work/mojo-backend/mojo/mc_logit_grad_server.mojo:37-77`

**Interfaces:**
- Consumes: the unchanged mode-1 request body containing one little-endian `UInt64` seed.
- Produces: `gen_noise(noise, n, seed)` that fills every index in `[0, n)` exactly once using at most `N_GEN_BLOCKS` runtime workers.

- [ ] **Step 1: Preserve an optimised pre-change binary for later A/B validation**

Run from the repository root:

```bash
cd work/mojo-backend
pixi run mojo build -O 3 mojo/mc_logit_grad_server.mojo \
  -o /private/tmp/sjsdm-mojo-v0.2.0-server
cd ../..
shasum -a 256 /private/tmp/sjsdm-mojo-v0.2.0-server
```

Expected: build exit 0 and one SHA-256 line. Do not replace this file during later tasks.

- [ ] **Step 2: Add the boundary and poisoned-tail regression**

Add this test to `TestSeedTransport`:

```python
    @pytest.mark.parametrize("n", [262_143, 262_144, 262_145, 524_289])
    def test_seeded_request_overwrites_the_entire_noise_buffer(self, n):
        mu = torch.zeros((n, 1), dtype=torch.float32)
        sigma = torch.ones((1, 1), dtype=torch.float32)
        y = torch.ones((n, 1), dtype=torch.float32)

        def seeded_after_poison(value):
            poison = torch.full((1, n, 1), value, dtype=torch.float32)
            run_worker(mu, sigma, y, poison, 1.0)
            return run_worker_seed(mu, sigma, y, 1, 91)

        after_zero = seeded_after_poison(0.0)
        after_one = seeded_after_poison(1.0)
        for left, right in zip(after_zero, after_one):
            assert np.array_equal(left, right)
```

Using sites rather than samples as the large dimension makes every tail noise value affect a separate returned loss, avoiding a weak assertion hidden by a large Monte Carlo reduction.

- [ ] **Step 3: Run the new test against the v0.2.0 worker and confirm the defect**

Run:

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestSeedTransport::test_seeded_request_overwrites_the_entire_noise_buffer \
  -vv
```

Expected: the 262,143 and 262,144 cases pass; at least the 262,145 and 524,289 cases fail because the poisoned tail survives seed generation.

- [ ] **Step 4: Implement strided block coverage and a runtime worker count**

Replace the body structure of `gen_noise` with this indexing scheme while retaining the existing Marsaglia-polar inner calculation verbatim:

```mojo
def gen_noise(mut noise: Pointer[Float32, MutUntrackedOrigin], n: Int, seed: UInt64):
    var n_workers = min(N_GEN_BLOCKS, (n + BLOCK - 1) // BLOCK)

    @parameter
    def fill(worker_id: Int):
        var start = worker_id * BLOCK
        var stride = n_workers * BLOCK
        while start < n:
            var stop = min(start + BLOCK, n)
            var idx = start
            while idx < stop:
                var attempt: UInt64 = 0
                while True:
                    var u1f = Float32((splitmix64(seed ^ UInt64(idx) ^ (attempt << 32)) >> 8) & 0xFFFFFF) * (1.0 / 8388608.0) - 1.0
                    var u2f = Float32((splitmix64(seed ^ UInt64(idx + 1) ^ (attempt << 32)) >> 8) & 0xFFFFFF) * (1.0 / 8388608.0) - 1.0
                    var ss = u1f * u1f + u2f * u2f
                    if ss < 1.0 and ss > 0.0:
                        var fac = sqrt(-2.0 * log(ss) / ss)
                        noise[idx] = u1f * fac
                        if idx + 1 < n:
                            noise[idx + 1] = u2f * fac
                        break
                    attempt += 1
                idx += 2
            start += stride

    parallelize[fill](n_workers)
```

`BLOCK` is even, so every block start and stride remain even and the existing pair mapping is preserved.

- [ ] **Step 5: Build the candidate and rerun the focused test**

Run:

```bash
cd work/mojo-backend
pixi run mojo build -O 3 mojo/mc_logit_grad_server.mojo -o mc_grad_server_bin
cd ../..
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestSeedTransport \
  -q
```

Expected: build exit 0 and all `TestSeedTransport` cases pass.

- [ ] **Step 6: Commit the RNG repair**

```bash
git add work/mojo-backend/mojo/mc_logit_grad_server.mojo \
  sjSDM/inst/python/tests/test_mojo_bridge.py
git commit -m "fix: fill complete Mojo seed buffers"
```

---

### Task 2: Replace packed shape keys with bounded buffer capacities

**Files:**
- Modify: `sjSDM/inst/python/tests/test_mojo_bridge.py:120-150`
- Modify: `work/mojo-backend/mojo/mc_logit_grad_server.mojo:92-291`

**Interfaces:**
- Consumes: checked per-request element counts introduced fully in Task 3; until then compute the same counts directly.
- Produces: one high-water capacity per allocation, `zbuf_all` slots of `samples * species`, and `llbuf_all` slots of `samples`.

- [ ] **Step 1: Add the historical collision regression**

Add to `TestMixedShapeRequests`:

```python
    def test_historical_packed_shape_collision(self):
        mu1, sigma1, y1, _ = make_case(1, 1, 2, 1, seed=71)
        mu2, sigma2, y2, _ = make_case(1, 1, 1, 8193, seed=73)

        run_worker_seed(mu1, sigma1, y1, 1, 101)
        collided = run_worker_seed(mu2, sigma2, y2, 8193, 103)

        mojo_bridge._WORKER.close()
        fresh = run_worker_seed(mu2, sigma2, y2, 8193, 103)

        for left, right in zip(collided, fresh):
            np.testing.assert_allclose(left, right, atol=1e-4, rtol=1e-5)
```

Also update the class docstring to say buffers are capacity-managed rather than keyed by a fingerprint.

- [ ] **Step 2: Run the collision test and confirm it fails**

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestMixedShapeRequests::test_historical_packed_shape_collision \
  -vv
```

Expected: FAIL because `(rank=2, samples=1)` and `(rank=1, samples=8193)` both produce packed key 16385 and the second request reuses undersized buffers.

- [ ] **Step 3: Introduce independent capacities and exact required sizes**

Replace `cap_shape` with these capacity variables next to the existing zero-length allocations:

```mojo
    var cap_mu = 0
    var cap_sigma = 0
    var cap_y = 0
    var cap_noise = 0
    var cap_out = 0
    var cap_gmu = 0
    var cap_gsigma_buf = 0
    var cap_gsigma = 0
    var cap_zbuf_all = 0
    var cap_llbuf_all = 0
```

For each request compute:

```mojo
        var n_mu = sites * species
        var n_sigma = species * rank
        var n_noise = samples * sites * rank
        var n_out = sites
        var n_gsigma_buf = N_CHUNKS * species * rank
        var n_zbuf_all = N_CHUNKS * samples * species
        var n_llbuf_all = N_CHUNKS * samples
```

Use this exact growth branch for every pointer/capacity pair:

```mojo
        if n_mu > cap_mu:
            mu.unsafe_free()
            mu = alloc[Float32](n_mu)
            cap_mu = n_mu
        if n_sigma > cap_sigma:
            sigma.unsafe_free()
            sigma = alloc[Float32](n_sigma)
            cap_sigma = n_sigma
        if n_mu > cap_y:
            y.unsafe_free()
            y = alloc[Float32](n_mu)
            cap_y = n_mu
        if n_noise > cap_noise:
            noise.unsafe_free()
            noise = alloc[Float32](n_noise)
            cap_noise = n_noise
        if n_out > cap_out:
            out.unsafe_free()
            out = alloc[Float32](n_out)
            cap_out = n_out
        if n_mu > cap_gmu:
            gmu.unsafe_free()
            gmu = alloc[Float32](n_mu)
            cap_gmu = n_mu
        if n_gsigma_buf > cap_gsigma_buf:
            gsigma_buf.unsafe_free()
            gsigma_buf = alloc[Float32](n_gsigma_buf)
            cap_gsigma_buf = n_gsigma_buf
        if n_sigma > cap_gsigma:
            gsigma.unsafe_free()
            gsigma = alloc[Float32](n_sigma)
            cap_gsigma = n_sigma
        if n_zbuf_all > cap_zbuf_all:
            zbuf_all.unsafe_free()
            zbuf_all = alloc[Float32](n_zbuf_all)
            cap_zbuf_all = n_zbuf_all
        if n_llbuf_all > cap_llbuf_all:
            llbuf_all.unsafe_free()
            llbuf_all = alloc[Float32](n_llbuf_all)
            cap_llbuf_all = n_llbuf_all
```

Do not shrink when `required <= capacity`.

- [ ] **Step 4: Reuse one scratch slot per concurrent chunk**

Change the chunk-local bases to:

```mojo
            var zbuf = zbuf_all + cid * samples * species
            var llbuf = llbuf_all + cid * samples

            for site in range(start, stop):
                var zbase = 0
                var llbase = 0
```

The site loop is serial within each `chunk_loss` invocation, so later sites overwrite scratch only after the previous site's pass 2 has finished.

- [ ] **Step 5: Free every allocation on clean EOF**

After the request loop, add exactly one `unsafe_free()` call for each of `hbuf`, `seedbuf`, `mu`, `sigma`, `y`, `noise`, `out`, `gmu`, `gsigma_buf`, `gsigma`, `zbuf_all`, and `llbuf_all`. This includes the initial zero-length allocations if the worker receives no request.

- [ ] **Step 6: Build and run mixed-shape coverage**

```bash
cd work/mojo-backend
pixi run mojo build -O 3 mojo/mc_logit_grad_server.mojo -o mc_grad_server_bin
cd ../..
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestMixedShapeRequests \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestPersistentProtocol \
  -q
```

Expected: all selected tests pass; no worker crash or non-finite output.

- [ ] **Step 7: Commit capacity management**

```bash
git add work/mojo-backend/mojo/mc_logit_grad_server.mojo \
  sjSDM/inst/python/tests/test_mojo_bridge.py
git commit -m "fix: bound Mojo worker buffer reuse"
```

---

### Task 3: Validate requests before transport or allocation

**Files:**
- Modify: `sjSDM/inst/python/tests/test_mojo_bridge.py:1-280`
- Modify: `sjSDM/inst/python/sjSDM_py/mojo_bridge.py:38-278`
- Modify: `work/mojo-backend/mojo/mc_logit_grad_server.mojo:37-170`

**Interfaces:**
- Produces: `_RequestValidationError(RuntimeError)`, `_positive_integer(value, name) -> int`, and `_request` guards that run before `_start()` or `stdin.write()`.
- Produces: Mojo `checked_mul(a: Int, b: Int) raises -> Int` and fail-fast header validation.
- Consumes: Task 2 capacity variables and required element counts.

- [ ] **Step 1: Add Python write-boundary tests that cannot hang or desynchronise a real worker**

Add the following helper and tests:

```python
class _NeverWrite:
    def write(self, _payload):
        pytest.fail("invalid request reached stdin.write")

    def flush(self):
        pytest.fail("invalid request reached stdin.flush")


class _NoIOProcess:
    stdin = _NeverWrite()

    def poll(self):
        return None


@pytest.mark.parametrize(
    "field",
    ["mu_dtype", "sigma_dtype", "y_dtype", "sigma_shape", "y_shape",
     "noise_dtype", "noise_shape"],
)
def test_invalid_payload_is_rejected_before_write(field):
    mu, sigma, y, noise = make_case(8, 4, 2, 5, seed=79)
    if field == "mu_dtype":
        mu = mu.double()
    elif field == "sigma_dtype":
        sigma = sigma.double()
    elif field == "y_dtype":
        y = y.double()
    elif field == "sigma_shape":
        sigma = torch.zeros((5, 2), dtype=torch.float32)
    elif field == "y_shape":
        y = torch.zeros((7, 4), dtype=torch.float32)
    elif field == "noise_dtype":
        noise = noise.double()
    else:
        noise = torch.zeros((5, 8, 3), dtype=torch.float32)

    worker = mojo_bridge._PersistentWorker()
    worker.proc = _NoIOProcess()
    with pytest.raises(RuntimeError, match="Mojo request"):
        worker.run(mu, sigma, y, noise, 1.0)
```

Add a public sampling guard without invoking the current broken worker path:

```python
@pytest.mark.parametrize("sampling", [0, -1, 1.5])
def test_public_loss_rejects_invalid_sampling(monkeypatch, sampling):
    mu, sigma, y, _ = make_case(8, 4, 2, 5, seed=83)
    monkeypatch.setattr(
        mojo_bridge._MojoLogitMCLoss,
        "apply",
        lambda *args: pytest.fail("invalid sampling reached autograd apply"),
    )
    with pytest.raises(RuntimeError, match="positive integer"):
        mojo_bridge.mojo_logit_loss(mu, y, sigma, sampling, 1.0)


@pytest.mark.parametrize("field", ["sigma_dtype", "y_dtype", "sigma_shape", "y_shape"])
def test_public_loss_validates_every_model_tensor(monkeypatch, field):
    mu, sigma, y, _ = make_case(8, 4, 2, 5, seed=87)
    if field == "sigma_dtype":
        sigma = sigma.double()
    elif field == "y_dtype":
        y = y.double()
    elif field == "sigma_shape":
        sigma = torch.zeros((5, 2), dtype=torch.float32)
    else:
        y = torch.zeros((7, 4), dtype=torch.float32)
    monkeypatch.setattr(
        mojo_bridge._MojoLogitMCLoss,
        "apply",
        lambda *args: pytest.fail("invalid tensor reached autograd apply"),
    )
    with pytest.raises(RuntimeError, match="Mojo request"):
        mojo_bridge.mojo_logit_loss(mu, y, sigma, 5, 1.0)
```

- [ ] **Step 2: Run the Python guards and confirm they fail before implementation**

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py \
  -k "invalid_payload or invalid_sampling or validates_every_model_tensor" -vv
```

Expected: FAIL because the current bridge neither validates all payload fields nor rejects non-integer/zero sampling before transport.

- [ ] **Step 3: Implement Python validation with a non-retryable exception**

Add these internal definitions near the top of `mojo_bridge.py`:

```python
class _RequestValidationError(RuntimeError):
    """A local request is invalid; restarting the worker cannot fix it."""


def _positive_integer(value, name):
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _RequestValidationError(
            f"Mojo request {name} must be a positive integer."
        ) from exc
    if converted <= 0 or converted != value:
        raise _RequestValidationError(
            f"Mojo request {name} must be a positive integer."
        )
    return converted


def _validate_tensor(name, tensor, expected_shape):
    if not isinstance(tensor, torch.Tensor):
        raise _RequestValidationError(f"Mojo request {name} must be a tensor.")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise _RequestValidationError(
            f"Mojo request {name} must be a CPU float32 tensor."
        )
    if tuple(tensor.shape) != tuple(expected_shape):
        raise _RequestValidationError(
            f"Mojo request {name} has shape {tuple(tensor.shape)}; "
            f"expected {tuple(expected_shape)}."
        )
    array = tensor.detach().numpy()
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 4
    if array.nbytes != expected_bytes:
        raise _RequestValidationError(
            f"Mojo request {name} has {array.nbytes} bytes; "
            f"expected {expected_bytes}."
        )
    return array
```

At the beginning of `_request`, before checking or starting `self.proc`, validate rank and shape, then cache the returned arrays for payload creation:

```python
        samples = _positive_integer(samples, "sampling")
        if not isinstance(mu, torch.Tensor) or mu.ndim != 2:
            raise _RequestValidationError("Mojo request mu must be a rank-2 tensor.")
        if not isinstance(sigma, torch.Tensor) or sigma.ndim != 2:
            raise _RequestValidationError("Mojo request sigma must be a rank-2 tensor.")
        sites, species = mu.shape
        rank = sigma.shape[1]
        if sites < 1 or species < 1 or rank < 1:
            raise _RequestValidationError(
                "Mojo request dimensions must be positive."
            )
        mu_array = _validate_tensor("mu", mu, (sites, species))
        sigma_array = _validate_tensor("sigma", sigma, (species, rank))
        y_array = _validate_tensor("y", y, (sites, species))
        noise_array = None
        if noise_tensor is not None:
            noise_array = _validate_tensor(
                "noise", noise_tensor, (samples, sites, rank)
            )
```

Use `mu_array`, `sigma_array`, `y_array`, and `noise_array` when constructing the existing payload. In `mojo_logit_loss`, call `_positive_integer` and validate all three model tensors/shapes before the NaN check so the one-shot path receives the same boundary protection.

- [ ] **Step 4: Prove validation errors are not retried**

After `_RequestValidationError` exists, add:

```python
def test_request_validation_error_is_not_retried(monkeypatch):
    mu, sigma, y, noise = make_case(8, 4, 2, 5, seed=89)
    calls = 0
    monkeypatch.setenv("SJSDM_MOJO_PERSISTENT", "1")

    def reject(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise mojo_bridge._RequestValidationError("Mojo request rejected")

    monkeypatch.setattr(mojo_bridge._WORKER, "run", reject)
    with pytest.raises(mojo_bridge._RequestValidationError):
        mojo_bridge._MojoLogitMCLoss.apply(mu, sigma, y, noise, 1.0)
    assert calls == 1
```

Run it before changing the retry block. Expected: FAIL with `calls == 2`. Then add an earlier exception arm in `_MojoLogitMCLoss.forward`:

```python
        try:
            out, gmu, gsigma = runner()
        except _RequestValidationError:
            raise
        except (RuntimeError, BrokenPipeError, subprocess.CalledProcessError):
            if not persistent:
                raise
            _WORKER.close()
            out, gmu, gsigma = runner()
```

- [ ] **Step 5: Add safe RED tests for server header validation**

Import `struct` in `test_mojo_bridge.py`, then add:

```python
def _raw_server_request(header, body=b""):
    return subprocess.run(
        [SERVER_BIN], input=header + body, capture_output=True, timeout=5
    )


@pytest.mark.parametrize(
    ("header", "body"),
    [
        (struct.pack("<qqqqfI", 0, 1, 1, 1, 1.0, 1),
         np.zeros(1, dtype=np.float32).tobytes() + struct.pack("<Q", 1)),
        (struct.pack("<qqqqfI", 1, 1, 1, 1, 1.0, 9),
         np.zeros(4, dtype=np.float32).tobytes()),
    ],
)
def test_server_rejects_invalid_header(header, body):
    result = _raw_server_request(header, body)
    assert result.returncode != 0
    assert b"invalid request header" in result.stderr
```

Run against the current candidate before adding the server guard. Expected: both cases FAIL safely with return code 0. Do not send an overflowing valid-mode header to the unguarded v0.2.0 worker because it may attempt a huge allocation.

- [ ] **Step 6: Implement checked server sizes before allocation**

Add:

```mojo
def checked_mul(a: Int, b: Int) raises -> Int:
    if a > Int.MAX // b:
        raise Error("invalid request header: dimension product overflow")
    return a * b
```

Immediately after decoding the header, before any request-sized allocation, add:

```mojo
        if sites < 1 or species < 1 or rank < 1 or samples < 1:
            raise Error("invalid request header: dimensions must be positive")
        if mode != 0 and mode != 1:
            raise Error("invalid request header: mode must be 0 or 1")

        var n_mu = checked_mul(sites, species)
        var n_sigma = checked_mul(species, rank)
        var n_noise = checked_mul(checked_mul(samples, sites), rank)
        var n_out = sites
        var n_gsigma_buf = checked_mul(N_CHUNKS, n_sigma)
        var n_zbuf_all = checked_mul(checked_mul(N_CHUNKS, samples), species)
        var n_llbuf_all = checked_mul(N_CHUNKS, samples)

        _ = checked_mul(n_mu, 4)
        _ = checked_mul(n_sigma, 4)
        _ = checked_mul(n_noise, 4)
        _ = checked_mul(n_out, 4)
```

Reuse these values in allocation, read, zeroing, indexing bounds, and response byte counts instead of recomputing unchecked products.

- [ ] **Step 7: Add and run the post-guard overflow regression**

Only after Step 6 and rebuilding the candidate, extend the header parameter list with:

```python
(struct.pack("<qqqqfI", 2**62, 4, 1, 1, 1.0, 1), b""),
```

Run:

```bash
cd work/mojo-backend
pixi run mojo build -O 3 mojo/mc_logit_grad_server.mojo -o mc_grad_server_bin
cd ../..
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py \
  -k "invalid_payload or invalid_sampling or validates_every_model_tensor or validation_error or invalid_header" \
  -q
```

Expected: all selected tests pass, including overflow rejection without an allocation attempt.

- [ ] **Step 8: Commit protocol validation**

```bash
git add work/mojo-backend/mojo/mc_logit_grad_server.mojo \
  sjSDM/inst/python/sjSDM_py/mojo_bridge.py \
  sjSDM/inst/python/tests/test_mojo_bridge.py
git commit -m "fix: validate Mojo protocol requests"
```

---

### Task 4: Restore the public one-shot autograd path

**Files:**
- Modify: `sjSDM/inst/python/tests/test_mojo_bridge.py:259-275`
- Modify: `sjSDM/inst/python/sjSDM_py/mojo_bridge.py:190-220`

**Interfaces:**
- Consumes: `mojo_logit_loss(mu, Ys, sigma, sampling: int, alpha)` and the internal explicit-noise `_MojoLogitMCLoss.apply` path.
- Produces: correct callable selection for `SJSDM_MOJO_PERSISTENT=0` without changing the public sampling argument.

- [ ] **Step 1: Add a public forward/backward one-shot integration test**

Add to `TestOneshotTransport`:

```python
    def test_public_oneshot_forward_and_backward(self, monkeypatch):
        sites, species, rank, samples = 24, 8, 3, 16
        mu, sigma, y, _ = make_case(sites, species, rank, samples, seed=97)

        mu_one = mu.clone().requires_grad_(True)
        sigma_one = sigma.clone().requires_grad_(True)
        torch.manual_seed(101)
        rng_state = torch.get_rng_state()
        monkeypatch.setenv("SJSDM_MOJO_PERSISTENT", "0")
        loss_one = mojo_bridge.mojo_logit_loss(
            mu_one, y, sigma_one, samples, 1.0
        )
        loss_one.mean().backward()

        torch.set_rng_state(rng_state)
        noise = torch.randn(
            (samples, sites, rank), device="cpu", dtype=torch.float32
        )
        mu_persistent = mu.clone().requires_grad_(True)
        sigma_persistent = sigma.clone().requires_grad_(True)
        monkeypatch.setenv("SJSDM_MOJO_PERSISTENT", "1")
        loss_persistent = mojo_bridge._MojoLogitMCLoss.apply(
            mu_persistent, sigma_persistent, y, noise, 1.0
        )
        loss_persistent.mean().backward()

        assert torch.allclose(loss_one, loss_persistent, atol=1e-4)
        assert torch.allclose(mu_one.grad, mu_persistent.grad, atol=1e-4)
        assert torch.allclose(
            sigma_one.grad, sigma_persistent.grad, atol=5e-4
        )
```

- [ ] **Step 2: Run it and confirm the conditional-lambda failure**

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestOneshotTransport::test_public_oneshot_forward_and_backward \
  -vv
```

Expected: FAIL with `TypeError` while unpacking a function object.

- [ ] **Step 3: Replace the ambiguous lambda expression**

In the non-seed branch of `_MojoLogitMCLoss.forward`, replace the conditional lambda with:

```python
            if persistent:
                runner = lambda: _WORKER.run(mu, sigma, y, noise, alpha)
            else:
                runner = lambda: _run_oneshot(mu, sigma, y, noise, alpha)
```

Do not change `mojo_logit_loss` to accept a tensor-valued `sampling` argument.

- [ ] **Step 4: Run all one-shot and autograd tests**

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestOneshotTransport \
  sjSDM/inst/python/tests/test_mojo_bridge.py::TestAutogradIntegration \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the one-shot repair**

```bash
git add sjSDM/inst/python/sjSDM_py/mojo_bridge.py \
  sjSDM/inst/python/tests/test_mojo_bridge.py
git commit -m "fix: restore Mojo one-shot autograd"
```

---

### Task 5: Select Mojo only for eligible whole operations

**Files:**
- Modify: `sjSDM/inst/python/tests/test_sjSDM.py:1-45`
- Modify: `sjSDM/inst/python/sjSDM_py/model_sjSDM.py:390-575,884-958`

**Interfaces:**
- Produces: `_loss_function_for_data(Y, train=True, individual=False) -> Callable`.
- Consumes: `_build_loss_function(..., allow_mojo: bool)` and `mojo_logit_loss` from Tasks 3-4.
- Guarantees: auto uses Mojo only on CPU float32 with no missing response anywhere in the operation; forced mode reports unsupported input before iteration.

- [ ] **Step 1: Isolate Torch's process-wide default dtype in tests**

Import `torch` in `test_sjSDM.py` and add:

```python
@pytest.fixture
def preserve_default_dtype():
    previous = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
```

Extend `model_base._get` with trailing `device="cpu", dtype="float32"` arguments and construct the model with:

```python
        model = fa.Model_sjSDM(device=device, dtype=dtype)
```

- [ ] **Step 2: Add dtype-selection regressions**

Add tests that monkeypatch the bridge before model construction:

```python
def test_auto_float64_uses_torch(
    monkeypatch, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "auto")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    monkeypatch.setattr(
        mojo_bridge,
        "mojo_logit_loss",
        lambda *args: pytest.fail("float64 auto mode selected Mojo"),
    )
    model = model_base(inp=2, out=3, df=2, dtype="float64")
    mu = torch.zeros((4, 3), dtype=torch.float64)
    y = torch.zeros((4, 3), dtype=torch.float64)
    loss = model._loss_function(
        mu, y, model.sigma, 4, 5, 2, model.alpha, "cpu", model.dtype
    )
    assert loss.dtype == torch.float64


def test_auto_float32_uses_mojo(
    monkeypatch, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    calls = 0

    def fake_mojo(mu, y, sigma, sampling, alpha):
        nonlocal calls
        calls += 1
        return mu.sum(dim=1) * 0.0

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "auto")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    monkeypatch.setattr(mojo_bridge, "mojo_logit_loss", fake_mojo)
    model = model_base(inp=2, out=3, df=2, dtype="float32")
    model._loss_function(
        torch.zeros((4, 3)), torch.zeros((4, 3)), model.sigma,
        4, 5, 2, model.alpha, "cpu", model.dtype,
    )
    assert calls == 1


def test_forced_float64_reaches_strict_bridge_guard(
    monkeypatch, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "1")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    model = model_base(inp=2, out=3, df=2, dtype="float64")
    with pytest.raises(RuntimeError, match="CPU float32"):
        model._loss_function(
            torch.zeros((4, 3), dtype=torch.float64),
            torch.zeros((4, 3), dtype=torch.float64),
            model.sigma, 4, 5, 2, model.alpha, "cpu", model.dtype,
        )
```

- [ ] **Step 3: Run dtype tests and verify auto float64 fails today**

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider sjSDM/inst/python/tests/test_sjSDM.py \
  -k "auto_float or forced_float" -vv
```

Expected: auto float64 fails with the bridge's CPU-float32 error; the float32 and forced-mode expectations document current behavior.

- [ ] **Step 4: Add whole-operation missing-data tests**

```python
def test_auto_missing_data_never_calls_mojo(
    monkeypatch, data, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "auto")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    monkeypatch.setattr(
        mojo_bridge,
        "mojo_logit_loss",
        lambda *args: pytest.fail("one batch selected Mojo in a missing-data fit"),
    )
    X, Y = data(a=2, b=3, c=20)
    Y[0, 0] = np.nan
    model = model_base(inp=2, out=3, df=2)
    model.fit(X, Y, epochs=1, batch_size=10, verbose=False)
    model.logLik(X, Y, batch_size=10)


def test_forced_missing_data_fails_before_dataloader(
    monkeypatch, data, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "1")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    X, Y = data(a=2, b=3, c=20)
    Y[0, 0] = np.nan
    model = model_base(inp=2, out=3, df=2)
    monkeypatch.setattr(
        model,
        "_get_DataLoader",
        lambda *args, **kwargs: pytest.fail("DataLoader constructed before guard"),
    )
    with pytest.raises(RuntimeError, match="missing responses"):
        model.fit(X, Y, epochs=1, batch_size=10, verbose=False)
```

Run these two tests before implementation. Expected: auto mode reaches the fake Mojo loss on a complete minibatch, and forced mode reaches DataLoader construction before reporting missing responses.

- [ ] **Step 5: Implement one selection point for `fit()` and `logLik()`**

Add this method immediately before `_build_loss_function`:

```python
    def _loss_function_for_data(
        self,
        Y: np.ndarray,
        train: bool = True,
        individual: bool = False,
    ) -> Callable:
        import os

        has_missing = bool(np.isnan(Y).any())
        mode = os.environ.get("SJSDM_MOJO_BACKEND", "auto").strip().lower()
        binary_training = train and self.link in ("logit", "probit")
        if binary_training and has_missing and mode == "1":
            raise RuntimeError(
                "SJSDM_MOJO_BACKEND=1 does not support missing responses."
            )
        return self._build_loss_function(
            train=train,
            individual=individual,
            allow_mojo=not has_missing,
        )
```

At the start of `fit()`, before `_get_DataLoader`, assign:

```python
        self._loss_function = self._loss_function_for_data(Y, train=True)
```

In `logLik()`, before `_get_DataLoader`, replace the direct `_build_loss_function` call with:

```python
        loss_function = self._loss_function_for_data(
            Y, train=train, individual=individual
        )
```

In `_build_loss_function`, change the auto predicate to:

```python
                    use_mojo = (
                        mode == "1"
                        or (
                            self.device.type == "cpu"
                            and self.dtype == torch.float32
                            and mojo_available()
                        )
                    )
```

When `use_mojo` is true, return `mojo_tmp` directly in both forced and auto modes. Delete the minibatch-level `torch.isnan(Ys).any()` wrapper; the bridge retains its direct-call NaN guard.

- [ ] **Step 6: Run all new selection tests and the existing sjSDM tests**

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider sjSDM/inst/python/tests/test_sjSDM.py -q
```

Expected: exit 0 with only the file's documented xfails; the test fixture restores the dtype seen before each new test.

- [ ] **Step 7: Commit model selection**

```bash
git add sjSDM/inst/python/sjSDM_py/model_sjSDM.py \
  sjSDM/inst/python/tests/test_sjSDM.py
git commit -m "fix: select Mojo for eligible whole fits"
```

---

### Task 6: Add reproducible release-safety validation

**Files:**
- Create: `Code/dev/benchmarks/validate_mojo_release.py`
- Modify: `Code/README.md:20-45`

**Interfaces:**
- Consumes: `--baseline-server PATH` and `--candidate-server PATH`.
- Produces: process exit 0 only when cross-binary bytes, large explicit-noise numerical tolerances, post-warm-up RSS growth, and absolute RSS all pass.

- [ ] **Step 1: Create the validation script**

Create `Code/dev/benchmarks/validate_mojo_release.py` with this structure and exact gates:

```python
#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import psutil
import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sjSDM" / "inst" / "python"))

from sjSDM_py import mojo_bridge


def make_case(sites, species, rank, samples, seed=2026):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn((sites, species), generator=generator)
    sigma = 0.5 * torch.randn((species, rank), generator=generator)
    y = torch.randint(0, 2, (sites, species), generator=generator).float()
    noise = torch.randn((samples, sites, rank), generator=generator)
    return mu, sigma, y, noise


def worker(path):
    os.environ["SJSDM_MOJO_SERVER_BIN"] = str(Path(path).resolve())
    return mojo_bridge._PersistentWorker()


def explicit_result(path, shape, seed):
    mu, sigma, y, noise = make_case(*shape, seed=seed)
    proc = worker(path)
    try:
        return proc.run(mu, sigma, y, noise, 1.0)
    finally:
        proc.close()


def check_cross_binary_bytes(baseline, candidate):
    for index, shape in enumerate(((8, 7, 3, 11), (256, 20, 5, 25))):
        before = explicit_result(baseline, shape, 300 + index)
        after = explicit_result(candidate, shape, 300 + index)
        assert all(
            left.tobytes() == right.tobytes()
            for left, right in zip(before, after)
        ), f"response bytes changed at shape {shape}"


def check_numerical(candidate):
    sites, species, rank, samples = 400, 10, 4, 400
    mu, sigma, y, noise = make_case(sites, species, rank, samples, seed=401)
    mu_ref = mu.clone().requires_grad_(True)
    sigma_ref = sigma.clone().requires_grad_(True)
    e = torch.sigmoid(
        torch.einsum("ijk,lk->ijl", noise, sigma_ref).add(mu_ref)
    ).mul(0.999999).add(0.0000005)
    ll = (y * e.log() + (1.0 - y) * (1.0 - e).log()).sum(dim=2)
    maximum = ll.max(dim=0).values
    reference = -(ll.sub(maximum).exp().mean(dim=0).log() + maximum)
    reference.sum().backward()

    proc = worker(candidate)
    try:
        loss, gmu, gsigma = proc.run(mu, sigma, y, noise, 1.0)
    finally:
        proc.close()
    errors = {
        "loss": float(np.max(np.abs(loss - reference.detach().numpy()))),
        "gmu": float(np.max(np.abs(gmu.reshape(mu.shape) - mu_ref.grad.numpy()))),
        "gsigma": float(np.max(np.abs(
            gsigma.reshape(sigma.shape) - sigma_ref.grad.numpy()
        ))),
    }
    assert errors["loss"] <= 1e-4, errors
    assert errors["gmu"] <= 1e-4, errors
    assert errors["gsigma"] <= 5e-4, errors
    return errors


def send_seed(proc, shape, seed=503):
    sites, species, rank, samples = shape
    mu = torch.zeros((sites, species), dtype=torch.float32)
    sigma = torch.zeros((species, rank), dtype=torch.float32)
    y = torch.zeros((sites, species), dtype=torch.float32)
    proc.run_seed(mu, sigma, y, samples, 1.0, seed)


def rss_mib(proc):
    return psutil.Process(proc.proc.pid).memory_info().rss / 1024 / 1024


def check_memory(candidate):
    alternating = ((200, 100, 5, 100), (199, 100, 5, 100))
    proc = worker(candidate)
    try:
        for shape in alternating:
            send_seed(proc, shape)
        start = rss_mib(proc)
        for index in range(20):
            send_seed(proc, alternating[index % 2], 600 + index)
        growth = rss_mib(proc) - start
    finally:
        proc.close()
    assert growth < 4.0, f"post-warm-up RSS grew {growth:.2f} MiB"

    proc = worker(candidate)
    try:
        heavy = (4096, 200, 8, 25)
        for index in range(5):
            send_seed(proc, heavy, 700 + index)
        absolute = rss_mib(proc)
    finally:
        proc.close()
    assert absolute < 40.0, f"heavy-shape RSS was {absolute:.2f} MiB"
    return {"growth_mib": growth, "heavy_mib": absolute}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-server", required=True)
    parser.add_argument("--candidate-server", required=True)
    args = parser.parse_args()
    check_cross_binary_bytes(args.baseline_server, args.candidate_server)
    numerical = check_numerical(args.candidate_server)
    memory = check_memory(args.candidate_server)
    print({"byte_parity": "pass", "numerical": numerical, "memory": memory})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the complete release-safety script**

```bash
work/mojo-backend/.pixi/envs/default/bin/python \
  Code/dev/benchmarks/validate_mojo_release.py \
  --baseline-server /private/tmp/sjsdm-mojo-v0.2.0-server \
  --candidate-server work/mojo-backend/mc_grad_server_bin
```

Expected: exit 0, `byte_parity: pass`, errors within the three specified tolerances, RSS growth below 4 MiB, and heavy RSS below 40 MiB.

- [ ] **Step 3: Document the validation command**

In `Code/README.md`, add a development bullet for `validate_mojo_release.py` and the exact command from Step 2. State that the absolute RSS gate is calibrated for the target Apple Silicon machine and is not a portable unit test.

- [ ] **Step 4: Commit the release gate**

```bash
git add Code/dev/benchmarks/validate_mojo_release.py Code/README.md
git commit -m "test: add Mojo release safety gates"
```

---

### Task 7: Update safety documentation and rerun performance baselines

**Files:**
- Modify: `Code/sjSDM_mojo_tutorial.Rmd:131-185,490-524`
- Modify: `README.Rmd:108-225`
- Regenerate: `README.md`
- Modify: `CHANGELOG.md:1-12`
- Modify: `AGENTS.md:160-188`

**Interfaces:**
- Consumes: passing Tasks 1-6 and benchmark output from the corrected `-O 3` build.
- Produces: documentation that distinguishes tagged v0.2.0 behavior from the corrected development branch and records high-water memory semantics.

- [ ] **Step 1: Run the corrected performance baselines from clean workers**

Run the shape grid:

```bash
work/mojo-backend/.pixi/envs/default/bin/python \
  Code/dev/benchmarks/bench_grid_mojo.py
```

Run the two end-to-end R workloads in a fresh process:

```bash
RETICULATE_PYTHON="$(pwd)/work/mojo-backend/reticulate-venv/bin/python" \
  Rscript Code/dev/bench_realdata.R
```

Expected: both commands exit 0. Preserve their console output for review, but do not commit generated CSVs. If a benchmark fails, stop this task and repair the regression before editing claims.

- [ ] **Step 2: Replace tutorial workarounds with release-aware wording**

In `Code/sjSDM_mojo_tutorial.Rmd`, replace the auto-mode and 262,144-limit paragraphs with:

```markdown
Auto mode selects Mojo only for CPU float32 logit/probit operations when the worker exists and the complete response matrix has no missing values. CPU float64 and operations with missing responses use PyTorch for the whole operation. Forced Mojo mode remains strict and reports unsupported inputs.

> **Release note:** The tagged v0.2.0 worker generates at most 262,144 noise values safely and uses a collision-prone packed shape key. The current development branch fixes both defects. If you are using the v0.2.0 tag, require `sampling * step_size * biotic_rank <= 262144` or select PyTorch/explicit-noise transport; if you build from the current branch, run the release-safety validation in `Code/README.md` before relying on larger requests.
```

Replace the stale shape-grid warning with a development-build label and summarize only the freshly observed direction/range from Step 1. Keep the historical v0.2.0 figures explicitly labelled rather than overwriting them.

Add this memory note near the backend limitations:

```markdown
The persistent worker retains each buffer's high-water capacity so repeated fits do not reallocate. A single unusually large fit can therefore keep that memory until the worker or R session exits. Restart the R session to release it; Python developers can call `sjSDM_py.mojo_bridge._WORKER.close()`.
```

Update the troubleshooting rows so large-request and float64-auto failures are identified as v0.2.0-tag behavior, not current-branch behavior.

- [ ] **Step 3: Add the same release distinction to the README source**

After the backend-switching examples in `README.Rmd`, add:

```markdown
> **v0.2.0 safety note:** The v0.2.0 tag has a 262,144-value seed-generation ceiling and collision-prone mixed-shape buffer reuse. Both are fixed on the current development branch. Tagged-release users should keep `sampling * step_size * biotic_rank <= 262144`, use PyTorch, or disable seed transport until a corrected release is available.
```

Also state that auto mode falls back for the entire operation when dtype is float64 or any response is missing. Regenerate `README.md` rather than editing generated prose independently:

```bash
Rscript -e "rmarkdown::render('README.Rmd', output_file='README.md', quiet=TRUE)"
```

Expected: render exit 0 and `README.md` contains the same safety note.

- [ ] **Step 4: Record the unreleased fixes and update maintainer guidance**

Add this section above 0.2.0 in `CHANGELOG.md`:

```markdown
## [Unreleased]

- Fill every server-side noise value for requests above 262,144 elements.
- Replace packed shape fingerprints with independently capacity-managed buffers and free replaced/final allocations.
- Reuse one z/ll scratch slot per concurrent chunk, reducing heavy-shape worker RSS by roughly 76% on the validation machine without changing response bytes.
- Reject malformed protocol requests before transport/allocation, restore one-shot autograd, and make auto dtype/missing-data fallback operation-wide.
```

In `AGENTS.md`:

- replace “future per-shape caching must key on the full shape” with “persistent buffers are managed by independent element capacities; never reintroduce a lossy composite key”;
- label the 0.18/1.9 GB observations as v0.2.0/pre-fix measurements;
- record the new 4 MiB growth and 40 MiB heavy-shape release gates;
- state that runtime `parallelize` counts are supported by the pinned Mojo 1.0.0 compiler;
- correct the Python test command to use `../../../work/...` from `sjSDM/inst/python`, or use the repository-root command from Task 8.

- [ ] **Step 5: Render and inspect the tutorial**

```bash
mkdir -p /private/tmp/sjsdm-mojo-tutorial
Rscript -e "rmarkdown::render(
  'Code/sjSDM_mojo_tutorial.Rmd',
  output_dir='/private/tmp/sjsdm-mojo-tutorial',
  quiet=TRUE
)"
test -s /private/tmp/sjsdm-mojo-tutorial/sjSDM_mojo_tutorial.html
```

Expected: render exit 0 and a non-empty HTML file. Inspect headings, code-block wrapping, tables, safety notes, and links before committing.

- [ ] **Step 6: Commit documentation separately from code**

```bash
git add AGENTS.md CHANGELOG.md README.Rmd README.md \
  Code/sjSDM_mojo_tutorial.Rmd
git commit -m "docs: describe corrected Mojo safety behavior"
```

---

### Task 8: Run the complete release gate and request review

**Files:**
- Verify only; modify earlier task files only if a failing gate exposes a defect.

**Interfaces:**
- Consumes: all prior task commits and `/private/tmp/sjsdm-mojo-v0.2.0-server`.
- Produces: one clean, reviewed branch with source, tests, validation tooling, benchmarks, and documentation aligned.

- [ ] **Step 1: Rebuild the final candidate from source**

```bash
cd work/mojo-backend
pixi run mojo build -O 3 mojo/mc_logit_grad_server.mojo -o mc_grad_server_bin
cd ../..
```

Expected: exit 0. Deprecation warnings already present in Mojo 1.0.0 are acceptable; compiler errors are not.

- [ ] **Step 2: Run the complete Python suite**

```bash
work/mojo-backend/.pixi/envs/default/bin/python -m pytest \
  -p no:cacheprovider sjSDM/inst/python/tests -q
```

Expected: exit 0, zero failures/errors, and only explicitly documented xfails/xpasses.

- [ ] **Step 3: Run kernel parity and release-safety validation**

```bash
work/mojo-backend/.pixi/envs/default/bin/python \
  work/mojo-backend/mojo/parity_check.py grid

work/mojo-backend/.pixi/envs/default/bin/python \
  work/mojo-backend/mojo/parity_check.py grad

work/mojo-backend/.pixi/envs/default/bin/python \
  Code/dev/benchmarks/validate_mojo_release.py \
  --baseline-server /private/tmp/sjsdm-mojo-v0.2.0-server \
  --candidate-server work/mojo-backend/mc_grad_server_bin
```

Expected: all three commands exit 0; the parity harness reports all grid cases and finite-difference gradients passing, and the release script reports byte parity, numerical errors within `1e-4/1e-4/5e-4`, RSS growth below 4 MiB, and heavy RSS below 40 MiB.

- [ ] **Step 4: Run the source-aware R suite**

```bash
RETICULATE_PYTHON="$(pwd)/work/mojo-backend/reticulate-venv/bin/python" \
Rscript -e "devtools::load_all('sjSDM', quiet=TRUE); \
  testthat::test_dir('sjSDM/tests/testthat', reporter='summary')"
```

Expected: R exits 0 with zero failed expectations. Record any existing warning from the bare nested pytest invocation for the separate installation/validation-reproducibility subproject rather than expanding this safety implementation.

- [ ] **Step 5: Run repository hygiene checks**

```bash
git diff --check
git status --short
git log --oneline -10
```

Expected: `git diff --check` exits 0; the worktree is clean; no binary, HTML, CSV, lockfile, cache, or temporary file is tracked.

- [ ] **Step 6: Request final code review**

Invoke `superpowers:requesting-code-review` against the complete branch. Require the reviewer to check:

- all eleven design invariants;
- validation-before-write and validation-before-allocation ordering;
- every pointer has exactly one live allocation and is freed once on replacement/final exit;
- explicit-noise numerical tolerances and cross-binary byte equality;
- auto/forced dtype and missing-data semantics;
- documentation distinguishes v0.2.0 from the corrected development build.

Address any confirmed finding with a focused regression test and a separate commit, then rerun Steps 1-5.

"""Phase 1/2 integration gate: end-to-end training with the Mojo backend.

Runs identical training runs (same seed, same data) through the PyTorch
loss and the opt-in Mojo backend (SJSDM_MOJO_BACKEND=1), compares full
training trajectories, and reports end-to-end wall-clock times on a
larger workload.

Usage (from repository root):
  work/mojo-backend/.pixi/envs/default/bin/python \
    Code/benchmarks/train_parity_mojo.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "sjSDM/inst/python")
from sjSDM_py.model_sjSDM import Model_sjSDM


def simulate(sites, species, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(sites, p)).astype(np.float32)
    W = rng.normal(scale=0.5, size=(p, species)).astype(np.float32)
    C = rng.normal(scale=0.4, size=(species, 2)).astype(np.float32)
    eta = X @ W + rng.normal(size=(sites, 1)).astype(np.float32) @ np.ones((1, species))
    noise = rng.normal(size=(sites, 2)).astype(np.float32) @ C.T
    Y = (rng.uniform(size=(sites, species)) < 1.0 / (1.0 + np.exp(-(eta + noise)))).astype(np.float32)
    return X, Y


def make_model(seed, p, species):
    torch.manual_seed(seed)
    m = Model_sjSDM(device="cpu", dtype="float32", seed=seed)
    m.add_env(input_shape=p, output_shape=species,
              hidden=[10], activation=["relu"], bias=[True])
    m.build(df=3, link="logit",
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-2))
    return m


def fit_once(mojo, X, Y, p, species, seed=123, epochs=30,
             batch_size=50, sampling=25):
    os.environ["SJSDM_MOJO_BACKEND"] = "1" if mojo else "0"
    m = make_model(seed, p, species)
    t0 = time.perf_counter()
    m.fit(X, Y, batch_size=batch_size, epochs=epochs,
          sampling=sampling, verbose=False)
    return m.history.copy(), time.perf_counter() - t0


def main():
    # --- Trajectory parity: small workload -------------------------------
    p, species = 5, 12
    X, Y = simulate(300, species, p, seed=55)
    assert not np.isnan(Y).any()

    hist_torch, _ = fit_once(False, X, Y, p, species)
    hist_mojo, _ = fit_once(True, X, Y, p, species)

    diff = np.abs(hist_torch - hist_mojo)
    denom = np.maximum(np.abs(hist_torch), 1e-6)
    print("epoch | torch     | mojo      | abs diff")
    for i in list(range(5)) + [len(hist_torch) - 1]:
        print(f"{i+1:5d} | {hist_torch[i]:9.4f} | {hist_mojo[i]:9.4f} | {diff[i]:.2e}")
    rel = (diff / denom).max()
    ok = rel < 5e-3
    print(f"trajectory max rel diff = {rel:.2e} -> {'PASS' if ok else 'FAIL'}")

    # --- End-to-end timing: larger workload -------------------------------
    p2, sp2 = 8, 60
    X2, Y2 = simulate(3000, sp2, p2, seed=77)
    _, t_torch = fit_once(False, X2, Y2, p2, sp2, epochs=5, batch_size=100, sampling=50)
    _, t_mojo = fit_once(True, X2, Y2, p2, sp2, epochs=5, batch_size=100, sampling=50)
    print(f"end-to-end 3000x{sp2}, 5 epochs: torch={t_torch:.1f}s mojo={t_mojo:.1f}s "
          f"(ratio {t_torch / t_mojo:.2f}x)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 0 baseline: time sjSDM fitting on CPU vs MPS.

Synthetic binary community data; linear env layer (the neural net is
trivial so timing is dominated by the MC likelihood). Grid over sites,
species, and Monte Carlo samples to find where MPS actually wins.

The first MPS run pays kernel-compile overhead, so each device does a
small warm-up fit before timing.
"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "sjSDM/inst/python")
from sjSDM_py.model_sjSDM import Model_sjSDM  # noqa: E402


def make_data(n=5000, s=50, p=10, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta = rng.normal(scale=0.5, size=(p, s)).astype(np.float32)
    logits = X @ beta
    Y = (rng.uniform(size=(n, s)) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    return X, Y


def run(device, n, s, sampling, epochs=3):
    X, Y = make_data(n=n, s=s)
    m = Model_sjSDM(device=device, seed=42)
    m.add_env(input_shape=X.shape[1], output_shape=Y.shape[1])
    m.build(df=5, link="logit", alpha=1.0, scheduler=False,
            optimizer=torch.optim.Adam)
    t0 = time.perf_counter()
    m.fit(X, Y, batch_size=100, epochs=epochs, sampling=sampling, verbose=False)
    dt = time.perf_counter() - t0
    print(f"{device:>4} n={n:<6d} s={s:<4d} sampling={sampling:<4d}: "
          f"{dt:7.2f}s (final loss {m.history[-1]:.4f})", flush=True)
    return dt


if __name__ == "__main__":
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    grid = [
        # (n, species, sampling)
        (5000, 50, 100),
        (20000, 50, 100),
        (20000, 200, 100),
        (20000, 200, 400),
        (50000, 500, 100),
    ]
    for device in devices:
        run(device, 2000, 20, 5)  # warm-up (esp. MPS kernel compilation)
        for n, s, sampling in grid:
            run(device, n, s, sampling)

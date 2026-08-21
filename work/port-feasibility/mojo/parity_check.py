"""Phase 1 parity + timing check: Mojo kernel vs PyTorch reference.

Generates inputs with externally supplied noise (no RNG in the kernel),
writes them as raw float32 binaries, runs both implementations, and
compares per-site losses.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "sjSDM/inst/python")
MOJO = ("pixi", "run", "mojo", "run")
KERNEL = "mojo/mc_logit_loss.mojo"


def torch_reference(mu, sigma, y, noise, alpha=1.0):
    z = torch.einsum("ijk,lk->ijl", noise, sigma).add(mu).mul(alpha)
    e = torch.sigmoid(z).mul(0.999999).add(0.0000005)
    ll = (e.log().mul(y) + (1.0 - e).log().mul(1.0 - y)).sum(dim=2)
    max_ll = ll.max(dim=0).values
    loss = -(ll.sub(max_ll).exp().mean(dim=0).log() + max_ll)
    return loss.numpy()


def run_case(sites, species, rank, samples, seed=7):
    rng = np.random.default_rng(seed)
    mu = rng.normal(scale=0.5, size=(sites, species)).astype(np.float32)
    sigma = rng.normal(scale=0.5, size=(species, rank)).astype(np.float32)
    y = rng.integers(0, 2, size=(sites, species)).astype(np.float32)
    noise = rng.normal(size=(samples, sites, rank)).astype(np.float32)

    tmp = Path("work/port-feasibility/mojo/tmp").resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    for name, arr in [("mu", mu), ("sigma", sigma), ("y", y), ("noise", noise)]:
        arr.tofile(tmp / f"{name}.bin")

    ref = torch_reference(
        torch.tensor(mu), torch.tensor(sigma), torch.tensor(y), torch.tensor(noise)
    )

    subprocess.run(
        [*MOJO, KERNEL,
         str(sites), str(species), str(rank), str(samples),
         str(tmp / "mu.bin"), str(tmp / "sigma.bin"), str(tmp / "y.bin"),
         str(tmp / "noise.bin"), str(tmp / "out.bin")],
        check=True, cwd="work/port-feasibility",
    )
    mojo_out = np.fromfile(tmp / "out.bin", dtype=np.float32)

    abs_err = np.abs(ref - mojo_out)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-6)
    print(f"sites={sites:<6d} s={species:<4d} d={rank} K={samples:<5d} "
          f"max|err|={abs_err.max():.3e}  med|rel_err|={np.median(rel_err):.3e}")
    return abs_err.max()


if __name__ == "__main__":
    run_case(64, 10, 5, 8)      # tiny smoke
    run_case(5000, 50, 5, 100)  # matches baseline workload shape

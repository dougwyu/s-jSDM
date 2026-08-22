"""Phase 1/2 parity + validation: Mojo kernels vs PyTorch reference.

Forward: compares per-site losses with externally supplied noise (no RNG
in the kernel).

Grid mode (arg "grid"): runs the forward parity across a randomized grid
of shapes to harden confidence.

Grad mode ("grad"): validates dL/dmu and dL/dsigma against central finite
differences of the reference loss on random coordinate subsets.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "sjSDM/inst/python")

MOJO = ("pixi", "run", "mojo", "run")
KERNEL = "mojo/mc_logit_loss.mojo"
GRAD_KERNEL = "mojo/mc_logit_grad.mojo"
TMP = Path("work/mojo-backend/mojo/tmp").resolve()


def make_inputs(sites, species, rank, samples, seed=7):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(scale=0.5, size=(sites, species)).astype(np.float32),
        rng.normal(scale=0.5, size=(species, rank)).astype(np.float32),
        rng.integers(0, 2, size=(sites, species)).astype(np.float32),
        rng.normal(size=(samples, sites, rank)).astype(np.float32),
    )


def torch_reference(mu, sigma, y, noise, alpha=1.0):
    z = torch.einsum("ijk,lk->ijl", noise, sigma).add(mu).mul(alpha)
    e = torch.sigmoid(z).mul(0.999999).add(0.0000005)
    ll = (e.log().mul(y) + (1.0 - e).log().mul(1.0 - y)).sum(dim=2)
    max_ll = ll.max(dim=0).values
    return -(ll.sub(max_ll).exp().mean(dim=0).log() + max_ll).numpy()


def run_mojo(kernel, mu, sigma, y, noise, sites, species, rank, samples, out_names):
    TMP.mkdir(parents=True, exist_ok=True)
    for name, arr in [("mu", mu), ("sigma", sigma), ("y", y), ("noise", noise)]:
        arr.tofile(TMP / f"{name}.bin")
    cmd = [*MOJO, kernel, str(sites), str(species), str(rank), str(samples),
           str(TMP / "mu.bin"), str(TMP / "sigma.bin"), str(TMP / "y.bin"),
           str(TMP / "noise.bin")]
    cmd += [str(TMP / f"{n}.bin") for n in out_names]
    subprocess.run(cmd, check=True, cwd="work/mojo-backend",
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {n: np.fromfile(TMP / f"{n}.bin", dtype=np.float32) for n in out_names}


def forward_parity(sites, species, rank, samples, seed=7):
    mu, sigma, y, noise = make_inputs(sites, species, rank, samples, seed)
    ref = torch_reference(torch.tensor(mu), torch.tensor(sigma),
                          torch.tensor(y), torch.tensor(noise))
    out = run_mojo(KERNEL, mu, sigma, y, noise, sites, species, rank, samples,
                   ["out"])["out"]
    abs_err = np.abs(ref - out)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-6)
    ok = abs_err.max() < 5e-4 * max(1.0, float(np.abs(ref).max()))
    print(f"{'PASS' if ok else 'FAIL'} sites={sites:<6d} s={species:<4d} "
          f"d={rank} K={samples:<5d} max|err|={abs_err.max():.3e} "
          f"med|rel|={np.median(rel_err):.1e}", flush=True)
    return ok


def run_grid(n_cases=12, seed=99):
    rng = np.random.default_rng(seed)
    ok = True
    for _ in range(n_cases):
        sites = int(rng.choice([7, 64, 333, 1000, 5000]))
        species = int(rng.choice([2, 5, 17, 50, 120]))
        rank = int(rng.integers(1, min(species, 8) + 1))
        samples = int(rng.choice([3, 8, 25, 100, 250]))
        ok &= forward_parity(sites, species, rank, samples,
                             seed=int(rng.integers(1, 10_000)))
    print("GRID:", "all passed" if ok else "FAILURES PRESENT", flush=True)
    return ok


def total_loss(mu, sigma, y, noise):
    return float(torch_reference(torch.tensor(mu), torch.tensor(sigma),
                                 torch.tensor(y), torch.tensor(noise)).sum())


def grad_validation(seed=11, n_mu=8, n_sigma=6, h=1e-2, tol=2e-2):
    sites, species, rank, samples = 40, 12, 4, 30
    mu, sigma, y, noise = make_inputs(sites, species, rank, samples, seed)

    outs = run_mojo(GRAD_KERNEL, mu, sigma, y, noise,
                    sites, species, rank, samples, ["out", "gmu", "gsigma"])
    mojo_loss = outs["out"].sum()
    gmu = outs["gmu"].reshape(sites, species)
    gsigma = outs["gsigma"].reshape(species, rank)

    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(n_mu):
        i, j = int(rng.integers(sites)), int(rng.integers(species))
        mp = mu.copy(); mp[i, j] += h
        mm = mu.copy(); mm[i, j] -= h
        fd = (total_loss(mp, sigma, y, noise) - total_loss(mm, sigma, y, noise)) / (2 * h)
        denom = max(abs(fd), abs(gmu[i, j]), 1e-3)
        rel = abs(fd - gmu[i, j]) / denom
        worst = max(worst, rel)
        print(f"gmu[{i:2d},{j:2d}]   mojo={gmu[i, j]:9.4f}  fd={fd:9.4f}  rel={rel:.2e}")
    for _ in range(n_sigma):
        j, d = int(rng.integers(species)), int(rng.integers(rank))
        sp = sigma.copy(); sp[j, d] += h
        sm = sigma.copy(); sm[j, d] -= h
        fd = (total_loss(mu, sp, y, noise) - total_loss(mu, sm, y, noise)) / (2 * h)
        denom = max(abs(fd), abs(gsigma[j, d]), 1e-3)
        rel = abs(fd - gsigma[j, d]) / denom
        worst = max(worst, rel)
        print(f"gsigma[{j:2d},{d}] mojo={gsigma[j, d]:9.4f}  fd={fd:9.4f}  rel={rel:.2e}")

    ref_loss = total_loss(mu, sigma, y, noise)
    loss_rel = abs(mojo_loss - ref_loss) / max(abs(ref_loss), 1e-6)
    ok = worst < tol and loss_rel < 1e-4
    print(f"worst grad rel err = {worst:.3e} (tol {tol}), "
          f"loss rel err = {loss_rel:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "forward"
    if mode == "grid":
        sys.exit(0 if run_grid() else 1)
    elif mode == "grad":
        sys.exit(0 if grad_validation() else 1)
    else:
        forward_parity(64, 10, 5, 8)
        forward_parity(5000, 50, 5, 100)

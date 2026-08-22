"""Grid benchmark: Mojo persistent-bridge MC loss vs PyTorch autograd.

For each (species, rank, samples) combination, times per-batch
forward+backward on a fixed batch of sites, using identical tensor shapes.
Reports the ratio torch/mojo (>1 means Mojo faster).

Usage (from repository root):
  work/mojo-backend/.pixi/envs/default/bin/python \
    Code/benchmarks/bench_grid_mojo.py
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "sjSDM/inst/python")
from sjSDM_py.mojo_bridge import mojo_logit_loss

SITES = 500
SPECIES = [20, 60, 200]
RANK = [2, 5, 8]
SAMPLES = [25, 100, 400]
REPS = 5


def bench(fn, warmup=2):
    for _ in range(warmup):
        fn().mean().backward()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn().mean().backward()
    return (time.perf_counter() - t0) / REPS * 1000


def main():
    rng = np.random.default_rng(21)
    print(f"sites={SITES}, {REPS} reps per cell (fwd+bwd ms)")
    print(f"{'species':>7} {'rank':>4} {'K':>4} {'torch':>9} {'mojo':>9} {'ratio':>6}")
    rows = []
    for sp in SPECIES:
        rank_list = [r for r in RANK if r < sp]
        Ys = torch.tensor(rng.integers(0, 2, size=(SITES, sp)).astype(np.float32))
        mu = torch.tensor(rng.normal(scale=.5, size=(SITES, sp)).astype(np.float32),
                          requires_grad=True)
        sigs = {}
        for rank in rank_list:
            sigs[rank] = torch.tensor(
                rng.normal(scale=.5, size=(sp, rank)).astype(np.float32),
                requires_grad=True)
        for rank in rank_list:
            sigma = sigs[rank]
            for K in SAMPLES:
                def torch_loss():
                    noise = torch.randn(size=(K, SITES, rank))
                    E = torch.sigmoid(
                        torch.einsum("ijk,lk->ijl", [noise, sigma]).add(mu)
                    ).mul(0.999999).add(0.0000005)
                    lp = E.log().mul(Ys).add((1 - E).log().mul(1 - Ys)).neg().sum(2).neg()
                    m = lp.max(0).values
                    return lp.sub(m).exp().mean(0).log().neg().sub(m)

                def mojo_loss():
                    return mojo_logit_loss(mu, Ys, sigma, K, 1.0)

                t_torch = bench(torch_loss)
                t_mojo = bench(mojo_loss)
                ratio = t_torch / t_mojo
                rows.append((sp, rank, K, ratio))
                print(f"{sp:>7} {rank:>4} {K:>4} {t_torch:>7.0f}ms {t_mojo:>7.0f}ms "
                      f"{ratio:>6.2f}", flush=True)
    wins = sum(1 for *_, r in rows if r > 1)
    best = max(rows, key=lambda r: r[3])
    worst = min(rows, key=lambda r: r[3])
    print(f"\nMojo faster in {wins}/{len(rows)} cells; "
          f"best ratio {best[3]:.2f} (sp={best[0]}, d={best[1]}, K={best[2]}); "
          f"worst {worst[3]:.2f} (sp={worst[0]}, d={worst[1]}, K={worst[2]})")


if __name__ == "__main__":
    main()

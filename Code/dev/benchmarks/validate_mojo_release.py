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

"""Tests for the opt-in Mojo MC-likelihood bridge.

Covers the persistent pipe protocol, mixed-shape requests within one
worker process, probit alpha scaling, autograd integration, and the
documented guard conditions. Skipped when the prebuilt worker binaries
are absent (they are not part of the repo; see AGENTS.md for the build
command).
"""

import os
import subprocess

import numpy as np
import pytest
import torch

from ..sjSDM_py import mojo_bridge

SERVER_BIN = mojo_bridge._server_path()
ONESHOT_BIN = mojo_bridge._oneshot_path()

pytestmark = pytest.mark.skipif(
    not os.path.exists(SERVER_BIN),
    reason="Mojo worker binary not built (work/port-feasibility/mc_grad_server_bin)",
)


@pytest.fixture(autouse=True)
def _clean_worker():
    yield
    mojo_bridge._WORKER.close()


def reference_logit_loss(mu, sigma, y, noise, alpha):
    """PyTorch reference for the binary logit MC loss with fixed noise.

    Returns the positive per-site MC negative log-likelihood, matching
    the kernel's logsumexp-style reduction.
    """
    E = torch.sigmoid(torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).mul(alpha))
    E = E.mul(0.999999).add(0.0000005)
    ll = E.log().mul(y).add((1.0 - E).log().mul(1.0 - y))
    nll = ll.neg().sum(dim=2)
    m = nll.max(dim=0).values
    return m.sub(m.unsqueeze(0).sub(nll).exp().mean(dim=0).log())


def make_case(sites, species, rank, samples, seed=0, alpha=1.0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(sites, species, generator=g)
    sigma = 0.5 * torch.randn(species, rank, generator=g)
    y = torch.bernoulli(torch.sigmoid(mu), generator=g)
    noise = torch.randn(samples, sites, rank, generator=g)
    return mu, sigma, y, noise


def run_worker(mu, sigma, y, noise, alpha):
    loss, gmu, gsigma = mojo_bridge._WORKER.run(mu, sigma, y, noise, alpha)
    sites, species = mu.shape
    rank = sigma.shape[1]
    return loss, gmu.reshape(sites, species), gsigma.reshape(species, rank)


class TestPersistentProtocol:
    def test_shapes_and_finiteness(self):
        mu, sigma, y, noise = make_case(50, 20, 3, 40)
        loss, gmu, gsigma = run_worker(mu, sigma, y, noise, 1.0)
        assert loss.shape == (50,)
        assert gmu.shape == (50, 20)
        assert gsigma.shape == (20, 3)
        for arr in (loss, gmu, gsigma):
            assert np.isfinite(arr).all()

    def test_forward_parity_with_reference(self):
        mu, sigma, y, noise = make_case(60, 25, 4, 50, seed=7)
        loss, _, _ = run_worker(mu, sigma, y, noise, 1.0)
        ref = reference_logit_loss(mu, sigma, y, noise, 1.0).numpy()
        assert np.abs(loss - ref).max() < 1e-4

    def test_probit_alpha_parity(self):
        mu, sigma, y, noise = make_case(60, 25, 4, 50, seed=11, alpha=1.7012)
        alpha = 1.7012
        loss, _, _ = run_worker(mu, sigma, y, noise, alpha)
        ref = reference_logit_loss(mu, sigma, y, noise, alpha).numpy()
        assert np.abs(loss - ref).max() < 1e-4

    def test_gradient_against_central_differences(self):
        torch.manual_seed(3)
        sites, species, rank, samples = 20, 10, 3, 30
        mu, sigma, y, noise = make_case(sites, species, rank, samples, seed=3)

        def total_loss(mu_, sigma_):
            return reference_logit_loss(mu_, sigma_, y, noise, 1.0).sum().item()

        _, gmu, gsigma = run_worker(mu, sigma, y, noise, 1.0)
        h = 1e-2
        for i in range(3):
            for j in range(2):
                mu_p, mu_m = mu.clone(), mu.clone()
                mu_p[i, j] += h
                mu_m[i, j] -= h
                fd = (total_loss(mu_p, sigma) - total_loss(mu_m, sigma)) / (2 * h)
                assert abs(gmu[i, j] - fd) < 5e-2 * max(1.0, abs(fd))
        for i in range(species):
            for j in range(rank):
                s_p, s_m = sigma.clone(), sigma.clone()
                s_p[i, j] += h
                s_m[i, j] -= h
                fd = (total_loss(mu, s_p) - total_loss(mu, s_m)) / (2 * h)
                assert abs(gsigma[i, j] - fd) < 5e-2 * max(1.0, abs(fd))


class TestMixedShapeRequests:
    """One worker process must handle shape changes safely: buffers are
    keyed on the full shape fingerprint, not single dimensions."""

    def test_shape_change_at_same_sites(self):
        # Regression: species change at constant sites once reused stale
        # undersized buffers and produced garbage gradients.
        mu1, sigma1, y1, noise1 = make_case(50, 20, 3, 40, seed=1)
        mu2, sigma2, y2, noise2 = make_case(50, 60, 3, 40, seed=2)
        run_worker(mu1, sigma1, y1, noise1, 1.0)
        loss2, gmu2, gsigma2 = run_worker(mu2, sigma2, y2, noise2, 1.0)
        ref = reference_logit_loss(mu2, sigma2, y2, noise2, 1.0).numpy()
        assert np.abs(loss2 - ref).max() < 1e-4
        assert np.isfinite(gmu2).all() and np.isfinite(gsigma2).all()

    def test_alternating_shapes(self):
        shapes = [(30, 20, 3, 40), (30, 60, 5, 25), (80, 20, 2, 40), (30, 20, 3, 40)]
        for k, (sites, species, rank, samples) in enumerate(shapes):
            mu, sigma, y, noise = make_case(sites, species, rank, samples, seed=k)
            loss, gmu, gsigma = run_worker(mu, sigma, y, noise, 1.0)
            ref = reference_logit_loss(mu, sigma, y, noise, 1.0).numpy()
            assert np.abs(loss - ref).max() < 1e-4
            assert gmu.shape == (sites, species)
            assert gsigma.shape == (species, rank)

    def test_same_shape_repeated_requests(self):
        mu, sigma, y, noise = make_case(40, 15, 3, 30, seed=5)
        l1, gmu1, _ = run_worker(mu, sigma, y, noise, 1.0)
        l2, gmu2, _ = run_worker(mu, sigma, y, noise, 1.0)
        assert np.array_equal(l1, l2)
        assert np.array_equal(gmu1, gmu2)


class TestWorkerRestart:
    def test_recovers_after_worker_death(self):
        mu, sigma, y, noise = make_case(30, 10, 3, 20, seed=9)
        run_worker(mu, sigma, y, noise, 1.0)
        mojo_bridge._WORKER.proc.kill()
        mojo_bridge._WORKER.proc.wait()
        loss, gmu, gsigma = run_worker(mu, sigma, y, noise, 1.0)
        assert np.isfinite(loss).all() and np.isfinite(gmu).all()

    def test_autograd_function_restarts_dead_worker(self):
        mu, sigma, y, noise = make_case(30, 10, 3, 20, seed=13)
        mojo_bridge._MojoLogitMCLoss.apply(mu, sigma, y, noise, 1.0)
        mojo_bridge._WORKER.proc.kill()
        mojo_bridge._WORKER.proc.wait()
        loss = mojo_bridge._MojoLogitMCLoss.apply(mu, sigma, y, noise, 1.0)
        assert torch.isfinite(loss).all()


class TestAutogradIntegration:
    def test_backward_matches_reference_gradients(self):
        mu, sigma, y, noise = make_case(40, 15, 3, 30, seed=17)
        mu1 = mu.clone().requires_grad_(True)
        sig1 = sigma.clone().requires_grad_(True)
        loss = mojo_bridge._MojoLogitMCLoss.apply(mu1, sig1, y, noise, 1.0)
        loss.mean().backward()

        mu2 = mu.clone().requires_grad_(True)
        sig2 = sigma.clone().requires_grad_(True)
        ref = reference_logit_loss(mu2, sig2, y, noise, 1.0).mean()
        ref.backward()

        assert torch.allclose(loss.mean(), ref.detach(), atol=1e-4)
        assert torch.allclose(mu1.grad, mu2.grad, atol=1e-4)
        assert torch.allclose(sig1.grad, sig2.grad, atol=1e-4)

    def test_backward_rejects_nonuniform_grad_output(self):
        mu, sigma, y, noise = make_case(30, 10, 3, 20, seed=19)
        mu1 = mu.clone().requires_grad_(True)
        loss = mojo_bridge._MojoLogitMCLoss.apply(mu1, sigma, y, noise, 1.0)
        weights = torch.linspace(0.1, 2.0, loss.shape[0])
        with pytest.raises(RuntimeError, match="uniform grad_output"):
            loss.backward(weights)


class TestGuards:
    def test_rejects_non_float32(self):
        mu, sigma, y, noise = make_case(20, 10, 3, 20, seed=23)
        with pytest.raises(RuntimeError, match="CPU float32"):
            mojo_bridge.mojo_logit_loss(mu.double(), y, sigma, 20, 1.0)

    def test_rejects_nan_responses(self):
        mu, sigma, y, noise = make_case(20, 10, 3, 20, seed=29)
        y = y.clone()
        y[0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="NaN"):
            mojo_bridge.mojo_logit_loss(mu, y, sigma, 20, 1.0)


@pytest.mark.skipif(
    not os.path.exists(ONESHOT_BIN),
    reason="Mojo one-shot binary not built (work/port-feasibility/mc_grad_bin)",
)
class TestOneshotTransport:
    def test_matches_persistent_results(self):
        mu, sigma, y, noise = make_case(40, 15, 3, 30, seed=31)
        loss_p, gmu_p, gsig_p = run_worker(mu, sigma, y, noise, 1.0)
        loss_o, gmu_o, gsig_o = mojo_bridge._run_oneshot(mu, sigma, y, noise, 1.0)
        sites, species = mu.shape
        rank = sigma.shape[1]
        loss_o = loss_o.reshape(sites)
        gmu_o = gmu_o.reshape(sites, species)
        gsig_o = gsig_o.reshape(species, rank)
        assert np.abs(loss_p - loss_o).max() < 1e-5
        assert np.abs(gmu_p - gmu_o).max() < 1e-5
        assert np.abs(gsig_p - gsig_o).max() < 1e-5

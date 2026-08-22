"""Narrow autograd bridge to the standalone Mojo Monte Carlo likelihood kernels.

Phase 1/2 integration gate: replaces the PyTorch binary logit/probit MC
loss (and its analytic gradients for mu and sigma) with calls to a Mojo
worker process, while keeping optimizers, predictor layers, and everything
else in PyTorch.

Two transports:
- persistent (default): a long-lived worker (`mc_grad_server_bin`) reads
  requests from stdin and writes results to stdout — no subprocess spawn
  or temp-file I/O per batch.
- one-shot fallback (`SJSDM_MOJO_PERSISTENT=0`): per-call invocation of
  `mc_grad_bin` via temp files (the original bridge).

Limitations (deliberate, until further validation):
- CPU float32 tensors only.
- Y must not contain NaNs (the masked-loss path stays in PyTorch).
- The backward pass assumes grad_output is constant across sites, which
  holds for the training path (`loss.mean()`) but not for weighted losses.
- External noise is supplied by torch.randn so RNG streams match the
  PyTorch path exactly under a fixed seed.

Binary locations can be overridden with SJSDM_MOJO_SERVER_BIN /
SJSDM_MOJO_BIN.
"""

import os
import struct
import subprocess
import tempfile

import numpy as np
import torch


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _server_path() -> str:
    return os.environ.get(
        "SJSDM_MOJO_SERVER_BIN",
        os.path.join(_REPO_ROOT, "work", "port-feasibility", "mc_grad_server_bin"),
    )


def _oneshot_path() -> str:
    return os.environ.get(
        "SJSDM_MOJO_BIN",
        os.path.join(_REPO_ROOT, "work", "port-feasibility", "mc_grad_bin"),
    )


class _PersistentWorker:
    """Long-lived `mc_grad_server_bin` process speaking a raw pipe protocol.

    Request: 36-byte header (Int64 sites/species/rank/samples + UInt32
    alpha bits), then mu, sigma, y, noise as contiguous float32 bytes.
    Response: loss, gmu, gsigma as contiguous float32 bytes.
    """

    def __init__(self):
        self.proc = None

    def _start(self):
        try:
            self.proc = subprocess.Popen(
                [_server_path()],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "Mojo server binary not found; build it with "
                "`pixi run mojo build mojo/mc_logit_grad_server.mojo "
                "-o mc_grad_server_bin` in work/port-feasibility/ or set "
                "SJSDM_MOJO_SERVER_BIN."
            ) from e

    def run(self, mu, sigma, y, noise, alpha):
        sites, species = mu.shape
        rank = sigma.shape[1]
        samples = noise.shape[0]
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        header = struct.pack("<qqqq", sites, species, rank, samples)
        header += struct.pack(
            "<I", struct.unpack("<I", struct.pack("<f", alpha))[0]
        )
        payload = b"".join(
            arr.detach().numpy().tobytes() for arr in (mu, sigma, y, noise)
        )
        n_out = sites * 4
        n_gmu = sites * species * 4
        n_gsig = species * rank * 4
        total = n_out + n_gmu + n_gsig

        self.proc.stdin.write(header + payload)
        self.proc.stdin.flush()

        buf = bytearray()
        while len(buf) < total:
            chunk = self.proc.stdout.read(total - len(buf))
            if not chunk:
                raise RuntimeError(
                    "Mojo worker died mid-request; retrying may restart it."
                )
            buf.extend(chunk)

        out = np.frombuffer(bytes(buf[:n_out]), dtype=np.float32)
        gmu = np.frombuffer(bytes(buf[n_out:n_out + n_gmu]), dtype=np.float32)
        gsigma = np.frombuffer(bytes(buf[n_out + n_gmu:total]), dtype=np.float32)
        return out.copy(), gmu.copy(), gsigma.copy()

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


_WORKER = _PersistentWorker()


def _run_oneshot(mu, sigma, y, noise, alpha):
    """Original transport: prebuilt binary + temp files, one process per call."""
    sites, species = mu.shape
    rank = sigma.shape[1]
    samples = noise.shape[0]

    bin_dir = getattr(_run_oneshot, "_tmpdir", None)
    if bin_dir is None:
        bin_dir = tempfile.mkdtemp(prefix="sjsdm_mojo_")
        _run_oneshot._tmpdir = bin_dir

    paths = {
        name: os.path.join(bin_dir, f"{name}.bin")
        for name in ["mu", "sigma", "y", "noise", "out", "gmu", "gsigma"]
    }
    mu.detach().numpy().tofile(paths["mu"])
    sigma.detach().numpy().tofile(paths["sigma"])
    y.detach().numpy().tofile(paths["y"])
    noise.detach().numpy().tofile(paths["noise"])

    cmd = [
        _oneshot_path(),
        str(sites), str(species), str(rank), str(samples),
        paths["mu"], paths["sigma"], paths["y"], paths["noise"],
        paths["out"], paths["gmu"], paths["gsigma"],
        repr(float(alpha)),
    ]
    subprocess.run(cmd, check=True)

    out = np.fromfile(paths["out"], dtype=np.float32)
    gmu = np.fromfile(paths["gmu"], dtype=np.float32)
    gsigma = np.fromfile(paths["gsigma"], dtype=np.float32)
    return out, gmu, gsigma


class _MojoLogitMCLoss(torch.autograd.Function):
    """Per-site negative log-likelihood with externally supplied noise.

    Forward returns loss (batch,), backward propagates the kernel's
    analytic d(sum_i L_i)/d(mu) and d(sum_i L_i)/d(sigma).
    """

    @staticmethod
    def forward(ctx, mu, sigma, y, noise, alpha):
        batch, species = mu.shape
        rank = sigma.shape[1]

        persistent = os.environ.get("SJSDM_MOJO_PERSISTENT", "1") == "1"
        runner = _WORKER.run if persistent else _run_oneshot
        try:
            out, gmu, gsigma = runner(mu, sigma, y, noise, alpha)
        except (RuntimeError, BrokenPipeError, subprocess.CalledProcessError):
            if not persistent:
                raise
            # Worker died (e.g. after an external signal): restart once.
            _WORKER.close()
            out, gmu, gsigma = _WORKER.run(mu, sigma, y, noise, alpha)

        ctx.save_for_backward(
            torch.from_numpy(gmu.reshape(batch, species)),
            torch.from_numpy(gsigma.reshape(species, rank)),
        )
        return torch.from_numpy(out)

    @staticmethod
    def backward(ctx, grad_output):
        gmu, gsigma = ctx.saved_tensors
        # Kernel returns gradients of the summed loss; valid only for
        # constant upstream weights (true for loss.mean() in fit()).
        go = grad_output.detach()
        if not torch.all(go == go.reshape(-1)[0]):
            raise RuntimeError(
                "Mojo loss backward requires uniform grad_output across sites."
            )
        scale = go.reshape(-1)[0]
        return gmu * scale, gsigma * scale, None, None, None


def mojo_logit_loss(mu, Ys, sigma, sampling, alpha):
    """Drop-in replacement for the binary logit/probit MC training loss."""
    if mu.device.type != "cpu" or mu.dtype != torch.float32:
        raise RuntimeError("Mojo backend requires CPU float32 tensors.")
    if torch.isnan(Ys).any():
        raise RuntimeError(
            "Mojo backend does not support NaN responses; "
            "unset SJSDM_MOJO_BACKEND for this data."
        )
    batch, species = mu.shape
    rank = sigma.shape[1]
    noise = torch.randn(size=(int(sampling), int(batch), int(rank)),
                        device=torch.device("cpu"), dtype=torch.float32)
    return _MojoLogitMCLoss.apply(mu, sigma, Ys.contiguous(), noise, float(alpha))

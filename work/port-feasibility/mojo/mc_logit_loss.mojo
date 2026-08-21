# Phase 1 forward kernel: binary logit Monte Carlo likelihood.
#
# Reads raw float32 binary files (little-endian, row-major):
#   argv[1] sites argv[2] species argv[3] rank argv[4] samples
#   argv[5] mu    : sites x species
#   argv[6] sigma : species x rank
#   argv[7] y     : sites x species
#   argv[8] noise : samples x sites x rank
#   argv[9] out   : sites (per-site negative log-likelihood)
#
# Computation mirrors sjSDM_py.dist_mvp.MVP_logLik for link="logit"
# (alpha = 1.0):
#   z = noise @ sigma^T + mu
#   E = sigmoid(z) clipped to [5e-7, 1 - 5e-7]
#   ll_k = sum_s y*log(E_k) + (1-y)*log(1-E_k)
#   loss_i = -log(mean_k exp(ll_k))
#
# Sites are processed in parallel on the CPU thread pool.

from max.algorithm import parallelize
from std.math import exp, log
from std.memory import alloc
from std.pathlib import Path
from std.sys import argv


def read_f32(path: String) raises -> Pointer[Float32, MutUntrackedOrigin]:
    var b = Path(path).read_bytes()
    var n = len(b) // 4
    var src = b.unsafe_ptr().unsafe_bitcast[Float32]()
    var dst = alloc[Float32](n)
    for i in range(n):
        dst[i] = src[i]
    return dst


def main() raises:
    var args = argv()
    var sites = atol(args[1])
    var species = atol(args[2])
    var rank = atol(args[3])
    var samples = atol(args[4])

    var mu = read_f32(args[5])
    var sigma = read_f32(args[6])
    var y = read_f32(args[7])
    var noise = read_f32(args[8])

    var out = alloc[Float32](sites)

    @parameter
    def site_loss(site: Int):
        # Single-pass streaming log-sum-exp over MC samples; no shared
        # scratch, so parallel sites are race-free.
        var max_ll: Float32 = -3.0e38
        var run_sum: Float32 = 0.0
        for k in range(samples):
            var acc_ll: Float32 = 0.0
            for s in range(species):
                var dot: Float32 = 0.0
                for d in range(rank):
                    dot += noise[k * sites * rank + site * rank + d] * sigma[s * rank + d]
                var e = 1.0 / (1.0 + exp(-(dot + mu[site * species + s])))
                e = e * 0.999999 + 0.0000005
                var ys = y[site * species + s]
                acc_ll += ys * log(e) + (1.0 - ys) * log(1.0 - e)
            if acc_ll > max_ll:
                run_sum = run_sum * exp(max_ll - acc_ll) + 1.0
                max_ll = acc_ll
            else:
                run_sum += exp(acc_ll - max_ll)
        out[site] = -(log(run_sum / Float32(samples)) + max_ll)

    parallelize[site_loss](sites)

    var byte_ptr = out.bitcast[UInt8]()
    var ob = Span[UInt8](unsafe_ptr=byte_ptr, length=sites * 4)
    Path(args[9]).write_bytes(ob)

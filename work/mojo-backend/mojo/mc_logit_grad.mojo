# Phase 2 forward+gradient kernel: binary logit Monte Carlo likelihood.
#
# Same inputs as mc_logit_loss.mojo, plus three outputs:
#   argv[9]  out    : sites               (per-site loss)
#   argv[10] gmu    : sites x species     (d total_loss / d mu)
#   argv[11] gsigma : species x rank      (d total_loss / d sigma)
#   argv[12] alpha  : scale factor (optional, default 1.0); z = alpha*(noise . sigma + mu)
#
# total_loss = sum_i loss_i,  loss_i = -log(mean_k exp(ll_ik))
#   z_iks = noise_kid . sigma_sd + mu_is ; E = clip(sigmoid(z))
#   ll_ik = sum_s y log E + (1-y) log(1-E)
#   w_ik  = exp(ll_ik - max_ll)/run_sum / K   (softmax over k)
#   dz_iks = w_ik * (E-y)/(E(1-E)) * 0.999999 * s(1-s), s = sigmoid(z)
#   gmu_is = sum_k dz_iks ; gsigma_sd = sum_{i,k} dz_iks * noise_kid
#
# Sites are split into N_CHUNKS contiguous chunks; each task accumulates a
# private gsigma chunk (race-free), merged serially afterwards.

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

    var alpha: Float32 = 1.0
    if len(args) > 12:
        alpha = Float32(atof(args[12]))

    comptime N_CHUNKS = 16

    var out = alloc[Float32](sites)
    var gmu = alloc[Float32](sites * species)
    var gsigma_buf = alloc[Float32](N_CHUNKS * species * rank)
    # alloc does not zero-initialize
    for i in range(sites * species):
        gmu[i] = 0.0
    for i in range(N_CHUNKS * species * rank):
        gsigma_buf[i] = 0.0
    var gsigma = alloc[Float32](species * rank)

    var chunk = (sites + N_CHUNKS - 1) // N_CHUNKS

    @parameter
    def chunk_loss(cid: Int):
        var start = cid * chunk
        var stop = min(start + chunk, sites)
        var my_gsigma = cid * species * rank

        for site in range(start, stop):
            # Pass 1: streaming log-sum-exp of per-sample log-likelihoods
            var max_ll: Float32 = -3.0e38
            var run_sum: Float32 = 0.0
            for k in range(samples):
                var acc_ll: Float32 = 0.0
                for s in range(species):
                    var dot: Float32 = 0.0
                    for d in range(rank):
                        dot += noise[k * sites * rank + site * rank + d] * sigma[s * rank + d]
                    var e = 1.0 / (1.0 + exp(-alpha * (dot + mu[site * species + s])))
                    e = e * 0.999999 + 0.0000005
                    var ys = y[site * species + s]
                    acc_ll += ys * log(e) + (1.0 - ys) * log(1.0 - e)
                if acc_ll > max_ll:
                    run_sum = run_sum * exp(max_ll - acc_ll) + 1.0
                    max_ll = acc_ll
                else:
                    run_sum += exp(acc_ll - max_ll)

            out[site] = -(log(run_sum / Float32(samples)) + max_ll)

            # Pass 2: per-sample weights, then gradient accumulation
            var inv_run = 1.0 / run_sum
            var gmu_base = site * species
            for k in range(samples):
                var acc_ll: Float32 = 0.0
                for s in range(species):
                    var dot: Float32 = 0.0
                    for d in range(rank):
                        dot += noise[k * sites * rank + site * rank + d] * sigma[s * rank + d]
                    var e = 1.0 / (1.0 + exp(-alpha * (dot + mu[site * species + s])))
                    e = e * 0.999999 + 0.0000005
                    var ys = y[site * species + s]
                    acc_ll += ys * log(e) + (1.0 - ys) * log(1.0 - e)
                # softmax over MC samples; sums to 1, so no extra 1/K here
                # (the 1/K inside log(mean(...)) cancels in the derivative)
                var w = exp(acc_ll - max_ll) * inv_run

                for s in range(species):
                    var dot: Float32 = 0.0
                    for d in range(rank):
                        dot += noise[k * sites * rank + site * rank + d] * sigma[s * rank + d]
                    var z = alpha * (dot + mu[site * species + s])
                    var sig = 1.0 / (1.0 + exp(-z))
                    var e = sig * 0.999999 + 0.0000005
                    var ys = y[site * species + s]
                    var dz = (w * (e - ys) / (e * (1.0 - e))
                              * 0.999999 * sig * (1.0 - sig) * alpha)
                    gmu[gmu_base + s] += dz
                    for d in range(rank):
                        gsigma_buf[my_gsigma + s * rank + d] += dz * noise[k * sites * rank + site * rank + d]

    parallelize[chunk_loss](N_CHUNKS)

    # merge per-chunk sigma gradients
    for s in range(species):
        for d in range(rank):
            var acc: Float32 = 0.0
            for c in range(N_CHUNKS):
                acc += gsigma_buf[c * species * rank + s * rank + d]
            gsigma[s * rank + d] = acc

    var ob = Span[UInt8](unsafe_ptr=out.bitcast[UInt8](), length=sites * 4)
    Path(args[9]).write_bytes(ob)
    var gb = Span[UInt8](unsafe_ptr=gmu.bitcast[UInt8](), length=sites * species * 4)
    Path(args[10]).write_bytes(gb)
    var gs = Span[UInt8](unsafe_ptr=gsigma.bitcast[UInt8](), length=species * rank * 4)
    Path(args[11]).write_bytes(gs)

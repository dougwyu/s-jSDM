# Persistent-process variant of mc_logit_grad.mojo.
#
# Runs as a long-lived worker reading requests from stdin and writing
# results to stdout (raw little-endian bytes). One request:
#   header : Int64 sites, Int64 species, Int64 rank, Int64 samples,
#            UInt32 alpha_bits (IEEE-754 float32 bit pattern),
#            UInt32 mode (0 = noise payload, 1 = seed)
#   mu     : sites x species float32
#   sigma  : species x rank float32
#   y      : sites x species float32
#   mode 0 : noise : samples x sites x rank float32
#   mode 1 : seed   : UInt64 (server generates standard normals)
# Response:
#   out    : sites float32
#   gmu    : sites x species float32
#   gsigma : species x rank float32
#
# A zero-byte read on the header means EOF and exits cleanly.
#
# Computation is identical to mc_logit_grad.mojo (see comments there):
#   z = alpha*(noise . sigma_s + mu_i); E = clip(sigmoid(z))
#   loss_i = -log(mean_k exp(ll_ik)); softmax weights w_ik
#   dz includes the alpha factor; sites split into N_CHUNKS chunks with
#   private per-chunk sigma-gradient buffers merged afterwards.
#
# Seed mode generates N(0,1) draws as box_muller(splitmix64(seed ^ idx)),
# one independent stream element per noise entry, so generation can be
# parallelized without inter-chunk dependencies.

from max.algorithm import parallelize
from std.io import FileDescriptor
from std.math import exp, log, sqrt
from std.memory import alloc, dealloc
from std.sys import argv, stdin, stdout


comptime HEADER_BYTES = 40
comptime BLOCK = 4096
comptime N_GEN_BLOCKS = 64


def splitmix64(state: UInt64) -> UInt64:
    var mut_state = state
    mut_state = mut_state + UInt64(0x9E3779B97F4A7C15)
    var z = mut_state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def gen_noise(mut noise: Pointer[Float32, MutUntrackedOrigin], n: Int, seed: UInt64):
    var n_workers = min(N_GEN_BLOCKS, (n + BLOCK - 1) // BLOCK)

    @parameter
    def fill(worker_id: Int):
        var start = worker_id * BLOCK
        var stride = n_workers * BLOCK
        while start < n:
            var stop = min(start + BLOCK, n)
            var idx = start
            while idx < stop:
                var attempt: UInt64 = 0
                while True:
                    var u1f = Float32((splitmix64(seed ^ UInt64(idx) ^ (attempt << 32)) >> 8) & 0xFFFFFF) * (1.0 / 8388608.0) - 1.0
                    var u2f = Float32((splitmix64(seed ^ UInt64(idx + 1) ^ (attempt << 32)) >> 8) & 0xFFFFFF) * (1.0 / 8388608.0) - 1.0
                    var ss = u1f * u1f + u2f * u2f
                    if ss < 1.0 and ss > 0.0:
                        var fac = sqrt(-2.0 * log(ss) / ss)
                        noise[idx] = u1f * fac
                        if idx + 1 < n:
                            noise[idx + 1] = u2f * fac
                        break
                    attempt += 1
                idx += 2
            start += stride

    parallelize[fill](n_workers)


def read_exact(mut sin: FileDescriptor, ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) raises -> Bool:
    # Pipes can return short reads; loop until the span is filled.
    var done: Int = 0
    while done < n:
        var sub = Span[UInt8](unsafe_ptr=ptr + done, length=n - done)
        var got = sin.read_bytes(sub)
        if got == 0:
            return False
        done += got
    return True


def main() raises:
    var sin = stdin
    var sout = stdout

    var hbuf = alloc[UInt8](HEADER_BYTES)
    var seedbuf = alloc[UInt8](8)
    # Reusable buffers grow independently to the high-water capacity
    # required by requests handled by this worker.
    var cap_mu = 0
    var cap_sigma = 0
    var cap_y = 0
    var cap_noise = 0
    var cap_out = 0
    var cap_gmu = 0
    var cap_gsigma_buf = 0
    var cap_gsigma = 0
    var cap_zbuf_all = 0
    var cap_llbuf_all = 0
    var mu = alloc[Float32](0)
    var sigma = alloc[Float32](0)
    var y = alloc[Float32](0)
    var noise = alloc[Float32](0)
    comptime N_CHUNKS = 16
    # Output and scratch buffers, reused across requests like the inputs.
    var out = alloc[Float32](0)
    var gmu = alloc[Float32](0)
    var gsigma_buf = alloc[Float32](0)
    var gsigma = alloc[Float32](0)
    var zbuf_all = alloc[Float32](0)
    var llbuf_all = alloc[Float32](0)

    while True:
        if not read_exact(sin, hbuf, HEADER_BYTES):
            break
        var h = Span[UInt8](unsafe_ptr=hbuf, length=HEADER_BYTES)
        var dims = h.unsafe_ptr().unsafe_bitcast[Int64]()
        var sites = Int(dims[0])
        var species = Int(dims[1])
        var rank = Int(dims[2])
        var samples = Int(dims[3])
        var alpha = h.unsafe_ptr().unsafe_bitcast[Float32]()[8]
        var mode = h.unsafe_ptr().unsafe_bitcast[UInt32]()[9]

        var n_mu = sites * species
        var n_sigma = species * rank
        var n_noise = samples * sites * rank
        var n_out = sites
        var n_gsigma_buf = N_CHUNKS * species * rank
        var n_zbuf_all = N_CHUNKS * samples * species
        var n_llbuf_all = N_CHUNKS * samples

        if n_mu > cap_mu:
            mu.unsafe_free()
            mu = alloc[Float32](n_mu)
            cap_mu = n_mu
        if n_sigma > cap_sigma:
            sigma.unsafe_free()
            sigma = alloc[Float32](n_sigma)
            cap_sigma = n_sigma
        if n_mu > cap_y:
            y.unsafe_free()
            y = alloc[Float32](n_mu)
            cap_y = n_mu
        if n_noise > cap_noise:
            noise.unsafe_free()
            noise = alloc[Float32](n_noise)
            cap_noise = n_noise
        if n_out > cap_out:
            out.unsafe_free()
            out = alloc[Float32](n_out)
            cap_out = n_out
        if n_mu > cap_gmu:
            gmu.unsafe_free()
            gmu = alloc[Float32](n_mu)
            cap_gmu = n_mu
        if n_gsigma_buf > cap_gsigma_buf:
            gsigma_buf.unsafe_free()
            gsigma_buf = alloc[Float32](n_gsigma_buf)
            cap_gsigma_buf = n_gsigma_buf
        if n_sigma > cap_gsigma:
            gsigma.unsafe_free()
            gsigma = alloc[Float32](n_sigma)
            cap_gsigma = n_sigma
        if n_zbuf_all > cap_zbuf_all:
            zbuf_all.unsafe_free()
            zbuf_all = alloc[Float32](n_zbuf_all)
            cap_zbuf_all = n_zbuf_all
        if n_llbuf_all > cap_llbuf_all:
            llbuf_all.unsafe_free()
            llbuf_all = alloc[Float32](n_llbuf_all)
            cap_llbuf_all = n_llbuf_all

        if not read_exact(sin, mu.bitcast[UInt8](), sites * species * 4):
            break
        if not read_exact(sin, sigma.bitcast[UInt8](), species * rank * 4):
            break
        if not read_exact(sin, y.bitcast[UInt8](), sites * species * 4):
            break
        if mode == 1:
            # seed transport: 8-byte seed, server generates the noise
            if not read_exact(sin, seedbuf, 8):
                break
            var seed = Span[UInt8](unsafe_ptr=seedbuf, length=8).unsafe_ptr().unsafe_bitcast[UInt64]()[0]
            gen_noise(noise, samples * sites * rank, seed)
        elif not read_exact(sin, noise.bitcast[UInt8](), samples * sites * rank * 4):
            break

        comptime N_CHUNKS = 16

        # alloc does not zero-initialize
        for i in range(sites * species):
            gmu[i] = 0.0
        for i in range(N_CHUNKS * species * rank):
            gsigma_buf[i] = 0.0

        var chunk = (sites + N_CHUNKS - 1) // N_CHUNKS

        @parameter
        def chunk_loss(cid: Int):
            var start = cid * chunk
            var stop = min(start + chunk, sites)
            var my_gsigma = cid * species * rank

            # Per-chunk scratch: cached z_ks = alpha*(noise_k . sigma_s + mu_i)
            # and per-sample log-likelihoods ll_ik, so the noise dot products
            # are computed exactly once per site (was three times).
            var zbuf = zbuf_all + cid * samples * species
            var llbuf = llbuf_all + cid * samples

            comptime W: Int = 4
            # rank split into a SIMD-full part and a scalar remainder;
            # recomputed once per site (constant across k and s)
            var sfull = species - (species % W)
            for site in range(start, stop):
                var zbase = 0
                var llbase = 0
                var full = rank - (rank % W)

                # Pass 1a: dot products once, cache z
                for k in range(samples):
                    var nz_base = k * sites * rank + site * rank
                    for s in range(species):
                        var dot: Float32 = 0.0
                        for d in range(0, full, W):
                            var nv = (noise + nz_base + d).load[W]()
                            var sv = (sigma + s * rank + d).load[W]()
                            dot += (nv * sv).reduce_add()
                        for d in range(full, rank):
                            dot += noise[nz_base + d] * sigma[s * rank + d]
                        zbuf[zbase + k * species + s] = alpha * (dot + mu[site * species + s])

                # Pass 1b: ll via SIMD sigmoid/log across species
                var max_ll: Float32 = -3.0e38
                var run_sum: Float32 = 0.0
                for k in range(samples):
                    var acc_ll: Float32 = 0.0
                    var zb = zbase + k * species
                    var yrow = site * species
                    for s in range(0, sfull, W):
                        var zv = (zbuf + zb + s).load[W]()
                        var e = 1.0 / (1.0 + exp(-zv))
                        e = e * 0.999999 + 0.0000005
                        var ys = (y + yrow + s).load[W]()
                        acc_ll += (ys * log(e) + (1.0 - ys) * log(1.0 - e)).reduce_add()
                    for s in range(sfull, species):
                        var z = zbuf[zb + s]
                        var e = 1.0 / (1.0 + exp(-z))
                        e = e * 0.999999 + 0.0000005
                        var ys = y[yrow + s]
                        acc_ll += ys * log(e) + (1.0 - ys) * log(1.0 - e)
                    llbuf[llbase + k] = acc_ll
                    if acc_ll > max_ll:
                        run_sum = run_sum * exp(max_ll - acc_ll) + 1.0
                        max_ll = acc_ll
                    else:
                        run_sum += exp(acc_ll - max_ll)

                out[site] = -(log(run_sum / Float32(samples)) + max_ll)

                # Pass 2: weights from cached ll, gradients from cached z.
                # sigmoid/exp vectorized across species; the gsigma update
                # stays per-lane because each species owns its own row.
                var inv_run = 1.0 / run_sum
                var gmu_base = site * species
                var wv = SIMD[DType.float32, W](1)
                var one = SIMD[DType.float32, W](1.0)
                var c1 = SIMD[DType.float32, W](0.999999)
                var c2 = SIMD[DType.float32, W](0.0000005)
                var av = SIMD[DType.float32, W](alpha)
                for k in range(samples):
                    # softmax over MC samples; sums to 1, so no extra 1/K here
                    # (the 1/K inside log(mean(...)) cancels in the derivative)
                    var w = exp(llbuf[llbase + k] - max_ll) * inv_run
                    var nz_base = k * sites * rank + site * rank
                    wv = SIMD[DType.float32, W](w)
                    for s in range(0, sfull, W):
                        var zv = (zbuf + zbase + k * species + s).load[W]()
                        var sig = 1.0 / (1.0 + exp(-zv))
                        var e = sig * c1 + c2
                        var ys = (y + gmu_base + s).load[W]()
                        var dz = (wv * (e - ys) / (e * (one - e)) * c1 * sig * (one - sig) * av)
                        var old = (gmu + gmu_base + s).load[W]()
                        (gmu + gmu_base + s).store(old + dz)
                        for j in range(W):
                            var dzj = dz[j]
                            var gp = gsigma_buf + my_gsigma + (s + j) * rank
                            for d in range(0, full, W):
                                var nv = (noise + nz_base + d).load[W]()
                                gp.store(d, (gp + d).load[W]() + SIMD[DType.float32, W](dzj) * nv)
                            for d in range(full, rank):
                                gsigma_buf[my_gsigma + (s + j) * rank + d] += dzj * noise[nz_base + d]
                    for s in range(sfull, species):
                        var sig = 1.0 / (1.0 + exp(-zbuf[zbase + k * species + s]))
                        var e = sig * 0.999999 + 0.0000005
                        var ys = y[gmu_base + s]
                        var dz = (w * (e - ys) / (e * (1.0 - e))
                                  * 0.999999 * sig * (1.0 - sig) * alpha)
                        gmu[gmu_base + s] += dz
                        var gp = gsigma_buf + my_gsigma + s * rank
                        for d in range(0, full, W):
                            var nv = (noise + nz_base + d).load[W]()
                            gp.store(d, (gp + d).load[W]() + SIMD[DType.float32, W](dz) * nv)
                        for d in range(full, rank):
                            gsigma_buf[my_gsigma + s * rank + d] += dz * noise[nz_base + d]

        parallelize[chunk_loss](N_CHUNKS)

        # merge per-chunk sigma gradients
        for s in range(species):
            for d in range(rank):
                var acc: Float32 = 0.0
                for c in range(N_CHUNKS):
                    acc += gsigma_buf[c * species * rank + s * rank + d]
                gsigma[s * rank + d] = acc

        sout.write_bytes(Span[UInt8](unsafe_ptr=out.bitcast[UInt8](), length=sites * 4))
        sout.write_bytes(Span[UInt8](unsafe_ptr=gmu.bitcast[UInt8](), length=sites * species * 4))
        sout.write_bytes(Span[UInt8](unsafe_ptr=gsigma.bitcast[UInt8](), length=species * rank * 4))

    hbuf.unsafe_free()
    seedbuf.unsafe_free()
    mu.unsafe_free()
    sigma.unsafe_free()
    y.unsafe_free()
    noise.unsafe_free()
    out.unsafe_free()
    gmu.unsafe_free()
    gsigma_buf.unsafe_free()
    gsigma.unsafe_free()
    zbuf_all.unsafe_free()
    llbuf_all.unsafe_free()

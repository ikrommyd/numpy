/*
 * Benchmark-only fused min/max reduction kernels.
 *
 * These mirror the optimizations in `loops_minmax.dispatch.c.src` (SIMD via
 * the universal intrinsics, 8x unrolled contiguous reduce, unrolled
 * contiguous binary loop, scalar unrolls) but compute the running minimum
 * and maximum in a single pass over the input.
 *
 * Only `double` is implemented, which is all the benchmark needs.
 */
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "simd/simd.h"
#include "numpy/npy_common.h"   // NPY_PREFETCH
#include "numpy/npy_math.h"     // npy_isnan
#include "npy_cpu_dispatch.h"

#include "_minmax_bench.dispatch.h"

/* NaN-propagating, same as `scalar_min`/`scalar_max` in loops_minmax */
#define SCALAR_MIN(A, B) ((A <= B || npy_isnan(A)) ? A : B)
#define SCALAR_MAX(A, B) ((A >= B || npy_isnan(A)) ? A : B)

NPY_CPU_DISPATCH_DECLARE(void minmax_bench_reduce_c,
        (const double *ip, double *op_min, double *op_max, npy_intp len))
NPY_CPU_DISPATCH_DECLARE(void minmax_bench_binary_ccc,
        (const double *ip, double *io_min, double *io_max, npy_intp len))


/*
 * Contiguous reduction: the accumulators are scalars (stride 0) and the
 * streamed input is contiguous.  This is the `axis=-1` shape of the problem.
 */
void NPY_CPU_DISPATCH_CURFX(minmax_bench_reduce_c)
(const double *ip, double *op_min, double *op_max, npy_intp len)
{
    if (len < 1) {
        return;
    }
    double rmin = op_min[0];
    double rmax = op_max[0];

#if NPY_SIMD_F64
    const int vstep = npyv_nlanes_f64;
    const int wstep = vstep * 8;

    if (len >= wstep) {
        npyv_f64 acc_min = npyv_setall_f64(rmin);
        npyv_f64 acc_max = npyv_setall_f64(rmax);
        for (; len >= wstep; len -= wstep, ip += wstep) {
        #ifdef NPY_HAVE_SSE2
            NPY_PREFETCH((const char*)(ip + wstep), 0, 3);
        #endif
            /*
             * The eight loads are shared between the min and the max tree,
             * which is the whole point of fusing the two reductions.
             */
            npyv_f64 v0 = npyv_load_f64(ip + vstep * 0);
            npyv_f64 v1 = npyv_load_f64(ip + vstep * 1);
            npyv_f64 v2 = npyv_load_f64(ip + vstep * 2);
            npyv_f64 v3 = npyv_load_f64(ip + vstep * 3);
            npyv_f64 v4 = npyv_load_f64(ip + vstep * 4);
            npyv_f64 v5 = npyv_load_f64(ip + vstep * 5);
            npyv_f64 v6 = npyv_load_f64(ip + vstep * 6);
            npyv_f64 v7 = npyv_load_f64(ip + vstep * 7);

            npyv_f64 n01 = npyv_minn_f64(v0, v1);
            npyv_f64 n23 = npyv_minn_f64(v2, v3);
            npyv_f64 n45 = npyv_minn_f64(v4, v5);
            npyv_f64 n67 = npyv_minn_f64(v6, v7);
            acc_min = npyv_minn_f64(acc_min,
                    npyv_minn_f64(npyv_minn_f64(n01, n23), npyv_minn_f64(n45, n67)));

            npyv_f64 x01 = npyv_maxn_f64(v0, v1);
            npyv_f64 x23 = npyv_maxn_f64(v2, v3);
            npyv_f64 x45 = npyv_maxn_f64(v4, v5);
            npyv_f64 x67 = npyv_maxn_f64(v6, v7);
            acc_max = npyv_maxn_f64(acc_max,
                    npyv_maxn_f64(npyv_maxn_f64(x01, x23), npyv_maxn_f64(x45, x67)));
        }
        for (; len >= vstep; len -= vstep, ip += vstep) {
            npyv_f64 v = npyv_load_f64(ip);
            acc_min = npyv_minn_f64(acc_min, v);
            acc_max = npyv_maxn_f64(acc_max, v);
        }
        rmin = npyv_reduce_minn_f64(acc_min);
        rmax = npyv_reduce_maxn_f64(acc_max);
    }
    npyv_cleanup();
#endif  // NPY_SIMD_F64

    /* scalar unroll / tail */
    for (; len > 0; --len, ++ip) {
        const double v = *ip;
        rmin = SCALAR_MIN(rmin, v);
        rmax = SCALAR_MAX(rmax, v);
    }
    op_min[0] = rmin;
    op_max[0] = rmax;
}


/*
 * Contiguous "binary" reduction: the accumulators are contiguous arrays that
 * are read and written each element, as when reducing over an outer axis.
 * `io_min`/`io_max` are both the accumulators and the outputs.
 */
void NPY_CPU_DISPATCH_CURFX(minmax_bench_binary_ccc)
(const double *ip, double *io_min, double *io_max, npy_intp len)
{
    npy_intp i = 0;

#if NPY_SIMD_F64
    #if NPY_SIMD_WIDTH == 128
        // Note, 6x unroll was chosen for best results on Apple M1
        const int vectorsPerLoop = 6;
    #else
        // To avoid memory bandwidth bottleneck
        const int vectorsPerLoop = 2;
    #endif
    const int elemPerVector = npyv_nlanes_f64;
    const npy_intp elemPerLoop = vectorsPerLoop * elemPerVector;

    for (; (i + elemPerLoop) <= len; i += elemPerLoop) {
        npyv_f64 v0 = npyv_load_f64(&ip[i + 0 * elemPerVector]);
        npyv_f64 v1 = npyv_load_f64(&ip[i + 1 * elemPerVector]);
    #if NPY_SIMD_WIDTH == 128
        npyv_f64 v2 = npyv_load_f64(&ip[i + 2 * elemPerVector]);
        npyv_f64 v3 = npyv_load_f64(&ip[i + 3 * elemPerVector]);
        npyv_f64 v4 = npyv_load_f64(&ip[i + 4 * elemPerVector]);
        npyv_f64 v5 = npyv_load_f64(&ip[i + 5 * elemPerVector]);
    #endif
        npyv_f64 a0 = npyv_load_f64(&io_min[i + 0 * elemPerVector]);
        npyv_f64 a1 = npyv_load_f64(&io_min[i + 1 * elemPerVector]);
    #if NPY_SIMD_WIDTH == 128
        npyv_f64 a2 = npyv_load_f64(&io_min[i + 2 * elemPerVector]);
        npyv_f64 a3 = npyv_load_f64(&io_min[i + 3 * elemPerVector]);
        npyv_f64 a4 = npyv_load_f64(&io_min[i + 4 * elemPerVector]);
        npyv_f64 a5 = npyv_load_f64(&io_min[i + 5 * elemPerVector]);
    #endif
        npyv_store_f64(&io_min[i + 0 * elemPerVector], npyv_minn_f64(a0, v0));
        npyv_store_f64(&io_min[i + 1 * elemPerVector], npyv_minn_f64(a1, v1));
    #if NPY_SIMD_WIDTH == 128
        npyv_store_f64(&io_min[i + 2 * elemPerVector], npyv_minn_f64(a2, v2));
        npyv_store_f64(&io_min[i + 3 * elemPerVector], npyv_minn_f64(a3, v3));
        npyv_store_f64(&io_min[i + 4 * elemPerVector], npyv_minn_f64(a4, v4));
        npyv_store_f64(&io_min[i + 5 * elemPerVector], npyv_minn_f64(a5, v5));
    #endif
        npyv_f64 b0 = npyv_load_f64(&io_max[i + 0 * elemPerVector]);
        npyv_f64 b1 = npyv_load_f64(&io_max[i + 1 * elemPerVector]);
    #if NPY_SIMD_WIDTH == 128
        npyv_f64 b2 = npyv_load_f64(&io_max[i + 2 * elemPerVector]);
        npyv_f64 b3 = npyv_load_f64(&io_max[i + 3 * elemPerVector]);
        npyv_f64 b4 = npyv_load_f64(&io_max[i + 4 * elemPerVector]);
        npyv_f64 b5 = npyv_load_f64(&io_max[i + 5 * elemPerVector]);
    #endif
        npyv_store_f64(&io_max[i + 0 * elemPerVector], npyv_maxn_f64(b0, v0));
        npyv_store_f64(&io_max[i + 1 * elemPerVector], npyv_maxn_f64(b1, v1));
    #if NPY_SIMD_WIDTH == 128
        npyv_store_f64(&io_max[i + 2 * elemPerVector], npyv_maxn_f64(b2, v2));
        npyv_store_f64(&io_max[i + 3 * elemPerVector], npyv_maxn_f64(b3, v3));
        npyv_store_f64(&io_max[i + 4 * elemPerVector], npyv_maxn_f64(b4, v4));
        npyv_store_f64(&io_max[i + 5 * elemPerVector], npyv_maxn_f64(b5, v5));
    #endif
    }
    for (; (i + elemPerVector) <= len; i += elemPerVector) {
        npyv_f64 v = npyv_load_f64(&ip[i]);
        npyv_store_f64(&io_min[i], npyv_minn_f64(npyv_load_f64(&io_min[i]), v));
        npyv_store_f64(&io_max[i], npyv_maxn_f64(npyv_load_f64(&io_max[i]), v));
    }
    npyv_cleanup();
#endif  // NPY_SIMD_F64

    for (; i < len; ++i) {
        const double v = ip[i];
        io_min[i] = SCALAR_MIN(io_min[i], v);
        io_max[i] = SCALAR_MAX(io_max[i], v);
    }
}

/*
 * Benchmark-only `minimummaximum` ufunc: a 2-in/2-out loop with a fused
 * (nout+1)->nout reduction loop registered through
 * NPY_METH_get_reduction_loop.
 *
 * Unlike `_reduction_loop_tests`, this one uses the same SIMD kernels and
 * unrolling that `loops_minmax.dispatch.c.src` uses for `np.minimum` and
 * `np.maximum`, so that fused-vs-two-pass can be compared at equal effort.
 * Only `float64` is implemented.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#if defined(NPY_INTERNAL_BUILD)
#undef NPY_INTERNAL_BUILD
#endif
#define NPY_TARGET_VERSION NPY_2_6_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
#include "numpy/npy_math.h"

#include "npy_cpu_dispatch.h"
#include "npy_cpu_features.h"

#include "_minmax_bench.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(void minmax_bench_reduce_c,
        (const double *ip, double *op_min, double *op_max, npy_intp len))
NPY_CPU_DISPATCH_DECLARE(void minmax_bench_binary_ccc,
        (const double *ip, double *io_min, double *io_max, npy_intp len))

#define SCALAR_MIN(A, B) ((A <= B || npy_isnan(A)) ? A : B)
#define SCALAR_MAX(A, B) ((A >= B || npy_isnan(A)) ? A : B)


static int
minmax_forward_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp n = dimensions[0];
    char *in1 = data[0], *in2 = data[1];
    char *out1 = data[2], *out2 = data[3];

    for (npy_intp i = 0; i < n; i++) {
        double a = *(double *)in1;
        double b = *(double *)in2;
        *(double *)out1 = SCALAR_MIN(a, b);
        *(double *)out2 = SCALAR_MAX(a, b);
        in1 += strides[0]; in2 += strides[1];
        out1 += strides[2]; out2 += strides[3];
    }
    return 0;
}


/*
 * Fused reduction loop, laid out as
 *     [acc_min, acc_max, x, out_min, out_max]
 * with out_i aliased to acc_i.  Specializes on the two shapes the reduce
 * machinery actually produces, exactly as loops_minmax does.
 */
static int
minmax_reduce_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp n = dimensions[0];
    const npy_intp elsize = (npy_intp)sizeof(double);

    /* reducing over the contiguous axis: accumulators are scalars */
    if (strides[0] == 0 && strides[1] == 0
            && strides[3] == 0 && strides[4] == 0
            && strides[2] == elsize) {
        NPY_CPU_DISPATCH_CALL(minmax_bench_reduce_c,
                ((const double *)data[2], (double *)data[3],
                 (double *)data[4], n));
        return 0;
    }
    /* reducing over an outer axis: everything contiguous */
    if (strides[0] == elsize && strides[1] == elsize && strides[2] == elsize
            && strides[3] == elsize && strides[4] == elsize) {
        NPY_CPU_DISPATCH_CALL(minmax_bench_binary_ccc,
                ((const double *)data[2], (double *)data[3],
                 (double *)data[4], n));
        return 0;
    }

    /* generic strided fallback */
    char *acc_min = data[0], *acc_max = data[1], *x = data[2];
    char *out_min = data[3], *out_max = data[4];
    for (npy_intp i = 0; i < n; i++) {
        double v = *(double *)x;
        *(double *)out_min = SCALAR_MIN(*(double *)acc_min, v);
        *(double *)out_max = SCALAR_MAX(*(double *)acc_max, v);
        acc_min += strides[0]; acc_max += strides[1]; x += strides[2];
        out_min += strides[3]; out_max += strides[4];
    }
    return 0;
}


static int
minmax_get_reduction_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *out_loop = &minmax_reduce_loop;
    *out_transferdata = NULL;
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    return 0;
}


static int
minmax_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (descr == NULL) {
        return -1;
    }
    PyArray_DTypeMeta *dt = NPY_DTYPE(descr);
    for (int i = 0; i < 4; i++) {
        PyArray_DTypeMeta *d = signature[i] != NULL ? signature[i] : dt;
        Py_INCREF(d);
        new_op_dtypes[i] = d;
    }
    Py_DECREF(descr);
    return 0;
}


static PyMethodDef MinMaxBenchMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_minmax_bench", NULL, -1, MinMaxBenchMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__minmax_bench(void)
{
    if (npy_cpu_init() < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    if (PyArray_ImportNumPyAPI() < 0 || PyUFunc_ImportUFuncAPI() < 0) {
        Py_DECREF(m);
        return NULL;
    }

    PyObject *uf = PyUFunc_FromFuncAndData(
            NULL, NULL, NULL, 0, 2, 2, PyUFunc_None,
            "minimummaximum", "fused min/max", 0);
    if (uf == NULL) {
        Py_DECREF(m);
        return NULL;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_DTypeMeta *dt = NPY_DTYPE(descr);
    PyArray_DTypeMeta *dtypes[4] = {dt, dt, dt, dt};

    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, (void *)&minmax_forward_loop},
        {NPY_METH_get_reduction_loop, (void *)&minmax_get_reduction_loop},
        {0, NULL},
    };
    PyArrayMethod_Spec spec = {
        .name = "double_minimummaximum_bench",
        .nin = 2,
        .nout = 2,
        .casting = NPY_NO_CASTING,
        .flags = NPY_METH_IS_REORDERABLE | NPY_METH_NO_FLOATINGPOINT_ERRORS,
        .dtypes = dtypes,
        .slots = slots,
    };
    int res = PyUFunc_AddLoopFromSpec(uf, &spec);
    Py_DECREF(descr);
    if (res < 0) {
        Py_DECREF(uf);
        Py_DECREF(m);
        return NULL;
    }

    PyObject *none_tuple = PyTuple_Pack(4, Py_None, Py_None, Py_None, Py_None);
    PyObject *promoter = PyCapsule_New(
            (void *)&minmax_promoter, "numpy._ufunc_promoter", NULL);
    if (none_tuple == NULL || promoter == NULL
            || PyUFunc_AddPromoter(uf, none_tuple, promoter) < 0
            || PyModule_AddObject(m, "minimummaximum", uf) < 0) {
        Py_XDECREF(none_tuple);
        Py_XDECREF(promoter);
        Py_DECREF(uf);
        Py_DECREF(m);
        return NULL;
    }
    Py_DECREF(none_tuple);
    Py_DECREF(promoter);

#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
    return m;
}

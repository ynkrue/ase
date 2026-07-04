# cuEV — Coding Conventions

## File Headers

Every `.cu` and `.cuh` file begins with a Doxygen block:

```cpp
/**
 * @file   dbbr.cu
 * @brief  One-line description.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */
```

---

## Naming

| Thing | Convention | Example |
|---|---|---|
| Device kernel (`__global__`) | `<prefix>_<op>_kernel` in anonymous namespace | `dbbr_syr2k_kernel` |
| Host launcher | `<prefix>_<op>` in `cuev::kernels` | `dbbr_syr2k` |
| cuBLAS wrapper | function name only in `cuev::cublas` | `cublas::gemm` |
| cuSOLVER wrapper | function name only in `cuev::cusolver` | `cusolver::geqrf` |
| Kernel prefixes | `dbbr_` band reduction, `bc_` bulge chasing + BC-Back, `bt_` back-transform | |
| Device pointer (host code) | `d` prefix | `dA`, `d_eval` |
| Host pointer | `h` prefix | `hA`, `h_eval` |
| Block size | `constexpr int BLOCK = N` inside launcher | |

---

## Namespaces

```
cuev              public API: symm_eig_solve, SolverHandle, handle_alloc/free
cuev::kernels     custom GPU kernel launchers (dbbr_*, bc_*, bt_*)
cuev::cublas      type-dispatching cuBLAS wrappers — all take SolverHandle<T>*
cuev::cusolver    workspace-aware cuSOLVER wrappers — all take SolverHandle<T>*
```

`__global__` kernels live in anonymous namespaces inside their `.cu` files — never exported.

---

## Handle pattern

All cuBLAS and cuSOLVER wrappers take `SolverHandle<T> *ws` as the first argument and
extract the handle internally. Never pass `cublasHandle_t` or `cusolverDnHandle_t` directly
at call sites.

```cpp
// correct
cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);

// wrong — old style, do not use
cublas::gemm(ws->cublas_handle, CUBLAS_OP_N, ...);
```

---

## Templates

- All numeric code templated on `T` (float or double)
- Use `if constexpr (std::is_same_v<T, float>)` for type dispatch
- Every `.cu` ends with explicit instantiations via the `INSTANTIATE` macro:

```cpp
#define INSTANTIATE(T) \
    template void dbbr_panel_qr<T>(SolverHandle<T> *, int, int);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
```

---

## Function Documentation

### `kernels.cuh` — full Doxygen per function

```cpp
/**
 * @brief Symmetric rank-2k update: A ← A − Z·Yᵀ − Y·Zᵀ
 *
 * @tparam T      float or double
 * @param[in]     ws    solver handle
 * @param[in,out] A     n×n symmetric matrix, column-major; updated in place
 * @param[in]     Z     n×k matrix, column-major
 * @param[in]     Y     n×k matrix, column-major
 * @param[in]     n     matrix dimension
 * @param[in]     k     number of columns in Z and Y
 */
template <typename T>
void dbbr_syr2k(SolverHandle<T> *ws, T *A, const T *Z, const T *Y, int n, int k);
```

Rules:
- `@brief` — single sentence ending with a period
- Math uses Unicode: `←`, `ᵀ`, `Σ`, `·`, `α`, `β`
- `@param` direction tags: `[in]`, `[out]`, `[in,out]`
- `@tparam T` is always `float or double`

### `.cu` bodies — non-obvious WHY only

Only comment the **why**: numerical subtlety, hardware constraint, workaround. Never restate
what the code does.

---

## Error Handling

- `CUDA_CHECK(...)` — every CUDA runtime call in host code
- `CUBLAS_CHECK(...)` — every cuBLAS call
- `CUSOLVER_CHECK(...)` — every cuSOLVER call
- No error checking inside `__global__` kernels

---

## Memory

- All matrices **column-major** (matches cuBLAS/cuSOLVER; leading dimension = number of rows)
- No `cudaMalloc` / `cudaFree` in the hot path — use `SolverHandle::pool` or pre-allocated buffers
- Eigenvectors stored as **columns**: `evec[j * n + i]` = i-th component of j-th eigenvector

---

## Kernel Design Patterns

- Reductions: block-reduce with `__syncthreads` → warp-reduce with `__shfl_down_sync(0xffffffff, val, N)`
- Block size: always compile-time constant (`constexpr` or template param)
- Prefer 1D `dim3 block(N)` with explicit index arithmetic over 2D blocks
- Annotate kernels with fixed thread counts: `__launch_bounds__(NUM_THREADS)`

---

## Braces

Always use `{}` except for single-line guard clauses that only `return`:

```cpp
if (row >= n) return;              // OK — guard clause
if (tid < s) { smem[tid] += ...; } // braces required
```

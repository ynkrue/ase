# ASE — Accelerated Symmetric Eigensolver

Real symmetric dense eigensolver for NVIDIA GPUs.

```cpp
ase::symm_eig_solve<T>(T *A, int n, T *eval, T *evec, cudaStream_t stream);
```

- `A` — n×n real symmetric, column-major, device pointer, overwritten on return.
- `eval` — eigenvalues ascending, length n.
- `evec` — eigenvectors as columns (column j = j-th eigenvector), n×n column-major.

## Algorithm

2-stage tridiagonalization EVD:

1. **Double Blocking Band Reduction (DBBR)** — symmetric dense → banded via compact WY panels.
2. **Data Repacking** — extract band into packed N×b contiguous array.
3. **Wavefront Bulge Chasing** — banded → tridiagonal via persistent kernel with point-to-point atomics.
4. **Divide & Conquer** — tridiagonal eigensolve via LAPACK `*stedc`.
5. **Back-Transformation** — Q = Q_s · Q_b · Q_d via register-sliding-window kernel + WY-block GEMMs.

References: Wang et al., "Improving Tridiagonalization Performance on GPU Architectures" (PPoPP'25);
Wang et al., "Rethinking Back Transformation in 2-stage EVD" (SC'25).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `ASE_ENABLE_OMP` | OFF | Multithreaded BLAS/LAPACK (MKL) |
| `ASE_ENABLE_TESTS` | ON | Build tests (`ase_test`) |
| `ASE_ENABLE_BENCH` | ON | Build benchmarks (`ase_bench`, `ase_bench_solver`) |
| `ASE_BUILD_DOCS` | OFF | Build Doxygen documentation |

## Tests

```bash
cmake -B build -DASE_ENABLE_TESTS=ON
cmake --build build
./build/ase_test
```

## Benchmarks

```bash
cmake -B build -DASE_ENABLE_BENCH=ON
cmake --build build
./build/ase_bench          # throughput vs cuSOLVER syevd
./build/ase_bench_solver   # full EVD with correctness validation
```

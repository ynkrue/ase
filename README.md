# ASE — Accelerated Symmetric Eigensolver

[![License MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)]()

Real symmetric dense eigensolver for a single NVIDIA GPU, double precision.

```cpp
ase::AseHandle *ws = ase::handle_alloc(stream);
ase::solve_ev(ws, A, n, eval, evec);       // host pointers
ase::handle_free(ws);
```

- `A` — n×n real symmetric, column-major, host pointer; only the lower triangle is read.
  Not modified.
- `eval` — eigenvalues ascending, length n (host).
- `evec` — eigenvectors as columns (column j = j-th eigenvector), n×n column-major (host).

Every solve goes through an `AseHandle`, an opaque type owned by the library: `handle_alloc`
only creates the cuBLAS/cuSOLVER contexts, and the workspace is sized lazily by the first
`solve_ev`/`solve_ev_d` call and cached across repeated solves of the same dimension — no
per-call allocation. A `solve_ev_d` overload takes device pointers directly (no H2D/D2H
copies, and none of the staging memory `solve_ev` needs) for callers who already manage
device memory. `include/ase/ase.h` is the only installed header and holds the full API.

## Algorithm

2-stage tridiagonalization EVD:

1. **Double Blocking Band Reduction (DBBR)** — symmetric dense → banded via compact WY panels.
2. **Data Repacking** — extract band into packed N×b contiguous array.
3. **Wavefront Bulge Chasing** — banded → tridiagonal via persistent kernel with point-to-point atomics.
4. **Divide & Conquer** — tridiagonal eigensolve via LAPACK `*stedc`.
5. **Back-Transformation** — Q = Q_s · Q_b · Q_d via register-sliding-window kernel + WY-block GEMMs.

## Requirements

- CMake >= 3.25
- CUDA Toolkit (nvcc + cuBLAS + cuSOLVER)
- A LAPACK implementation (MKL, OpenBLAS, or generic reference LAPACK)
- OpenMP (optional — parallelizes the leaf solves in the D&C stage; falls back to serial
  otherwise)
- `clang-format` (optional — enables the `format` / `format-check` targets)
- Doxygen + `dot` (optional — enables the `docs` target)

## Build

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

Toolchain (compiler, flags) is selected the usual CMake way via cache variables or
environment: `CUDACXX`, `CUDAFLAGS`, `CXXFLAGS`, `CUDAHOSTCXX`.

The LAPACK implementation is selected with `-DBLA_VENDOR=<...>` (passed through to
CMake's `FindLAPACK`), e.g. `Intel10_64lp` / `Intel10_64lp_seq` for MKL, `OpenBLAS`, or
`Generic`.

### Options

| Option | Default | Description |
|---|---|---|
| `ASE_BUILD_TESTS` | `ON` | Build the test binary (`ase_test`) |
| `ASE_BUILD_BENCH` | `ON` | Build the benchmark binary (`ase_bench`) |
| `ASE_BUILD_DOCS` | `OFF` | Add the Doxygen `docs` target |
| `ASE_WITH_OPENMP` | `ON` | Use OpenMP for the D&C solver |
| `BUILD_SHARED_LIBS` | `ON` | Build `ase` as a shared library (`OFF` for static) |
| `CMAKE_CUDA_ARCHITECTURES` | `90` | Target GPU compute capability, e.g. `80` for A100 |
| `CMAKE_BUILD_TYPE` | `Release` | Standard CMake build type |
| `BLA_VENDOR` | — | LAPACK vendor for `FindLAPACK`, e.g. `Intel10_64lp_seq`, `OpenBLAS` |

Example — MKL, Hopper, static library:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DBLA_VENDOR=Intel10_64lp_seq \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build -j
```

See `build.sh` for the environment used on the racklettes cluster (Kez-provided CUDA and
MKL via `CUDACXX` / `MKLROOT`).

## Tests

```bash
cmake --build build --target ase_test
./build/ase_test
```

or via CTest: `ctest --test-dir build`.

## Benchmarks

```bash
cmake --build build --target ase_bench
./build/ase_bench
```

## Documentation

```bash
cmake -S . -B build -DASE_BUILD_DOCS=ON
cmake --build build --target docs
```

Requires the `doxygen/` submodule (doxygen-awesome-css theme) for styled output:
`git submodule update --init`. Output is written to `docs/html`.

## References

- J. J. M. Cuppen, "A divide and conquer method for the symmetric tridiagonal eigenproblem," *Numerische Mathematik*, vol. 36, no. 2, pp. 177-195, 1980.
- C. Bischof and C. Van Loan, "The WY representation for products of Householder matrices," *SIAM Journal on Scientific and Statistical Computing*, vol. 8, no. 1, pp. s2-s13, 1987.
- H. Wang, S. Wu, Z. Duan, and S. Zheng, "Improving Tridiagonalization Performance on GPU Architectures," *Proceedings of the ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP)*, 2025.
- H. Wang et al., "Rethinking Back Transformation in 2-stage EVD," *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)*, 2025.

## License

MIT — see [LICENSE](LICENSE).

# cuEV — CUDA Eigensolver

Real symmetric dense eigensolver on NVIDIA GPUs.

```cpp
cuev::symm_eig_solve<T>(T *A, int n, T *eval, T *evec, cudaStream_t stream);
```

- `A` — n×n real symmetric, column-major, device pointer, overwritten on return.
- `eval` — eigenvalues ascending, length n.
- `evec` — eigenvectors as columns (column j = j-th eigenvector), n×n column-major.

## Algorithm

2-stage tridiagonalization EVD:

### 1. Double Blocking Band Reduction (DBBR): Symm → Banded
* **Mechanism**: Accumulate Householder reflections in compact $WY$ panels to increase arithmetic intensity.
* **Strategy**: Update the next panel (1st column blocking) and defer the full trailing matrix update until the working index reaches block $k$ (2nd panel blocking).

### 2. Data Repacking
* **Mechanism**: Extract the reduced band from the hollowed-out dense matrix.
* **Strategy**: Move the band into a packed $N \times b$ contiguous array. This eliminates non-contiguous memory accesses, ensuring the entire band fits into the L2 cache for subsequent stages.

### 3. Wavefront Bulge Chasing: Banded → Tridiagonal
* **Mechanism**: Launch a persistent CUDA kernel where thread blocks are statically assigned to horizontal band tiles.
* **Strategy**: Implement a wavefront pipeline where bulges are chased across tiles and passed across thread blocks. Synchronize thread block handoffs with point-to-point atomics (`cuda::atomic_thread_fence`, `wait`/`notify`).

### 4. Divide & Conquer (D&C)
* **Mechanism**: Tridiagonal eigensolve on the **CPU** via LAPACK `dstedc` (MKL). cuSOLVER exposes no standalone tridiagonal D&C (only dense `syevd`, which re-tridiagonalizes), so it is unusable here; running on the CPU also frees the GPU for the back-transform (stage 5).

### 5. Back-Transformation
* **Mechanism**: Direct workflow $Q = Q_s \cdot (Q_b \cdot Q_d)$, applied to $Q_d$ in place.
* **Strategy**: `bc_back` applies $Q_b$ (the bulge-chasing reflectors) to $Q_d$ with a register-resident sliding-window kernel — the reflectors compose to $Q_b^\top$ in forward order, so they are applied in reverse (upward slide) to get $Q_b$. Then `sbr_back` applies $Q_s$ via WY-block GEMMs $(I - W Y^\top)$ per panel. The SC'25 **reordered** scheme (build $Q_s \cdot Q_b$ while CPU D&C runs, then one GEMM $\cdot Q_d$) is a deferred performance layer; the current implementation is synchronous.


References: Wang et al., "Improving Tridiagonalization Performance on GPU Architectures"
(PPoPP'25); Wang et al., "Rethinking Back Transformation in 2-stage EVD" (SC'25).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80   # single GPU
cmake --build build
```

### Stage status

- [x] **1 DBBR** (full → band) — complete, tested (eigenvalues vs cuSOLVER), benchmarked
- [x] **2 Data repacking** (`bc_pack` / `dbbr_pack`) — complete
- [x] **3 Bulge chasing** (`bc_chase`) — complete, tested (spectrum vs cuSOLVER)
- [x] **4 D&C** (LAPACK `*stedc`, MKL) — functional on CPU; dominates wall time at large n, to be optimized
- [x] **5 Back-transform** (`bc_back` + `sbr_back`) — complete, tested (full residual + stage isolation)

The full solve is correct end-to-end (n=16000 fp64: relative residual ~2e-14, eigenvectors ~7e-14
vs cuSOLVER). GPU stages are competitive; CPU `*stedc` is the remaining wall-time bottleneck.

DBBR on A100 80GB (fp64, b=64, k=512): ~9.9 TFLOP/s at n=32k (~1.15× over single-blocked SBR;
the bigger DBBR win is a ≥49k phenomenon). Profile levers for later: per-block square companion
(`symm` ~34%), custom panel QR (cuSOLVER `geqrf` ~26%), custom `dbbr_syr2k` (~18%).

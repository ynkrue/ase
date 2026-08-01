#!/bin/bash
# Run the full GPU test suite under a single GPU allocation.
#
# The suite needs a CUDA device, so it runs via srun (never directly on the login node).
# GPU tests cannot run in GitHub Actions (no GPU runner), so this is the canonical local
# gate. Pass ctest flags through, e.g.:
#   ./scripts/test.sh                     # full suite
#   ./scripts/test.sh -R ase_test_solve   # one stage's suite
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ASE_BUILD_DIR:=build}"

# --mpi=none: this cluster's pmix plugin fails to init; the test binary is not MPI.
exec srun -N1 -n1 --mpi=none --gpus-per-task=1 \
  ctest --test-dir "$ASE_BUILD_DIR" --output-on-failure "$@"

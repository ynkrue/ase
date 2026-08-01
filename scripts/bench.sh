#!/bin/bash
# Run the benchmark under a single GPU allocation (mirrors scripts/test.sh).
#
#   ./scripts/bench.sh                          # default sizes
#   ./scripts/bench.sh 4096 8192 --iters 5 --host
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ASE_BUILD_DIR:=build}"

exec srun -N1 -n1 --mpi=none --gpus-per-task=1 "$ASE_BUILD_DIR/ase_bench" "$@"

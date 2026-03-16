#!/bin/bash
# Evaluate the performance of matrix multiplication algorithms
# 0: naive, 1: jk reordering, 2: tiling, 3: tiling + transposed, 4: tiling + transposed + simd

if [ $# -lt 1 ]; then
    echo "Usage: $0 <algorithm>"
    echo "  algorithm - 0: naive, 1: jk reordering, 2: tiling, 3: tiling + transposed, 4: tiling + transposed + simd"
    exit 1
fi
algo="$1"

echo "n ws_kib dur"
echo "-----------------------------"
for n in 16 32 64 128 256 512 1024 2048; do
    m="$n"
    dur=$(./matrix -n "$n" -m "$m" -a "$algo" | awk '{ print $2 }')
    # working set KiB: (A[n*m] + B[m*n] + C[n*n]) * sizeof(float)
    ws=$(( ( (n * m) + (m * n) + (n * n) ) * 4 / 1024 ))
    echo "$n" "$ws" "$dur"
done

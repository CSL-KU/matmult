#!/bin/bash
# Evaluate the performance of matrix multiplication algorithms
# 0: naive, 1: jk reordering, 2: tiling, 3: tiling + transposed, 4: tiling + transposed + simd

if [ $# -ne 2 ]; then
    echo "Usage: $0 <algorithm> <dtype>"
    echo "  algorithm - 0: naive, 1: jk reordering, 2: tiling, 3: tiling + transposed, 4: tiling + transposed + simd"
    echo "  dtype - fp32 or int8"
    exit 1
fi
algo=$1
dtype=$2
echo "Evaluating algorithm $algo with data type $dtype"
echo "n ws dur"
echo "-----------------------------"
for n in 16 32 64 128 256 512 1024 2048; do
    ./matrix -n $n -a $algo -t $dtype > tmp.txt
    dur=`cat tmp.txt | grep opt | awk '{ print $2 }'`
    ws=`expr $n \* $n \* 4 \* 3 / 1024`
    echo $n $ws $dur
done

# Matrix Multiplication Microbenchmark (`matmult`)

A C microbenchmark for comparing several matrix multiplication implementations, from a naive triple loop to cache-aware and SIMD-accelerated variants.

This project is useful for studying how loop ordering, cache behavior, transposition, and vector instructions affect dense GEMM-like workloads on modern CPUs.

## Features

- Single-binary benchmark (`matrix`) with selectable algorithm variants
- Configurable matrix shape for `(n,m) x (m,n)` multiplication (`-n`, `-m`)
- Deterministic random initialization (fixed seed) for reproducible checksums
- Simple timing output suitable for scripting
- Includes a helper evaluation script (`eval.sh`) for size sweeps

## Repository Layout

- `matrix.c`: benchmark implementation and algorithm variants
- `Makefile`: build configuration
- `eval.sh`: quick performance sweep by matrix size for a selected algorithm

## Build

### Prerequisites

- Linux
- GCC toolchain with C11 support (`gcc`, `make`)

### Compile

```bash
make
```

By default, the `Makefile` uses:

- `-Ofast`
- `-march=native`
- `-flto`
- `-std=c11`

To clean build artifacts:

```bash
make clean
```

## Usage

```bash
./matrix [-n n] [-m m] [-a algorithm]
```

- `-n`: A rows / B cols (default: `1024`)
- `-m`: shared dimension, A cols / B rows (default: `1024`)
- `-a`: algorithm selector

Algorithm IDs:

- `0`: naive `i-j-k`
- `1`: loop reorder `i-k-j`
- `2`: loop reorder + tiling (block size 256)
- `3`: transpose `B` then multiply
- `4`: transpose + SIMD path (AVX2/SSE/NEON depending on build target)
- `99`: run all algorithms in sequence

Show help:

```bash
./matrix -h
```

## Example

Run all variants for `(n,m) = (256,128)`:

```bash
./matrix -n 256 -m 128 -a 99
```

Typical output format:

```text
matmult_opt0  0.044723  chsum: -648.131751
matmult_opt1  0.004007  chsum: -648.131751
matmult_opt2  0.003787  chsum: -648.131751
matmult_opt3  0.016045  chsum: -648.131751
matmult_opt4  0.003884  chsum: -648.131646
```

Fields:

- first column: algorithm label
- second column: elapsed time in seconds
- third column: checksum of output matrix (`C`) for quick correctness sanity checks

## Batch Evaluation

`eval.sh` sweeps dimensions and prints a compact table.

```bash
./eval.sh <algorithm>
```

Example:

```bash
./eval.sh 1
```

Output columns:

- `n`: shared dimension, A cols / B rows (fixed if provided; otherwise `m=n`)
- `ws_kib`: approximate working-set size in KiB (`(n*m + m*n + n*n)*4/1024`)
- `dur`: measured duration in seconds

## Notes On Correctness And Comparisons

- Random inputs are generated with a fixed seed (`srand(292)`), so checksums are reproducible for the same build and architecture.
- Small checksum differences between SIMD and non-SIMD paths are expected due to floating-point accumulation order.

## Known Limitations

- Tiled and SIMD kernels perform best when dimensions align with vector/block widths.
- Algorithm `99` runs all variants sequentially, so runtime can be long for large matrices.
- SIMD behavior depends on target ISA selected by the compiler and CPU (`AVX2`, `SSE`, or `NEON`).

## Reproducibility Tips

- Run on an otherwise idle machine.
- Pin process to a core if needed using `taskset`.
- Repeat runs and compare medians rather than single samples.

## Acknowledgments

The implementation references optimization ideas discussed in:

- https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa
- https://www.dropbox.com/scl/fi/42b23nby5k5d09bpwd1cx/lec11.pdf?rlkey=e2ce7bs8ssgtb82isxgv4y7ij&dl=0


# matmult

This is a matrix multiplication microbenchmark. 

## Building

```
make            # builds C (matrix), C++ (matrix-cpp) 
```

## C vs C++ implementation

The original C implementation lives in `matrix.c` (build target `matrix`). A C++ version with identical algorithms plus minor robustness tweaks (correct block tiling bounds and RAII for aligned allocations) is in `matrix.cpp` (build target `matrix-cpp`). Both accept the same CLI flags:

```
./matrix-cpp -n <dimension> -a <algorithm>
```

Algorithms:

| id | name | notes |
|----|------|-------|
| 0  | naive | i-j-k triple loop |
| 1  | jk | i-k-j order for better A row locality |
| 2  | jk_tiling | adds simple 256x256 tiling (bounds fixed for edges in C++) |
| 3  | transposed | transposes B for contiguous row access during multiply |
| 4  | simd | transposed + SIMD (AVX2 / SSE / NEON) fallback to (3) |
| 99 | all | runs all algorithms sequentially |

Example:

```
./matrix-cpp -n 512 -a 99
```

Outputs timing and checksum per algorithm.

## A*B on M1
| dimension	| dense	    |
|-----------|-----------|
|10	        | 0.000012	|
|100	      | 0.000224	| 
|1000	      | 0.074843	| 
|2000	      | 0.484162	|
				
## A*B on xeon E5-2658			
| dimension | dense	    | 
|-----------|-----------|
| 10	      | 0.000008	|
| 100	      | 0.000717	|
| 1000	    | 0.219490	|
| 2000	    | 1.488919	|	

(units are in seconds).

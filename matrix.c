// References:
// - https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa
// - https://www.dropbox.com/scl/fi/42b23nby5k5d09bpwd1cx/lec11.pdf?rlkey=e2ce7bs8ssgtb82isxgv4y7ij&dl=0 
//
// how to compile with gcc:
// $ gcc -Ofast -march=native -flto -std=c11 -o matrix matrix.c

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE             /* See feature_test_macros(7) */
#endif

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
// #include <omp.h>

/* change dimension size as needed */
struct timeval tv; 
int dimension = 1024;
double start, end; /* time */

static int calc_matrix_bytes(int dimension, size_t *bytes)
{
    if (dimension <= 0) {
        return -1;
    }

    size_t n = (size_t)dimension;
    if (n > SIZE_MAX / n) {
        return -1;
    }

    size_t elems = n * n;
    if (elems > SIZE_MAX / sizeof(float)) {
        return -1;
    }

    *bytes = elems * sizeof(float);
    return 0;
}

double timestamp()
{
    double t;
    gettimeofday(&tv, NULL);
    t = tv.tv_sec + (tv.tv_usec/1000000.0);
    return t;
}

void init_data(float *A, float *B, float *C, int dimension)
{
    int i, j, k;
    srand(292);
    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            A[dimension*i+j] = (float)rand()/(float)(RAND_MAX) - 0.5;
            B[dimension*i+j] = (float)rand()/(float)(RAND_MAX) - 0.5;
            C[dimension*i+j] = 0.0;
        }
        // printf("%f %f\n", A[dimension*i+j], B[dimension*i+j]);
    }
}

double print_checksum(float *C, int dimention)
{
    double sum = 0.0;
    for(int i = 0; i < dimention; i++) {
        for(int j = 0; j < dimention; j++) {
            sum += C[i*dimention+j];
        }
    }
    return sum;
}

#define BENCH(func) \
    init_data(A, B, C, dimension); \
    start = timestamp(); \
    func; \
    end = timestamp(); \
    print_checksum(C, dimension); \
    printf("%.12s  %.6f  chsum: %.6f\n", #func, end-start, print_checksum(C, dimension));


// a naive matrix multiplication implementation. 
void matmult_opt0_naive(    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            for(int k = 0; k < dimension; k++) {
                C[dimension*i+j] += (A[dimension*i+k] * B[dimension*k+j]);
            }
        }
    }	
}

// matrix multiplication with jk order switch
void matmult_opt1_jk(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int k = 0; k < dimension; k++) {
            for(int j = 0; j < dimension; j++) {
                C[dimension*i+j] += (A[dimension*i+k] * B[dimension*k+j]);
            }
        }
    }	
}

// matrix multiplication with jk order switch and tiling
// Handles tail tiles when dimension is not a multiple of block size.
void matmult_opt2_jk_tiling(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int dimension)
{
    int i,j,k,ii,jj,kk;
    int bs = 256; // block size = 256*256*4 = 256KB

    for(i = 0; i < dimension; i+=bs) {
        int i_end = (i + bs < dimension) ? i + bs : dimension;
        for(k = 0; k < dimension; k+=bs) {
            int k_end = (k + bs < dimension) ? k + bs : dimension;
            for(j = 0; j < dimension; j+=bs) {
                int j_end = (j + bs < dimension) ? j + bs : dimension;
                for(ii = i; ii < i_end; ii++) {
                    for(kk = k; kk < k_end; kk++) {
                        for(jj = j; jj < j_end; jj++) {
                            C[dimension*ii+jj] += (A[dimension*ii+kk] * B[dimension*kk+jj]);
                        }
                    }
                }
            }
        }
    }
}   


// transpose matrix
void transpose_naive(float *src, float *dst, int src_row, int src_col)
// src: m(src_row) x n(src_col)  -> dst: n x m
{
    for (int i = 0; i < src_col; i++) {
        for (int j = 0; j < src_row; j++) {
            dst[i*src_row+j] = src[j*src_col+i];
        }
    }
}

// matrix multiplicaiton after transposed
void matmult_opt3_transposed(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int dimension)
{
    int i,j,k;
    size_t alloc_size;
    if (calc_matrix_bytes(dimension, &alloc_size) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return;
    }
    float* __restrict__ Bt = (float*)malloc(alloc_size);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    transpose_naive(B, Bt, dimension, dimension);

    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            for(k = 0; k < dimension; k++) {                            
                C[dimension*i+j] += (A[dimension*i+k] * Bt[dimension*j+k]);
            }
        }
    }
    free(Bt);
}



#ifdef __AVX2__
#include <immintrin.h> // AVX2 Intrinsics
// matrix multiplicaiton transposed with AVX2 SIMD
void matmult_opt4_transposed_simd(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int dimension) {

    size_t alloc_size;
    if (calc_matrix_bytes(dimension, &alloc_size) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return;
    }
    float *Bt = (float*)aligned_alloc(32, alloc_size); // 32-byte aligned allocation for AVX2
    if (!Bt) {
        fprintf(stderr, "Failed to allocate aligned memory\n");
        return;
    }
    transpose_naive(B, Bt, dimension, dimension);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            __m256 acc = _mm256_setzero_ps(); // Initialize accumulator to zero
            int k;
            // Process 8 elements at a time
            for (k = 0; k <= dimension - 8; k += 8) {
                __m256 a = _mm256_loadu_ps(A + i * dimension + k);
                __m256 b = _mm256_loadu_ps(Bt + j * dimension + k);
                __m256 mul = _mm256_mul_ps(a, b); // Multiply vectors
                acc = _mm256_add_ps(acc, mul); // Accumulate
            }

            // Horizontal sum of the 8 elements in acc
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float result = _mm_cvtss_f32(sum128);

            // Handle remaining elements (if dimension is not divisible by 8)
            for (; k < dimension; k++) {
                result += A[i * dimension + k] * Bt[j * dimension + k];
            }

            // Store the result in the output matrix
            C[i * dimension + j] = result;
        }
    }
    free(Bt);
}
#elif __SSE__
#include <emmintrin.h> // SSE2 Intrinsics
#include <smmintrin.h> // SSE4.2 Intrinsics

// matrix multiplicaiton transposed with SIMD
void matmult_opt4_transposed_simd(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int dimension) {

    size_t alloc_size;
    if (calc_matrix_bytes(dimension, &alloc_size) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return;
    }
    float *Bt = (float*)malloc(alloc_size);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    transpose_naive(B, Bt, dimension, dimension);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            float accumulators[4] = {0, 0, 0, 0};
            __m128 *acc = (__m128 *) accumulators;
            int k;
            for (k = 0; k <= dimension - 4; k += 4) {
                // fprintf(stderr, "[%d,%d,%d]\n", i, j, k);
                __m128 a = _mm_loadu_ps(A + i * dimension + k); // Load 4 values from matrixA
                __m128 b = _mm_loadu_ps(Bt + j * dimension + k); // Load 4 values from matrixB
                __m128 mul = _mm_mul_ps(a, b); // Multiply and accumulate using dot product
                *acc = _mm_add_ps(*acc, mul);
                // Repeat the above steps for the remaining elements of the current row and column
            }
            // Store the result in the output matrix
            float result = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
            for (; k < dimension; k++) {
                result += A[i * dimension + k] * Bt[j * dimension + k];
            }
            *(C + i * dimension + j) = result;
            // fprintf(stderr, "[%d,%d]=%.2f\n", i, j, result[i*dimension+j]);
        }
    }
    free(Bt);
}
#elif __ARM_NEON
#include <arm_neon.h>
// matrix multiplicaiton transposed with SIMD
void matmult_opt4_transposed_simd(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int dimension) {

    size_t alloc_size;
    if (calc_matrix_bytes(dimension, &alloc_size) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return;
    }
    float *Bt = (float*)malloc(alloc_size);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    transpose_naive(B, Bt, dimension, dimension);

    // matrix multiplication of A and B into C
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            float accumulators[4] = {0, 0, 0, 0};
            float32x4_t *acc = (float32x4_t *) accumulators;
            int k;
            for (k = 0; k <= dimension - 4; k += 4) {
                // fprintf(stderr, "[%d,%d,%d]\n", i, j, k);
                float32x4_t a = vld1q_f32(A + i * dimension + k); // Load 4 values from matrixA
                float32x4_t b = vld1q_f32(Bt + j * dimension + k); // Load 4 values from matrixB
                float32x4_t mul = vmulq_f32(a, b); // Multiply and accumulate using dot product
                *acc = vaddq_f32(*acc, mul);
                // Repeat the above steps for the remaining elements of the current row and column
            }
            // Store the result in the output matrix
            float result = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
            for (; k < dimension; k++) {
                result += A[i * dimension + k] * Bt[j * dimension + k];
            }
            *(C + i * dimension + j) = result;
            // fprintf(stderr, "[%d,%d]=%.2f\n", i, j, result[i*dimension+j]);
        }
    }
    free(Bt);
}
#endif // AVX2 __SSE__ __ARM_NEON


int main(int argc, char *argv[])
{
    float *A, *B, *Bt, *C;
    unsigned finish = 0;
    int i, j, k;
    
    int opt;
    int algo = 99;
    
    /*
     * get command line options 
     */
    while ((opt = getopt(argc, argv, "m:n:a:h")) != -1) {
        switch (opt) {
        case 'n':
        {
            long parsed = strtol(optarg, NULL, 0);
            if (parsed <= 0 || parsed > INT_MAX) {
                fprintf(stderr, "Invalid dimension: %s\n", optarg);
                return EXIT_FAILURE;
            }
            dimension = (int)parsed;
            break;
        }
        case 'a':
            algo = strtol(optarg, NULL, 0);
            break;
        case 'h':
        default: /* '?' */
            printf("Usage: %s [-n dimension] [-a algorithm]\n", argv[0]);
            printf("  -n dimension: matrix dimension (default: 1024)\n");
            printf("  -a algorithm: 0: naive, 1: jk, 2: jk_tiling, 3: transposed, 4: simd\n");
            exit(EXIT_SUCCESS);
        }

    }

#if 0
    // set CPU priority to high
    if (setpriority(PRIO_PROCESS, 0, -20) < 0) {
        perror("setpriority");
    }
#endif

    // printf("dimension: %d, algorithm: %d ws: %.1f\n", dimension, algo,
    //        (float)dimension*dimension*sizeof(float)*3/1024);

    size_t alloc_size;
    if (calc_matrix_bytes(dimension, &alloc_size) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return EXIT_FAILURE;
    }
    // Use aligned allocation for better SIMD performance
    A = (float*)aligned_alloc(32, alloc_size);
    B = (float*)aligned_alloc(32, alloc_size);
    C = (float*)aligned_alloc(32, alloc_size);

    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate aligned memory for matrices\n");
        exit(EXIT_FAILURE);
    }

    memset(A, 0, alloc_size);
    memset(B, 0, alloc_size);
    memset(C, 0, alloc_size);
    
    // do matrix multiplication

    switch(algo) {
    case 0:
        BENCH(matmult_opt0_naive(A, B, C, dimension))
        break;
    case 1:
        BENCH(matmult_opt1_jk(A, B, C, dimension))
        break;
    case 2:
        BENCH(matmult_opt2_jk_tiling(A, B, C, dimension))
        break;
    case 3:
        BENCH(matmult_opt3_transposed(A, B, C, dimension))
        break;
    case 4:
        BENCH(matmult_opt4_transposed_simd(A, B, C, dimension))
        break;
    case 99:
        BENCH(matmult_opt0_naive(A, B, C, dimension))
        BENCH(matmult_opt1_jk(A, B, C, dimension))
        BENCH(matmult_opt2_jk_tiling(A, B, C, dimension))
        BENCH(matmult_opt3_transposed(A, B, C, dimension))
        BENCH(matmult_opt4_transposed_simd(A, B, C, dimension))
        break;
    default:
        printf("invalid algorithm\n");
        break;
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}

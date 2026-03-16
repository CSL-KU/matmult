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
int rows_n = 1024;
int inner_m = 1024;
double start, end; /* time */

static int calc_matrix_bytes_2d(int rows, int cols, size_t *bytes)
{
    if (rows <= 0 || cols <= 0) {
        return -1;
    }

    size_t r = (size_t)rows;
    size_t c = (size_t)cols;
    if (r > SIZE_MAX / c) {
        return -1;
    }

    size_t elems = r * c;
    if (elems > SIZE_MAX / sizeof(float)) {
        return -1;
    }

    *bytes = elems * sizeof(float);
    return 0;
}

static float *alloc_matrix_aligned(int rows, int cols)
{
    size_t alloc_size;
    if (calc_matrix_bytes_2d(rows, cols, &alloc_size) != 0) {
        return NULL;
    }

    void *ptr = NULL;
    if (posix_memalign(&ptr, 32, alloc_size) != 0) {
        return NULL;
    }

    return (float *)ptr;
}

double timestamp()
{
    double t;
    gettimeofday(&tv, NULL);
    t = tv.tv_sec + (tv.tv_usec/1000000.0);
    return t;
}

void init_data(float *A, float *B, float *C, int n, int m)
{
    srand(292);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[m * i + j] = (float)rand()/(float)(RAND_MAX) - 0.5f;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[n * i + j] = (float)rand()/(float)(RAND_MAX) - 0.5f;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[n * i + j] = 0.0f;
        }
    }
}

double print_checksum(float *C, int rows, int cols)
{
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += C[i * cols + j];
        }
    }
    return sum;
}

#define BENCH(func) \
    init_data(A, B, C, rows_n, inner_m); \
    start = timestamp(); \
    func; \
    end = timestamp(); \
    print_checksum(C, rows_n, rows_n); \
    printf("%.12s  %.6f  chsum: %.6f\n", #func, end-start, print_checksum(C, rows_n, rows_n));


// a naive matrix multiplication implementation. 
void matmult_opt0_naive(float *A, float *B, float *C, int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < m; k++) {
                C[n * i + j] += (A[m * i + k] * B[n * k + j]);
            }
        }
    }
}

// matrix multiplication with jk order switch
void matmult_opt1_jk(float *A, float *B, float *C, int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < n; j++) {
                C[n * i + j] += (A[m * i + k] * B[n * k + j]);
            }
        }
    }
}

// matrix multiplication with jk order switch and tiling
// Handles tail tiles when dimension is not a multiple of block size.
void matmult_opt2_jk_tiling(float *A, float *B, float *C, int n, int m)
{
    int i,j,k,ii,jj,kk;
    int bs = 256; // block size = 256*256*4 = 256KB

    for (i = 0; i < n; i += bs) {
        int i_end = (i + bs < n) ? i + bs : n;
        for (k = 0; k < m; k += bs) {
            int k_end = (k + bs < m) ? k + bs : m;
            for (j = 0; j < n; j += bs) {
                int j_end = (j + bs < n) ? j + bs : n;
                for(ii = i; ii < i_end; ii++) {
                    for(kk = k; kk < k_end; kk++) {
                        for(jj = j; jj < j_end; jj++) {
                            C[n * ii + jj] += (A[m * ii + kk] * B[n * kk + jj]);
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
void matmult_opt3_transposed(float *A, float *B, float *C, int n, int m)
{
    int i,j,k;
    float *Bt = alloc_matrix_aligned(n, m);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    transpose_naive(B, Bt, m, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < m; k++) {
                C[n * i + j] += (A[m * i + k] * Bt[m * j + k]);
            }
        }
    }
    free(Bt);
}



#ifdef __AVX2__
#include <immintrin.h> // AVX2 Intrinsics
// matrix multiplicaiton transposed with AVX2 SIMD
void matmult_opt4_transposed_simd(float* A, float* B, float* C, int n, int m) {

    float *Bt = alloc_matrix_aligned(n, m);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate aligned memory\n");
        return;
    }
    transpose_naive(B, Bt, m, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __m256 acc = _mm256_setzero_ps(); // Initialize accumulator to zero
            int k;
            // Process 8 elements at a time
            for (k = 0; k <= m - 8; k += 8) {
                __m256 a = _mm256_loadu_ps(A + i * m + k);
                __m256 b = _mm256_loadu_ps(Bt + j * m + k);
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
            for (; k < m; k++) {
                result += A[i * m + k] * Bt[j * m + k];
            }

            // Store the result in the output matrix
            C[i * n + j] = result;
        }
    }
    free(Bt);
}
#elif __SSE__
#include <emmintrin.h> // SSE2 Intrinsics
#include <smmintrin.h> // SSE4.2 Intrinsics

// matrix multiplicaiton transposed with SIMD
void matmult_opt4_transposed_simd(float* A, float* B, float* C, int n, int m) {

    float *Bt = alloc_matrix_aligned(n, m);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    transpose_naive(B, Bt, m, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float accumulators[4] = {0, 0, 0, 0};
            __m128 *acc = (__m128 *) accumulators;
            int k;
            for (k = 0; k <= m - 4; k += 4) {
                // fprintf(stderr, "[%d,%d,%d]\n", i, j, k);
                __m128 a = _mm_loadu_ps(A + i * m + k); // Load 4 values from matrixA
                __m128 b = _mm_loadu_ps(Bt + j * m + k); // Load 4 values from matrixB
                __m128 mul = _mm_mul_ps(a, b); // Multiply and accumulate using dot product
                *acc = _mm_add_ps(*acc, mul);
                // Repeat the above steps for the remaining elements of the current row and column
            }
            // Store the result in the output matrix
            float result = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
            for (; k < m; k++) {
                result += A[i * m + k] * Bt[j * m + k];
            }
            *(C + i * n + j) = result;
            // fprintf(stderr, "[%d,%d]=%.2f\n", i, j, result[i*dimension+j]);
        }
    }
    free(Bt);
}
#elif __ARM_NEON
#include <arm_neon.h>
// matrix multiplicaiton transposed with SIMD
void matmult_opt4_transposed_simd(float* A, float* B, float* C, int n, int m) {

    float *Bt = alloc_matrix_aligned(n, m);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    transpose_naive(B, Bt, m, n);

    // matrix multiplication of A and B into C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float accumulators[4] = {0, 0, 0, 0};
            float32x4_t *acc = (float32x4_t *) accumulators;
            int k;
            for (k = 0; k <= m - 4; k += 4) {
                // fprintf(stderr, "[%d,%d,%d]\n", i, j, k);
                float32x4_t a = vld1q_f32(A + i * m + k); // Load 4 values from matrixA
                float32x4_t b = vld1q_f32(Bt + j * m + k); // Load 4 values from matrixB
                float32x4_t mul = vmulq_f32(a, b); // Multiply and accumulate using dot product
                *acc = vaddq_f32(*acc, mul);
                // Repeat the above steps for the remaining elements of the current row and column
            }
            // Store the result in the output matrix
            float result = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
            for (; k < m; k++) {
                result += A[i * m + k] * Bt[j * m + k];
            }
            *(C + i * n + j) = result;
            // fprintf(stderr, "[%d,%d]=%.2f\n", i, j, result[i*dimension+j]);
        }
    }
    free(Bt);
}
#endif // AVX2 __SSE__ __ARM_NEON


int main(int argc, char *argv[])
{
    float *A, *B, *C;
    
    int opt;
    int algo = 99;
    
    /*
     * get command line options 
     */
    while ((opt = getopt(argc, argv, "m:n:a:h")) != -1) {
        switch (opt) {
        case 'm':
        {
            long parsed = strtol(optarg, NULL, 0);
            if (parsed <= 0 || parsed > INT_MAX) {
                fprintf(stderr, "Invalid inner dimension m: %s\n", optarg);
                return EXIT_FAILURE;
            }
            inner_m = (int)parsed;
            break;
        }
        case 'n':
        {
            long parsed = strtol(optarg, NULL, 0);
            if (parsed <= 0 || parsed > INT_MAX) {
                fprintf(stderr, "Invalid output dimension n: %s\n", optarg);
                return EXIT_FAILURE;
            }
            rows_n = (int)parsed;
            break;
        }
        case 'a':
            algo = strtol(optarg, NULL, 0);
            break;
        case 'h':
        default: /* '?' */
            printf("Usage: %s [-n n] [-m m] [-a algorithm]\n", argv[0]);
            printf("  -n n: A rows / B cols (default: 1024)\n");
            printf("  -m m: shared dimension A cols / B rows (default: 1024)\n");
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

    size_t a_bytes, b_bytes, c_bytes;
    if (calc_matrix_bytes_2d(rows_n, inner_m, &a_bytes) != 0 ||
        calc_matrix_bytes_2d(inner_m, rows_n, &b_bytes) != 0 ||
        calc_matrix_bytes_2d(rows_n, rows_n, &c_bytes) != 0) {
        fprintf(stderr, "Invalid matrix dimensions for allocation\n");
        return EXIT_FAILURE;
    }

    A = alloc_matrix_aligned(rows_n, inner_m);
    B = alloc_matrix_aligned(inner_m, rows_n);
    C = alloc_matrix_aligned(rows_n, rows_n);

    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate aligned memory for matrices\n");
        exit(EXIT_FAILURE);
    }
    
    // do matrix multiplication

    switch(algo) {
    case 0:
        BENCH(matmult_opt0_naive(A, B, C, rows_n, inner_m))
        break;
    case 1:
        BENCH(matmult_opt1_jk(A, B, C, rows_n, inner_m))
        break;
    case 2:
        BENCH(matmult_opt2_jk_tiling(A, B, C, rows_n, inner_m))
        break;
    case 3:
        BENCH(matmult_opt3_transposed(A, B, C, rows_n, inner_m))
        break;
    case 4:
        BENCH(matmult_opt4_transposed_simd(A, B, C, rows_n, inner_m))
        break;
    case 99:
        BENCH(matmult_opt0_naive(A, B, C, rows_n, inner_m))
        BENCH(matmult_opt1_jk(A, B, C, rows_n, inner_m))
        BENCH(matmult_opt2_jk_tiling(A, B, C, rows_n, inner_m))
        BENCH(matmult_opt3_transposed(A, B, C, rows_n, inner_m))
        BENCH(matmult_opt4_transposed_simd(A, B, C, rows_n, inner_m))
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

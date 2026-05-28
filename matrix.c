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

static int calc_matrix_bytes_with_size(int dimension, size_t elem_size, size_t *bytes)
{
    if (dimension <= 0) {
        return -1;
    }

    size_t n = (size_t)dimension;
    if (n > SIZE_MAX / n) {
        return -1;
    }

    size_t elems = n * n;
    if (elems > SIZE_MAX / elem_size) {
        return -1;
    }

    *bytes = elems * elem_size;
    return 0;
}

double timestamp()
{
    double t;
    gettimeofday(&tv, NULL);
    t = tv.tv_sec + (tv.tv_usec/1000000.0);
    return t;
}

void init_data(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    int i, j;
    srand(292);
    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            /* small integer values to keep products reasonable */
            A[dimension*i+j] = (int8_t)((rand() % 101) - 50);
            B[dimension*i+j] = (int8_t)((rand() % 101) - 50);
            C[dimension*i+j] = 0;
        }
    }
}

long long print_checksum(int32_t *C, int dimention)
{
    long long sum = 0;
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
    printf("%.12s  %.6f  chsum: %lld\n", #func, end-start, print_checksum(C, dimension));


// a naive matrix multiplication implementation. 
void matmult_opt0_naive(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            for(int k = 0; k < dimension; k++) {
                C[dimension*i+j] += (int32_t)A[dimension*i+k] * (int32_t)B[dimension*k+j];
            }
        }
    }	
}

// matrix multiplication with jk order switch
void matmult_opt1_jk(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int k = 0; k < dimension; k++) {
            for(int j = 0; j < dimension; j++) {
                C[dimension*i+j] += (int32_t)A[dimension*i+k] * (int32_t)B[dimension*k+j];
            }
        }
    }	
}

// matrix multiplication with jk order switch and tiling
// Handles tail tiles when dimension is not a multiple of block size.
void matmult_opt2_jk_tiling(int8_t *A, int8_t *B, int32_t *C, int dimension)
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
                            C[dimension*ii+jj] += (int32_t)A[dimension*ii+kk] * (int32_t)B[dimension*kk+jj];
                        }
                    }
                }
            }
        }
    }
}   


// transpose matrix
void transpose_naive(int8_t *src, int8_t *dst, int src_row, int src_col)
// src: m(src_row) x n(src_col)  -> dst: n x m
{
    for (int i = 0; i < src_col; i++) {
        for (int j = 0; j < src_row; j++) {
            dst[i*src_row+j] = src[j*src_col+i];
        }
    }
}

// matrix multiplicaiton after transposed
void matmult_opt3_transposed(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    int i,j,k;
    size_t alloc_size;
    if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return;
    }
    int8_t *Bt = (int8_t*)malloc(alloc_size);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }
    transpose_naive(B, Bt, dimension, dimension);

    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            for(k = 0; k < dimension; k++) {                            
                C[dimension*i+j] += (int32_t)A[dimension*i+k] * (int32_t)Bt[dimension*j+k];
            }
        }
    }
    free(Bt);
}



// Integer transposed multiplication (portable fallback for int8 inputs)
void matmult_opt4_transposed_simd(int8_t* A, int8_t* B, int32_t* C, int dimension) {
    size_t alloc_size;
    if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return;
    }
    int8_t *Bt = (int8_t*)aligned_alloc(32, alloc_size);
    if (!Bt) {
        fprintf(stderr, "Failed to allocate aligned memory\n");
        return;
    }
    transpose_naive(B, Bt, dimension, dimension);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            long long acc = 0;
            for (int k = 0; k < dimension; k++) {
                acc += (long long)A[i * dimension + k] * (long long)Bt[j * dimension + k];
            }
            C[i * dimension + j] = (int32_t)acc;
        }
    }
    free(Bt);
}


int main(int argc, char *argv[])
{
    int8_t *A, *B, *Bt;
    int32_t *C;
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

    size_t alloc_size8, alloc_size32;
    if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size8) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return EXIT_FAILURE;
    }
    if (calc_matrix_bytes_with_size(dimension, sizeof(int32_t), &alloc_size32) != 0) {
        fprintf(stderr, "Invalid dimension for allocation\n");
        return EXIT_FAILURE;
    }
    // Use aligned allocation for better SIMD performance
    A = (int8_t*)aligned_alloc(32, alloc_size8);
    B = (int8_t*)aligned_alloc(32, alloc_size8);
    C = (int32_t*)aligned_alloc(32, alloc_size32);

    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate aligned memory for matrices\n");
        exit(EXIT_FAILURE);
    }

    memset(A, 0, alloc_size8);
    memset(B, 0, alloc_size8);
    memset(C, 0, alloc_size32);
    
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

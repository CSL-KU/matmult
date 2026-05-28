// Small integer matrix multiply benchmark
// Supports int8 inputs and int32 outputs with SIMD kernels.

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>

struct timeval tv;
int dimension = 1024;

double timestamp() { gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec/1000000.0; }

static int calc_matrix_bytes_with_size(int dimension, size_t elem_size, size_t *bytes)
{
    if (dimension <= 0) return -1;
    size_t n = (size_t)dimension;
    if (n > SIZE_MAX / n) return -1;
    size_t elems = n * n;
    if (elems > SIZE_MAX / elem_size) return -1;
    *bytes = elems * elem_size;
    return 0;
}

void init_data(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    srand(292);
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++) {
            A[dimension*i+j] = (int8_t)((rand() % 101) - 50);
            B[dimension*i+j] = (int8_t)((rand() % 101) - 50);
            C[dimension*i+j] = 0;
        }
}

long long print_checksum(int32_t *C, int dimention)
{
    long long sum = 0;
    for (int i = 0; i < dimention; i++)
        for (int j = 0; j < dimention; j++)
            sum += C[i*dimention+j];
    return sum;
}

#define BENCH(func) \
    init_data(A, B, C, dimension); \
    { double t0 = timestamp(); func; double t1 = timestamp(); printf("%.12s  %.6f  chsum: %lld\n", #func, t1-t0, print_checksum(C, dimension)); }

void transpose_naive(int8_t *src, int8_t *dst, int src_row, int src_col)
{
    for (int i = 0; i < src_col; i++)
        for (int j = 0; j < src_row; j++)
            dst[i*src_row+j] = src[j*src_col+i];
}

void matmult_opt0_naive(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            for (int k = 0; k < dimension; k++)
                C[dimension*i+j] += (int32_t)A[dimension*i+k] * (int32_t)B[dimension*k+j];
}

void matmult_opt1_jk(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    for (int i = 0; i < dimension; i++)
        for (int k = 0; k < dimension; k++)
            for (int j = 0; j < dimension; j++)
                C[dimension*i+j] += (int32_t)A[dimension*i+k] * (int32_t)B[dimension*k+j];
}

void matmult_opt2_jk_tiling(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    int bs = 256;
    for (int i = 0; i < dimension; i += bs) {
        int i_end = (i + bs < dimension) ? i + bs : dimension;
        for (int k = 0; k < dimension; k += bs) {
            int k_end = (k + bs < dimension) ? k + bs : dimension;
            for (int j = 0; j < dimension; j += bs) {
                int j_end = (j + bs < dimension) ? j + bs : dimension;
                for (int ii = i; ii < i_end; ii++)
                    for (int kk = k; kk < k_end; kk++)
                        for (int jj = j; jj < j_end; jj++)
                            C[dimension*ii+jj] += (int32_t)A[dimension*ii+kk] * (int32_t)B[dimension*kk+jj];
            }
        }
    }
}

void matmult_opt3_transposed(int8_t *A, int8_t *B, int32_t *C, int dimension)
{
    size_t alloc_size;
    if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return; }
    int8_t *Bt = (int8_t*)malloc(alloc_size);
    if (!Bt) { fprintf(stderr, "Failed to allocate memory\n"); return; }
    transpose_naive(B, Bt, dimension, dimension);
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            for (int k = 0; k < dimension; k++)
                C[dimension*i+j] += (int32_t)A[dimension*i+k] * (int32_t)Bt[dimension*j+k];
    free(Bt);
}

// SIMD version with NEON dotprod option
#if defined(__AVX2__)
#include <immintrin.h>
void matmult_opt4_transposed_simd(int8_t* A, int8_t* B, int32_t* C, int dimension) {
    size_t alloc_size; if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return; }
    int8_t *Bt = (int8_t*)aligned_alloc(32, alloc_size); if (!Bt) { fprintf(stderr, "Failed to allocate aligned memory\n"); return; }
    transpose_naive(B, Bt, dimension, dimension);
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            __m256i acc_lo = _mm256_setzero_si256(); __m256i acc_hi = _mm256_setzero_si256();
            int k;
            for (k = 0; k <= dimension - 16; k += 16) {
                __m128i a128 = _mm_loadu_si128((const __m128i*)(A + i * dimension + k));
                __m128i b128 = _mm_loadu_si128((const __m128i*)(Bt + j * dimension + k));
                __m256i a16 = _mm256_cvtepi8_epi16(a128);
                __m256i b16 = _mm256_cvtepi8_epi16(b128);
                __m128i a16_lo = _mm256_castsi256_si128(a16); __m128i a16_hi = _mm256_extracti128_si256(a16, 1);
                __m128i b16_lo = _mm256_castsi256_si128(b16); __m128i b16_hi = _mm256_extracti128_si256(b16, 1);
                __m256i a32_lo = _mm256_cvtepi16_epi32(a16_lo); __m256i a32_hi = _mm256_cvtepi16_epi32(a16_hi);
                __m256i b32_lo = _mm256_cvtepi16_epi32(b16_lo); __m256i b32_hi = _mm256_cvtepi16_epi32(b16_hi);
                __m256i mul_lo = _mm256_mullo_epi32(a32_lo, b32_lo); __m256i mul_hi = _mm256_mullo_epi32(a32_hi, b32_hi);
                acc_lo = _mm256_add_epi32(acc_lo, mul_lo); acc_hi = _mm256_add_epi32(acc_hi, mul_hi);
            }
            long long sum = 0; int32_t tmp[8]; _mm256_storeu_si256((__m256i*)tmp, acc_lo); for (int t = 0; t < 8; t++) sum += tmp[t]; _mm256_storeu_si256((__m256i*)tmp, acc_hi); for (int t = 0; t < 8; t++) sum += tmp[t];
            for (; k < dimension; k++) sum += (int32_t)A[i * dimension + k] * (int32_t)Bt[j * dimension + k];
            C[i *dimension + j] = (int32_t)sum;
        }
    }
    free(Bt);
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
void matmult_opt4_transposed_simd(int8_t* A, int8_t* B, int32_t* C, int dimension) {
    size_t alloc_size; if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return; }
    int8_t *Bt = (int8_t*)malloc(alloc_size); if (!Bt) { fprintf(stderr, "Failed to allocate memory\n"); return; }
    transpose_naive(B, Bt, dimension, dimension);

#if defined(__ARM_FEATURE_DOTPROD)
    // Use vdot when available
    for (int i = 0; i < dimension; i++) {
        int j;
        for (j = 0; j <= dimension - 4; j += 4) {
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0), acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            int k;
            for (k = 0; k <= dimension - 16; k += 16) {
                int8x16_t a_vec = vld1q_s8(A + i * dimension + k);
                int8x16_t b0_vec = vld1q_s8(Bt + (j + 0) * dimension + k);
                int8x16_t b1_vec = vld1q_s8(Bt + (j + 1) * dimension + k);
                int8x16_t b2_vec = vld1q_s8(Bt + (j + 2) * dimension + k);
                int8x16_t b3_vec = vld1q_s8(Bt + (j + 3) * dimension + k);
                acc0 = vdotq_s32(acc0, a_vec, b0_vec);
                acc1 = vdotq_s32(acc1, a_vec, b1_vec);
                acc2 = vdotq_s32(acc2, a_vec, b2_vec);
                acc3 = vdotq_s32(acc3, a_vec, b3_vec);
            }
            int64_t sum0 = (int64_t)vaddvq_s32(acc0); int64_t sum1 = (int64_t)vaddvq_s32(acc1);
            int64_t sum2 = (int64_t)vaddvq_s32(acc2); int64_t sum3 = (int64_t)vaddvq_s32(acc3);
            for (; k < dimension; k++) {
                int32_t a = (int32_t)A[i * dimension + k];
                sum0 += a * (int32_t)Bt[(j + 0) * dimension + k];
                sum1 += a * (int32_t)Bt[(j + 1) * dimension + k];
                sum2 += a * (int32_t)Bt[(j + 2) * dimension + k];
                sum3 += a * (int32_t)Bt[(j + 3) * dimension + k];
            }
            C[i * dimension + (j + 0)] = (int32_t)sum0; C[i * dimension + (j + 1)] = (int32_t)sum1;
            C[i * dimension + (j + 2)] = (int32_t)sum2; C[i * dimension + (j + 3)] = (int32_t)sum3;
        }
        for (; j < dimension; j++) {
            long long acc = 0; int k;
            for (k = 0; k <= dimension - 16; k += 16) {
                int8x16_t a_vec = vld1q_s8(A + i * dimension + k);
                int8x16_t b_vec = vld1q_s8(Bt + j * dimension + k);
                int32x4_t tmp = vdotq_s32(vdupq_n_s32(0), a_vec, b_vec);
                acc += (long long)vaddvq_s32(tmp);
            }
            for (; k < dimension; k++) acc += (int32_t)A[i * dimension + k] * (int32_t)Bt[j * dimension + k];
            C[i * dimension + j] = (int32_t)acc;
        }
    }
#else
    // NEON fallback: 4-column blocked implementation
    for (int i = 0; i < dimension; i++) {
        int j;
        for (j = 0; j <= dimension - 4; j += 4) {
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0), acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            int k = 0;
            for (; k <= dimension - 16; k += 16) {
                int8x16_t a_vec = vld1q_s8(A + i * dimension + k);
                int16x8_t a_lo = vmovl_s8(vget_low_s8(a_vec)); int16x8_t a_hi = vmovl_s8(vget_high_s8(a_vec));
                int32x4_t a0 = vmovl_s16(vget_low_s16(a_lo)); int32x4_t a1 = vmovl_s16(vget_high_s16(a_lo));
                int32x4_t a2 = vmovl_s16(vget_low_s16(a_hi)); int32x4_t a3 = vmovl_s16(vget_high_s16(a_hi));
                int8x16_t b0_vec = vld1q_s8(Bt + (j + 0) * dimension + k);
                int8x16_t b1_vec = vld1q_s8(Bt + (j + 1) * dimension + k);
                int8x16_t b2_vec = vld1q_s8(Bt + (j + 2) * dimension + k);
                int8x16_t b3_vec = vld1q_s8(Bt + (j + 3) * dimension + k);
                int16x8_t b0_lo = vmovl_s8(vget_low_s8(b0_vec)); int16x8_t b0_hi = vmovl_s8(vget_high_s8(b0_vec));
                int16x8_t b1_lo = vmovl_s8(vget_low_s8(b1_vec)); int16x8_t b1_hi = vmovl_s8(vget_high_s8(b1_vec));
                int16x8_t b2_lo = vmovl_s8(vget_low_s8(b2_vec)); int16x8_t b2_hi = vmovl_s8(vget_high_s8(b2_vec));
                int16x8_t b3_lo = vmovl_s8(vget_low_s8(b3_vec)); int16x8_t b3_hi = vmovl_s8(vget_high_s8(b3_vec));
                int32x4_t b0_0 = vmovl_s16(vget_low_s16(b0_lo)); int32x4_t b0_1 = vmovl_s16(vget_high_s16(b0_lo));
                int32x4_t b0_2 = vmovl_s16(vget_low_s16(b0_hi)); int32x4_t b0_3 = vmovl_s16(vget_high_s16(b0_hi));
                int32x4_t b1_0 = vmovl_s16(vget_low_s16(b1_lo)); int32x4_t b1_1 = vmovl_s16(vget_high_s16(b1_lo));
                int32x4_t b1_2 = vmovl_s16(vget_low_s16(b1_hi)); int32x4_t b1_3 = vmovl_s16(vget_high_s16(b1_hi));
                int32x4_t b2_0 = vmovl_s16(vget_low_s16(b2_lo)); int32x4_t b2_1 = vmovl_s16(vget_high_s16(b2_lo));
                int32x4_t b2_2 = vmovl_s16(vget_low_s16(b2_hi)); int32x4_t b2_3 = vmovl_s16(vget_high_s16(b2_hi));
                int32x4_t b3_0 = vmovl_s16(vget_low_s16(b3_lo)); int32x4_t b3_1 = vmovl_s16(vget_high_s16(b3_lo));
                int32x4_t b3_2 = vmovl_s16(vget_low_s16(b3_hi)); int32x4_t b3_3 = vmovl_s16(vget_high_s16(b3_hi));
                acc0 = vaddq_s32(acc0, vmulq_s32(a0, b0_0)); acc0 = vaddq_s32(acc0, vmulq_s32(a1, b0_1));
                acc0 = vaddq_s32(acc0, vmulq_s32(a2, b0_2)); acc0 = vaddq_s32(acc0, vmulq_s32(a3, b0_3));
                acc1 = vaddq_s32(acc1, vmulq_s32(a0, b1_0)); acc1 = vaddq_s32(acc1, vmulq_s32(a1, b1_1));
                acc1 = vaddq_s32(acc1, vmulq_s32(a2, b1_2)); acc1 = vaddq_s32(acc1, vmulq_s32(a3, b1_3));
                acc2 = vaddq_s32(acc2, vmulq_s32(a0, b2_0)); acc2 = vaddq_s32(acc2, vmulq_s32(a1, b2_1));
                acc2 = vaddq_s32(acc2, vmulq_s32(a2, b2_2)); acc2 = vaddq_s32(acc2, vmulq_s32(a3, b2_3));
                acc3 = vaddq_s32(acc3, vmulq_s32(a0, b3_0)); acc3 = vaddq_s32(acc3, vmulq_s32(a1, b3_1));
                acc3 = vaddq_s32(acc3, vmulq_s32(a2, b3_2)); acc3 = vaddq_s32(acc3, vmulq_s32(a3, b3_3));
            }
            int64_t sum0 = (int64_t)vaddvq_s32(acc0); int64_t sum1 = (int64_t)vaddvq_s32(acc1);
            int64_t sum2 = (int64_t)vaddvq_s32(acc2); int64_t sum3 = (int64_t)vaddvq_s32(acc3);
            for (; k < dimension; k++) {
                int32_t a = (int32_t)A[i * dimension + k];
                sum0 += a * (int32_t)Bt[(j + 0) * dimension + k];
                sum1 += a * (int32_t)Bt[(j + 1) * dimension + k];
                sum2 += a * (int32_t)Bt[(j + 2) * dimension + k];
                sum3 += a * (int32_t)Bt[(j + 3) * dimension + k];
            }
            C[i * dimension + (j + 0)] = (int32_t)sum0; C[i * dimension + (j + 1)] = (int32_t)sum1;
            C[i * dimension + (j + 2)] = (int32_t)sum2; C[i * dimension + (j + 3)] = (int32_t)sum3;
        }
        for (; j < dimension; j++) {
            long long acc = 0; int k = 0;
            for (; k <= dimension - 16; k += 16) {
                int8x16_t a_vec = vld1q_s8(A + i * dimension + k);
                int8x16_t b_vec = vld1q_s8(Bt + j * dimension + k);
                int16x8_t a_lo = vmovl_s8(vget_low_s8(a_vec)); int16x8_t a_hi = vmovl_s8(vget_high_s8(a_vec));
                int16x8_t b_lo = vmovl_s8(vget_low_s8(b_vec)); int16x8_t b_hi = vmovl_s8(vget_high_s8(b_vec));
                int32x4_t a0 = vmovl_s16(vget_low_s16(a_lo)); int32x4_t a1 = vmovl_s16(vget_high_s16(a_lo));
                int32x4_t a2 = vmovl_s16(vget_low_s16(a_hi)); int32x4_t a3 = vmovl_s16(vget_high_s16(a_hi));
                int32x4_t b0 = vmovl_s16(vget_low_s16(b_lo)); int32x4_t b1 = vmovl_s16(vget_high_s16(b_lo));
                int32x4_t b2 = vmovl_s16(vget_low_s16(b_hi)); int32x4_t b3 = vmovl_s16(vget_high_s16(b_hi));
                acc += (long long)vaddvq_s32(vmulq_s32(a0,b0)); acc += (long long)vaddvq_s32(vmulq_s32(a1,b1));
                acc += (long long)vaddvq_s32(vmulq_s32(a2,b2)); acc += (long long)vaddvq_s32(vmulq_s32(a3,b3));
            }
            for (; k < dimension; k++) acc += (int32_t)A[i * dimension + k] * (int32_t)Bt[j * dimension + k];
            C[i * dimension + j] = (int32_t)acc;
        }
    }
#endif
    free(Bt);
}
#else
// Scalar fallback
void matmult_opt4_transposed_simd(int8_t* A, int8_t* B, int32_t* C, int dimension) {
    size_t alloc_size; if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return; }
    int8_t *Bt = (int8_t*)malloc(alloc_size); if (!Bt) { fprintf(stderr, "Failed to allocate memory\n"); return; }
    transpose_naive(B, Bt, dimension, dimension);
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++) {
            long long acc = 0;
            for (int k = 0; k < dimension; k++) acc += (long long)A[i * dimension + k] * (long long)Bt[j * dimension + k];
            C[i * dimension + j] = (int32_t)acc;
        }
    free(Bt);
}
#endif

int main(int argc, char *argv[])
{
    int8_t *A, *B; int32_t *C;
    int opt; int algo = 99;
    while ((opt = getopt(argc, argv, "m:n:a:h")) != -1) {
        switch (opt) {
        case 'n': { long parsed = strtol(optarg, NULL, 0); if (parsed <= 0 || parsed > INT_MAX) { fprintf(stderr, "Invalid dimension: %s\n", optarg); return EXIT_FAILURE; } dimension = (int)parsed; break; }
        case 'a': algo = strtol(optarg, NULL, 0); break;
        default: printf("Usage: %s [-n dimension] [-a algorithm]\n", argv[0]); exit(EXIT_SUCCESS);
        }
    }
    size_t alloc_size8, alloc_size32;
    if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size8) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return EXIT_FAILURE; }
    if (calc_matrix_bytes_with_size(dimension, sizeof(int32_t), &alloc_size32) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return EXIT_FAILURE; }
    A = (int8_t*)aligned_alloc(32, alloc_size8); B = (int8_t*)aligned_alloc(32, alloc_size8); C = (int32_t*)aligned_alloc(32, alloc_size32);
    if (!A || !B || !C) { fprintf(stderr, "Failed to allocate aligned memory for matrices\n"); exit(EXIT_FAILURE); }
    memset(A, 0, alloc_size8); memset(B, 0, alloc_size8); memset(C, 0, alloc_size32);

    if (algo == 100) {
        int32_t *D = (int32_t*)aligned_alloc(32, alloc_size32); if (!D) { fprintf(stderr, "Failed to allocate D\n"); exit(EXIT_FAILURE); }
        init_data(A, B, C, dimension); matmult_opt3_transposed(A, B, C, dimension); memcpy(D, C, alloc_size32); memset(C, 0, alloc_size32);
        init_data(A, B, C, dimension); matmult_opt4_transposed_simd(A, B, C, dimension);
        int diffs = 0; for (int idx = 0; idx < dimension*dimension; idx++) { if (C[idx] != D[idx]) { if (diffs < 10) printf("diff idx %d: opt3=%d opt4=%d\n", idx, D[idx], C[idx]); diffs++; } }
        printf("diff count: %d\n", diffs); free(D); free(A); free(B); free(C); return 0;
    }

    switch(algo) {
    case 0: BENCH(matmult_opt0_naive(A, B, C, dimension)); break;
    case 1: BENCH(matmult_opt1_jk(A, B, C, dimension)); break;
    case 2: BENCH(matmult_opt2_jk_tiling(A, B, C, dimension)); break;
    case 3: BENCH(matmult_opt3_transposed(A, B, C, dimension)); break;
    case 4: BENCH(matmult_opt4_transposed_simd(A, B, C, dimension)); break;
    case 99: 
        BENCH(matmult_opt0_naive(A, B, C, dimension));
        BENCH(matmult_opt1_jk(A, B, C, dimension));
        BENCH(matmult_opt2_jk_tiling(A, B, C, dimension));
        BENCH(matmult_opt3_transposed(A, B, C, dimension));
        BENCH(matmult_opt4_transposed_simd(A, B, C, dimension));
        break;
    default: 
        printf("Unknown algorithm: %d\n", algo); exit(EXIT_FAILURE);
    }

    free(A); free(B); free(C);
    return 0;
}

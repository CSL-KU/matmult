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
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

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
    init_data(A_i8, B_i8, C_i32, dimension); \
    { double t0 = timestamp(); func; double t1 = timestamp(); printf("%.12s  %.6f  chsum: %lld\n", #func, t1-t0, print_checksum(C_i32, dimension)); }

/* FP32 support */
void init_data_fp32(float *A, float *B, float *C, int dimension)
{
    srand(292);
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++) {
            A[dimension*i+j] = (float)((rand() % 101) - 50);
            B[dimension*i+j] = (float)((rand() % 101) - 50);
            C[dimension*i+j] = 0.0f;
        }
}

double print_checksum_fp32(float *C, int dimention)
{
    double sum = 0.0;
    for (int i = 0; i < dimention; i++)
        for (int j = 0; j < dimention; j++)
            sum += (double)C[i*dimention+j];
    return sum;
}

#define BENCH_FP32(func) \
    init_data_fp32(A_f, B_f, C_f, dimension); \
    { double t0 = timestamp(); func; double t1 = timestamp(); printf("%.12s  %.6f  chsum: %.6f\n", #func, t1-t0, print_checksum_fp32(C_f, dimension)); }

void matmult_opt0_naive_fp32(float *A, float *B, float *C, int dimension)
{
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            for (int k = 0; k < dimension; k++)
                C[dimension*i+j] += A[dimension*i+k] * B[dimension*k+j];
}

void matmult_opt1_jk_fp32(float *A, float *B, float *C, int dimension)
{
    for (int i = 0; i < dimension; i++)
        for (int k = 0; k < dimension; k++)
            for (int j = 0; j < dimension; j++)
                C[dimension*i+j] += A[dimension*i+k] * B[dimension*k+j];
}

void matmult_opt2_jk_tiling_fp32(float *A, float *B, float *C, int dimension)
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
                            C[dimension*ii+jj] += A[dimension*ii+kk] * B[dimension*kk+jj];
            }
        }
    }
}

void matmult_opt3_transposed_fp32(float *A, float *B, float *C, int dimension)
{
    size_t alloc_size;
    if (calc_matrix_bytes_with_size(dimension, sizeof(float), &alloc_size) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return; }
    float *Bt = (float*)malloc(alloc_size);
    if (!Bt) { fprintf(stderr, "Failed to allocate memory\n"); return; }
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
            Bt[i*dimension+j] = B[j*dimension+i];
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++) {
            double acc = 0.0;
            for (int k = 0; k < dimension; k++) acc += (double)A[i*dimension+k] * (double)Bt[j*dimension+k];
            C[i*dimension+j] = (float)acc;
        }
    free(Bt);
}

void matmult_opt4_transposed_simd_fp32(float *A, float *B, float *C, int dimension)
{
    size_t alloc_size; if (calc_matrix_bytes_with_size(dimension, sizeof(float), &alloc_size) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return; }
#if defined(__AVX2__)
    float *Bt = (float*)aligned_alloc(32, alloc_size); if (!Bt) { fprintf(stderr, "Failed to allocate aligned memory\n"); return; }
    for (int i = 0; i < dimension; i++) for (int j = 0; j < dimension; j++) Bt[i*dimension + j] = B[j*dimension + i];
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            __m256 acc = _mm256_setzero_ps();
            int k;
            for (k = 0; k <= dimension - 8; k += 8) {
                __m256 a = _mm256_loadu_ps(A + i * dimension + k);
                __m256 b = _mm256_loadu_ps(Bt + j * dimension + k);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
            }
            float tmp[8]; _mm256_storeu_ps(tmp, acc); float sum = 0.0f; for (int t = 0; t < 8; t++) sum += tmp[t];
            for (; k < dimension; k++) sum += A[i * dimension + k] * Bt[j * dimension + k];
            C[i * dimension + j] = sum;
        }
    }
    free(Bt);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    /* NEON FP32 implementation */
    float *Bt = (float*)malloc(alloc_size); if (!Bt) { fprintf(stderr, "Failed to allocate memory\n"); return; }
    for (int i = 0; i < dimension; i++) for (int j = 0; j < dimension; j++) Bt[i*dimension + j] = B[j*dimension + i];
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k <= dimension - 8; k += 8) {
                float32x4_t a0 = vld1q_f32(A + i * dimension + k);
                float32x4_t a1 = vld1q_f32(A + i * dimension + k + 4);
                float32x4_t b0 = vld1q_f32(Bt + j * dimension + k);
                float32x4_t b1 = vld1q_f32(Bt + j * dimension + k + 4);
                acc0 = vmlaq_f32(acc0, a0, b0);
                acc1 = vmlaq_f32(acc1, a1, b1);
            }
            for (; k <= dimension - 4; k += 4) {
                float32x4_t a0 = vld1q_f32(A + i * dimension + k);
                float32x4_t b0 = vld1q_f32(Bt + j * dimension + k);
                acc0 = vmlaq_f32(acc0, a0, b0);
            }
            float sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
            for (; k < dimension; k++) sum += A[i * dimension + k] * Bt[j * dimension + k];
            C[i * dimension + j] = sum;
        }
    }
    free(Bt);
#else
    /* Non-AVX2 fallback: scalar transposed multiply */
    float *Bt = (float*)malloc(alloc_size); if (!Bt) { fprintf(stderr, "Failed to allocate memory\n"); return; }
    for (int i = 0; i < dimension; i++) for (int j = 0; j < dimension; j++) Bt[i*dimension + j] = B[j*dimension + i];
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++) {
            double acc = 0.0;
            for (int k = 0; k < dimension; k++) acc += (double)A[i * dimension + k] * (double)Bt[j * dimension + k];
            C[i * dimension + j] = (float)acc;
        }
    free(Bt);
#endif
}

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
    // Use vdot when available — 8-column blocked implementation
    for (int i = 0; i < dimension; i++) {
        int j;
        for (j = 0; j <= dimension - 8; j += 8) {
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0), acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0), acc5 = vdupq_n_s32(0), acc6 = vdupq_n_s32(0), acc7 = vdupq_n_s32(0);
            int k;
            for (k = 0; k <= dimension - 16; k += 16) {
                int8x16_t a_vec = vld1q_s8(A + i * dimension + k);
                int8x16_t b0_vec = vld1q_s8(Bt + (j + 0) * dimension + k);
                int8x16_t b1_vec = vld1q_s8(Bt + (j + 1) * dimension + k);
                int8x16_t b2_vec = vld1q_s8(Bt + (j + 2) * dimension + k);
                int8x16_t b3_vec = vld1q_s8(Bt + (j + 3) * dimension + k);
                int8x16_t b4_vec = vld1q_s8(Bt + (j + 4) * dimension + k);
                int8x16_t b5_vec = vld1q_s8(Bt + (j + 5) * dimension + k);
                int8x16_t b6_vec = vld1q_s8(Bt + (j + 6) * dimension + k);
                int8x16_t b7_vec = vld1q_s8(Bt + (j + 7) * dimension + k);
                acc0 = vdotq_s32(acc0, a_vec, b0_vec);
                acc1 = vdotq_s32(acc1, a_vec, b1_vec);
                acc2 = vdotq_s32(acc2, a_vec, b2_vec);
                acc3 = vdotq_s32(acc3, a_vec, b3_vec);
                acc4 = vdotq_s32(acc4, a_vec, b4_vec);
                acc5 = vdotq_s32(acc5, a_vec, b5_vec);
                acc6 = vdotq_s32(acc6, a_vec, b6_vec);
                acc7 = vdotq_s32(acc7, a_vec, b7_vec);
            }
            int64_t sum0 = (int64_t)vaddvq_s32(acc0); int64_t sum1 = (int64_t)vaddvq_s32(acc1);
            int64_t sum2 = (int64_t)vaddvq_s32(acc2); int64_t sum3 = (int64_t)vaddvq_s32(acc3);
            int64_t sum4 = (int64_t)vaddvq_s32(acc4); int64_t sum5 = (int64_t)vaddvq_s32(acc5);
            int64_t sum6 = (int64_t)vaddvq_s32(acc6); int64_t sum7 = (int64_t)vaddvq_s32(acc7);
            for (; k < dimension; k++) {
                int32_t a = (int32_t)A[i * dimension + k];
                sum0 += a * (int32_t)Bt[(j + 0) * dimension + k];
                sum1 += a * (int32_t)Bt[(j + 1) * dimension + k];
                sum2 += a * (int32_t)Bt[(j + 2) * dimension + k];
                sum3 += a * (int32_t)Bt[(j + 3) * dimension + k];
                sum4 += a * (int32_t)Bt[(j + 4) * dimension + k];
                sum5 += a * (int32_t)Bt[(j + 5) * dimension + k];
                sum6 += a * (int32_t)Bt[(j + 6) * dimension + k];
                sum7 += a * (int32_t)Bt[(j + 7) * dimension + k];
            }
            C[i * dimension + (j + 0)] = (int32_t)sum0; C[i * dimension + (j + 1)] = (int32_t)sum1;
            C[i * dimension + (j + 2)] = (int32_t)sum2; C[i * dimension + (j + 3)] = (int32_t)sum3;
            C[i * dimension + (j + 4)] = (int32_t)sum4; C[i * dimension + (j + 5)] = (int32_t)sum5;
            C[i * dimension + (j + 6)] = (int32_t)sum6; C[i * dimension + (j + 7)] = (int32_t)sum7;
        }
        // handle remaining 4-column blocks
        for (; j <= dimension - 4; j += 4) {
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
        // remaining single columns
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
    // NEON fallback: 8-column blocked implementation
    for (int i = 0; i < dimension; i++) {
        int j;
        for (j = 0; j <= dimension - 8; j += 8) {
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0), acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0), acc5 = vdupq_n_s32(0), acc6 = vdupq_n_s32(0), acc7 = vdupq_n_s32(0);
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
                int8x16_t b4_vec = vld1q_s8(Bt + (j + 4) * dimension + k);
                int8x16_t b5_vec = vld1q_s8(Bt + (j + 5) * dimension + k);
                int8x16_t b6_vec = vld1q_s8(Bt + (j + 6) * dimension + k);
                int8x16_t b7_vec = vld1q_s8(Bt + (j + 7) * dimension + k);

                int16x8_t b0_lo = vmovl_s8(vget_low_s8(b0_vec)); int16x8_t b0_hi = vmovl_s8(vget_high_s8(b0_vec));
                int16x8_t b1_lo = vmovl_s8(vget_low_s8(b1_vec)); int16x8_t b1_hi = vmovl_s8(vget_high_s8(b1_vec));
                int16x8_t b2_lo = vmovl_s8(vget_low_s8(b2_vec)); int16x8_t b2_hi = vmovl_s8(vget_high_s8(b2_vec));
                int16x8_t b3_lo = vmovl_s8(vget_low_s8(b3_vec)); int16x8_t b3_hi = vmovl_s8(vget_high_s8(b3_vec));
                int16x8_t b4_lo = vmovl_s8(vget_low_s8(b4_vec)); int16x8_t b4_hi = vmovl_s8(vget_high_s8(b4_vec));
                int16x8_t b5_lo = vmovl_s8(vget_low_s8(b5_vec)); int16x8_t b5_hi = vmovl_s8(vget_high_s8(b5_vec));
                int16x8_t b6_lo = vmovl_s8(vget_low_s8(b6_vec)); int16x8_t b6_hi = vmovl_s8(vget_high_s8(b6_vec));
                int16x8_t b7_lo = vmovl_s8(vget_low_s8(b7_vec)); int16x8_t b7_hi = vmovl_s8(vget_high_s8(b7_vec));

                int32x4_t b0_0 = vmovl_s16(vget_low_s16(b0_lo)); int32x4_t b0_1 = vmovl_s16(vget_high_s16(b0_lo));
                int32x4_t b0_2 = vmovl_s16(vget_low_s16(b0_hi)); int32x4_t b0_3 = vmovl_s16(vget_high_s16(b0_hi));
                int32x4_t b1_0 = vmovl_s16(vget_low_s16(b1_lo)); int32x4_t b1_1 = vmovl_s16(vget_high_s16(b1_lo));
                int32x4_t b1_2 = vmovl_s16(vget_low_s16(b1_hi)); int32x4_t b1_3 = vmovl_s16(vget_high_s16(b1_hi));
                int32x4_t b2_0 = vmovl_s16(vget_low_s16(b2_lo)); int32x4_t b2_1 = vmovl_s16(vget_high_s16(b2_lo));
                int32x4_t b2_2 = vmovl_s16(vget_low_s16(b2_hi)); int32x4_t b2_3 = vmovl_s16(vget_high_s16(b2_hi));
                int32x4_t b3_0 = vmovl_s16(vget_low_s16(b3_lo)); int32x4_t b3_1 = vmovl_s16(vget_high_s16(b3_lo));
                int32x4_t b3_2 = vmovl_s16(vget_low_s16(b3_hi)); int32x4_t b3_3 = vmovl_s16(vget_high_s16(b3_hi));
                int32x4_t b4_0 = vmovl_s16(vget_low_s16(b4_lo)); int32x4_t b4_1 = vmovl_s16(vget_high_s16(b4_lo));
                int32x4_t b4_2 = vmovl_s16(vget_low_s16(b4_hi)); int32x4_t b4_3 = vmovl_s16(vget_high_s16(b4_hi));
                int32x4_t b5_0 = vmovl_s16(vget_low_s16(b5_lo)); int32x4_t b5_1 = vmovl_s16(vget_high_s16(b5_lo));
                int32x4_t b5_2 = vmovl_s16(vget_low_s16(b5_hi)); int32x4_t b5_3 = vmovl_s16(vget_high_s16(b5_hi));
                int32x4_t b6_0 = vmovl_s16(vget_low_s16(b6_lo)); int32x4_t b6_1 = vmovl_s16(vget_high_s16(b6_lo));
                int32x4_t b6_2 = vmovl_s16(vget_low_s16(b6_hi)); int32x4_t b6_3 = vmovl_s16(vget_high_s16(b6_hi));
                int32x4_t b7_0 = vmovl_s16(vget_low_s16(b7_lo)); int32x4_t b7_1 = vmovl_s16(vget_high_s16(b7_lo));
                int32x4_t b7_2 = vmovl_s16(vget_low_s16(b7_hi)); int32x4_t b7_3 = vmovl_s16(vget_high_s16(b7_hi));

                acc0 = vaddq_s32(acc0, vmulq_s32(a0, b0_0)); acc0 = vaddq_s32(acc0, vmulq_s32(a1, b0_1));
                acc0 = vaddq_s32(acc0, vmulq_s32(a2, b0_2)); acc0 = vaddq_s32(acc0, vmulq_s32(a3, b0_3));
                acc1 = vaddq_s32(acc1, vmulq_s32(a0, b1_0)); acc1 = vaddq_s32(acc1, vmulq_s32(a1, b1_1));
                acc1 = vaddq_s32(acc1, vmulq_s32(a2, b1_2)); acc1 = vaddq_s32(acc1, vmulq_s32(a3, b1_3));
                acc2 = vaddq_s32(acc2, vmulq_s32(a0, b2_0)); acc2 = vaddq_s32(acc2, vmulq_s32(a1, b2_1));
                acc2 = vaddq_s32(acc2, vmulq_s32(a2, b2_2)); acc2 = vaddq_s32(acc2, vmulq_s32(a3, b2_3));
                acc3 = vaddq_s32(acc3, vmulq_s32(a0, b3_0)); acc3 = vaddq_s32(acc3, vmulq_s32(a1, b3_1));
                acc3 = vaddq_s32(acc3, vmulq_s32(a2, b3_2)); acc3 = vaddq_s32(acc3, vmulq_s32(a3, b3_3));
                acc4 = vaddq_s32(acc4, vmulq_s32(a0, b4_0)); acc4 = vaddq_s32(acc4, vmulq_s32(a1, b4_1));
                acc4 = vaddq_s32(acc4, vmulq_s32(a2, b4_2)); acc4 = vaddq_s32(acc4, vmulq_s32(a3, b4_3));
                acc5 = vaddq_s32(acc5, vmulq_s32(a0, b5_0)); acc5 = vaddq_s32(acc5, vmulq_s32(a1, b5_1));
                acc5 = vaddq_s32(acc5, vmulq_s32(a2, b5_2)); acc5 = vaddq_s32(acc5, vmulq_s32(a3, b5_3));
                acc6 = vaddq_s32(acc6, vmulq_s32(a0, b6_0)); acc6 = vaddq_s32(acc6, vmulq_s32(a1, b6_1));
                acc6 = vaddq_s32(acc6, vmulq_s32(a2, b6_2)); acc6 = vaddq_s32(acc6, vmulq_s32(a3, b6_3));
                acc7 = vaddq_s32(acc7, vmulq_s32(a0, b7_0)); acc7 = vaddq_s32(acc7, vmulq_s32(a1, b7_1));
                acc7 = vaddq_s32(acc7, vmulq_s32(a2, b7_2)); acc7 = vaddq_s32(acc7, vmulq_s32(a3, b7_3));
            }
            int64_t sum0 = (int64_t)vaddvq_s32(acc0); int64_t sum1 = (int64_t)vaddvq_s32(acc1);
            int64_t sum2 = (int64_t)vaddvq_s32(acc2); int64_t sum3 = (int64_t)vaddvq_s32(acc3);
            int64_t sum4 = (int64_t)vaddvq_s32(acc4); int64_t sum5 = (int64_t)vaddvq_s32(acc5);
            int64_t sum6 = (int64_t)vaddvq_s32(acc6); int64_t sum7 = (int64_t)vaddvq_s32(acc7);
            for (; k < dimension; k++) {
                int32_t a = (int32_t)A[i * dimension + k];
                sum0 += a * (int32_t)Bt[(j + 0) * dimension + k];
                sum1 += a * (int32_t)Bt[(j + 1) * dimension + k];
                sum2 += a * (int32_t)Bt[(j + 2) * dimension + k];
                sum3 += a * (int32_t)Bt[(j + 3) * dimension + k];
                sum4 += a * (int32_t)Bt[(j + 4) * dimension + k];
                sum5 += a * (int32_t)Bt[(j + 5) * dimension + k];
                sum6 += a * (int32_t)Bt[(j + 6) * dimension + k];
                sum7 += a * (int32_t)Bt[(j + 7) * dimension + k];
            }
            C[i * dimension + (j + 0)] = (int32_t)sum0; C[i * dimension + (j + 1)] = (int32_t)sum1;
            C[i * dimension + (j + 2)] = (int32_t)sum2; C[i * dimension + (j + 3)] = (int32_t)sum3;
            C[i * dimension + (j + 4)] = (int32_t)sum4; C[i * dimension + (j + 5)] = (int32_t)sum5;
            C[i * dimension + (j + 6)] = (int32_t)sum6; C[i * dimension + (j + 7)] = (int32_t)sum7;
        }
        // handle remaining 4-column blocks
        for (j = (j <= dimension - 4) ? j : j; j <= dimension - 4; j += 4) {
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
    int8_t *A_i8 = NULL, *B_i8 = NULL; int32_t *C_i32 = NULL;
    float *A_f = NULL, *B_f = NULL, *C_f = NULL;
    int opt; int algo = 99; int dtype = 0; /* 0=int8,1=fp32 */
    while ((opt = getopt(argc, argv, "t:n:a:h")) != -1) {
        switch (opt) {
        case 't': {
            if (!optarg) break;
            if (strcmp(optarg, "fp32") == 0 || strcmp(optarg, "float") == 0 || strcmp(optarg, "f") == 0) dtype = 1;
            else dtype = 0;
            break; }
        case 'n': { long parsed = strtol(optarg, NULL, 0); if (parsed <= 0 || parsed > INT_MAX) { fprintf(stderr, "Invalid dimension: %s\n", optarg); return EXIT_FAILURE; } dimension = (int)parsed; break; }
        case 'a': algo = strtol(optarg, NULL, 0); break;
        default: printf("Usage: %s [-t {int8|fp32}] [-n dimension] [-a algorithm]\n", argv[0]); exit(EXIT_SUCCESS);
        }
    }
    if (dtype == 0) {
        size_t alloc_size8, alloc_size32;
        printf("Allocating int8 matrices of dimension %d and result matrix of int32\n", dimension);
        if (calc_matrix_bytes_with_size(dimension, sizeof(int8_t), &alloc_size8) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return EXIT_FAILURE; }
        if (calc_matrix_bytes_with_size(dimension, sizeof(int32_t), &alloc_size32) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return EXIT_FAILURE; }
        A_i8 = (int8_t*)aligned_alloc(32, alloc_size8); B_i8 = (int8_t*)aligned_alloc(32, alloc_size8); C_i32 = (int32_t*)aligned_alloc(32, alloc_size32);
        if (!A_i8 || !B_i8 || !C_i32) { fprintf(stderr, "Failed to allocate aligned memory for int8 matrices\n"); exit(EXIT_FAILURE); }
        memset(A_i8, 0, alloc_size8); memset(B_i8, 0, alloc_size8); memset(C_i32, 0, alloc_size32);
    } else {
        size_t alloc_sizef;
        printf("Allocating fp32 matrices of dimension %d and result matrix of fp32\n", dimension);
        if (calc_matrix_bytes_with_size(dimension, sizeof(float), &alloc_sizef) != 0) { fprintf(stderr, "Invalid dimension for allocation\n"); return EXIT_FAILURE; }
        A_f = (float*)aligned_alloc(32, alloc_sizef); B_f = (float*)aligned_alloc(32, alloc_sizef); C_f = (float*)aligned_alloc(32, alloc_sizef);
        if (!A_f || !B_f || !C_f) { fprintf(stderr, "Failed to allocate aligned memory for fp32 matrices\n"); exit(EXIT_FAILURE); }
        memset(A_f, 0, alloc_sizef); memset(B_f, 0, alloc_sizef); memset(C_f, 0, alloc_sizef);
    }

    if (algo == 100) {
        if (dtype == 0) {
            size_t alloc_size32; calc_matrix_bytes_with_size(dimension, sizeof(int32_t), &alloc_size32);
            int32_t *D = (int32_t*)aligned_alloc(32, alloc_size32); if (!D) { fprintf(stderr, "Failed to allocate D\n"); exit(EXIT_FAILURE); }
            init_data(A_i8, B_i8, C_i32, dimension); matmult_opt3_transposed(A_i8, B_i8, C_i32, dimension); memcpy(D, C_i32, alloc_size32); memset(C_i32, 0, alloc_size32);
            init_data(A_i8, B_i8, C_i32, dimension); matmult_opt4_transposed_simd(A_i8, B_i8, C_i32, dimension);
            int diffs = 0; for (int idx = 0; idx < dimension*dimension; idx++) { if (C_i32[idx] != D[idx]) { if (diffs < 10) printf("diff idx %d: opt3=%d opt4=%d\n", idx, D[idx], C_i32[idx]); diffs++; } }
            printf("diff count: %d\n", diffs); free(D); free(A_i8); free(B_i8); free(C_i32); return 0;
        } else {
            size_t alloc_sizef; calc_matrix_bytes_with_size(dimension, sizeof(float), &alloc_sizef);
            float *D = (float*)aligned_alloc(32, alloc_sizef); if (!D) { fprintf(stderr, "Failed to allocate D\n"); exit(EXIT_FAILURE); }
            init_data_fp32(A_f, B_f, C_f, dimension); matmult_opt3_transposed_fp32(A_f, B_f, C_f, dimension); memcpy(D, C_f, alloc_sizef); memset(C_f, 0, alloc_sizef);
            init_data_fp32(A_f, B_f, C_f, dimension); matmult_opt4_transposed_simd_fp32(A_f, B_f, C_f, dimension);
            int diffs = 0; for (int idx = 0; idx < dimension*dimension; idx++) { if (C_f[idx] != D[idx]) { if (diffs < 10) printf("diff idx %d: opt3=%f opt4=%f\n", idx, D[idx], C_f[idx]); diffs++; } }
            printf("diff count: %d\n", diffs); free(D); free(A_f); free(B_f); free(C_f); return 0;
        }
    }
    switch(algo) {
    case 0:
        if (dtype == 0) { BENCH(matmult_opt0_naive(A_i8, B_i8, C_i32, dimension)); }
        else { BENCH_FP32(matmult_opt0_naive_fp32(A_f, B_f, C_f, dimension)); }
        break;
    case 1:
        if (dtype == 0) { BENCH(matmult_opt1_jk(A_i8, B_i8, C_i32, dimension)); }
        else { BENCH_FP32(matmult_opt1_jk_fp32(A_f, B_f, C_f, dimension)); }
        break;
    case 2:
        if (dtype == 0) { BENCH(matmult_opt2_jk_tiling(A_i8, B_i8, C_i32, dimension)); }
        else { BENCH_FP32(matmult_opt2_jk_tiling_fp32(A_f, B_f, C_f, dimension)); }
        break;
    case 3:
        if (dtype == 0) { BENCH(matmult_opt3_transposed(A_i8, B_i8, C_i32, dimension)); }
        else { BENCH_FP32(matmult_opt3_transposed_fp32(A_f, B_f, C_f, dimension)); }
        break;
    case 4:
        if (dtype == 0) { BENCH(matmult_opt4_transposed_simd(A_i8, B_i8, C_i32, dimension)); }
        else { BENCH_FP32(matmult_opt4_transposed_simd_fp32(A_f, B_f, C_f, dimension)); }
        break;
    case 99:
        if (dtype == 0) {
            BENCH(matmult_opt0_naive(A_i8, B_i8, C_i32, dimension));
            BENCH(matmult_opt1_jk(A_i8, B_i8, C_i32, dimension));
            BENCH(matmult_opt2_jk_tiling(A_i8, B_i8, C_i32, dimension));
            BENCH(matmult_opt3_transposed(A_i8, B_i8, C_i32, dimension));
            BENCH(matmult_opt4_transposed_simd(A_i8, B_i8, C_i32, dimension));
        } else {
            BENCH_FP32(matmult_opt0_naive_fp32(A_f, B_f, C_f, dimension));
            BENCH_FP32(matmult_opt1_jk_fp32(A_f, B_f, C_f, dimension));
            BENCH_FP32(matmult_opt2_jk_tiling_fp32(A_f, B_f, C_f, dimension));
            BENCH_FP32(matmult_opt3_transposed_fp32(A_f, B_f, C_f, dimension));
            BENCH_FP32(matmult_opt4_transposed_simd_fp32(A_f, B_f, C_f, dimension));
        }
        break;
    default:
        printf("Unknown algorithm: %d\n", algo); exit(EXIT_FAILURE);
    }

    if (dtype == 0) {
        free(A_i8); free(B_i8); free(C_i32);
    } else {
        free(A_f); free(B_f); free(C_f);
    }
    return 0;
}

// C++ version of matrix.c with identical algorithms and a few safety fixes
// - Uses RAII where practical
// - Corrects tiling bounds to handle non-multiples of block size
// - Adds safe aligned allocation with size padded to alignment

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>

#include <algorithm>
#include <memory>

// #include <omp.h>

static struct timeval tv;
static int dimension = 1024;
static double start_ts, end_ts;

static inline double timestamp()
{
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

static void init_data(float *A, float *B, float *C, int n)
{
    std::srand(292);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[n * i + j] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
            B[n * i + j] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
            C[n * i + j] = 0.0f;
        }
    }
}

static double checksum(const float *C, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += C[i * n + j];
        }
    }
    return sum;
}

#define BENCH(expr)                                                                        \
    do {                                                                                   \
        init_data(A.get(), B.get(), C.get(), dimension);                                   \
        start_ts = timestamp();                                                             \
        expr;                                                                               \
        end_ts = timestamp();                                                               \
        std::printf("%.12s  %.6f  chsum: %.6f\n", #expr, end_ts - start_ts,               \
                    checksum(C.get(), dimension));                                         \
    } while (0)

// naive i-j-k
static void matmult_opt0_naive(const float *A, const float *B, float *C, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[n * i + j] += (A[n * i + k] * B[n * k + j]);
            }
        }
    }
}

// i-k-j (better access for A)
static void matmult_opt1_jk(const float *A, const float *B, float *C, int n)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[n * i + j] += (A[n * i + k] * B[n * k + j]);
            }
        }
    }
}

// i-k-j with tiling (fixed bounds for non-multiples of bs)
static void matmult_opt2_jk_tiling(const float *A, const float *B, float *C, int n)
{
    const int bs = 256; // 256*256*4 = 256KB block working set
    for (int i = 0; i < n; i += bs) {
        const int i_max = std::min(i + bs, n);
        for (int k = 0; k < n; k += bs) {
            const int k_max = std::min(k + bs, n);
            for (int j = 0; j < n; j += bs) {
                const int j_max = std::min(j + bs, n);
                for (int ii = i; ii < i_max; ii++) {
                    for (int kk = k; kk < k_max; kk++) {
                        for (int jj = j; jj < j_max; jj++) {
                            C[n * ii + jj] += (A[n * ii + kk] * B[n * kk + jj]);
                        }
                    }
                }
            }
        }
    }
}

static void transpose_naive(const float *src, float *dst, int src_row, int src_col)
{
    for (int i = 0; i < src_col; i++) {
        for (int j = 0; j < src_row; j++) {
            dst[i * src_row + j] = src[j * src_col + i];
        }
    }
}

static void matmult_opt3_transposed(const float *A, const float *B, float *C, int n)
{
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    float *Bt = static_cast<float*>(std::malloc(bytes));
    if (!Bt) {
        std::perror("malloc");
        return;
    }
    transpose_naive(B, Bt, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[n * i + j] += (A[n * i + k] * Bt[n * j + k]);
            }
        }
    }
    std::free(Bt);
}

// SIMD variant(s)
#if defined(__AVX2__)
#  include <immintrin.h>
static void matmult_opt4_transposed_simd(const float *A, const float *B, float *C, int n)
{
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    float *Bt = static_cast<float*>(std::aligned_alloc(32, ((bytes + 31) / 32) * 32));
    if (!Bt) { std::fprintf(stderr, "aligned_alloc failed\n"); return; }
    transpose_naive(B, Bt, n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k <= n - 8; k += 8) {
                __m256 a = _mm256_load_ps(A + i * n + k);
                __m256 b = _mm256_load_ps(Bt + j * n + k);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
            }
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float result = _mm_cvtss_f32(sum128);
            for (; k < n; k++) {
                result += A[i * n + k] * Bt[j * n + k];
            }
            C[i * n + j] = result;
        }
    }
    std::free(Bt);
}
#elif defined(__SSE__)
#  include <emmintrin.h>
#  include <smmintrin.h>
static void matmult_opt4_transposed_simd(const float *A, const float *B, float *C, int n)
{
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    float *Bt = static_cast<float*>(std::malloc(bytes));
    if (!Bt) { std::perror("malloc"); return; }
    transpose_naive(B, Bt, n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __m128 acc = _mm_setzero_ps();
            int k = 0;
            for (; k <= n - 4; k += 4) {
                __m128 a = _mm_load_ps(A + i * n + k);
                __m128 b = _mm_load_ps(Bt + j * n + k);
                acc = _mm_add_ps(acc, _mm_mul_ps(a, b));
            }
            float tmp[4];
            _mm_storeu_ps(tmp, acc);
            float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (; k < n; k++) {
                result += A[i * n + k] * Bt[j * n + k];
            }
            C[i * n + j] = result;
        }
    }
    std::free(Bt);
}
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
static void matmult_opt4_transposed_simd(const float *A, const float *B, float *C, int n)
{
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    float *Bt = static_cast<float*>(std::malloc(bytes));
    if (!Bt) { std::perror("malloc"); return; }
    transpose_naive(B, Bt, n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k <= n - 4; k += 4) {
                float32x4_t a = vld1q_f32(A + i * n + k);
                float32x4_t b = vld1q_f32(Bt + j * n + k);
                acc = vaddq_f32(acc, vmulq_f32(a, b));
            }
            float tmp[4];
            vst1q_f32(tmp, acc);
            float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (; k < n; k++) {
                result += A[i * n + k] * Bt[j * n + k];
            }
            C[i * n + j] = result;
        }
    }
    std::free(Bt);
}
#else
// Fallback: just call the transposed scalar path
static void matmult_opt4_transposed_simd(const float *A, const float *B, float *C, int n)
{
    matmult_opt3_transposed(A, B, C, n);
}
#endif

// Utility: aligned allocation with RAII
struct FreeDeleter {
    void operator()(void *p) const noexcept { std::free(p); }
};

static size_t align_up(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }

int main(int argc, char *argv[])
{
    int algo = 99;
    int opt;
    while ((opt = ::getopt(argc, argv, "m:n:a:h")) != -1) {
        switch (opt) {
        case 'n':
            dimension = static_cast<int>(std::strtol(optarg, nullptr, 0));
            break;
        case 'a':
            algo = static_cast<int>(std::strtol(optarg, nullptr, 0));
            break;
        case 'h':
        default:
            std::printf("Usage: %s [-n dimension] [-a algorithm]\n", argv[0]);
            std::printf("  -n dimension: matrix dimension (default: 1024)\n");
            std::printf("  -a algorithm: 0: naive, 1: jk, 2: jk_tiling, 3: transposed, 4: simd\n");
            return 0;
        }
    }

    // set CPU priority to high
    if (setpriority(PRIO_PROCESS, 0, -20) < 0) {
        std::perror("setpriority");
    }

    const size_t bytes = static_cast<size_t>(dimension) * static_cast<size_t>(dimension) * sizeof(float);
    const size_t aligned_bytes = align_up(bytes, 32);

    std::unique_ptr<float, FreeDeleter> A(static_cast<float*>(std::aligned_alloc(32, aligned_bytes)));
    std::unique_ptr<float, FreeDeleter> B(static_cast<float*>(std::aligned_alloc(32, aligned_bytes)));
    std::unique_ptr<float, FreeDeleter> C(static_cast<float*>(std::aligned_alloc(32, aligned_bytes)));

    if (!A || !B || !C) {
        std::fprintf(stderr, "Failed to allocate aligned memory for matrices\n");
        return 1;
    }
    std::memset(A.get(), 0, aligned_bytes);
    std::memset(B.get(), 0, aligned_bytes);
    std::memset(C.get(), 0, aligned_bytes);

    switch (algo) {
    case 0:
        BENCH(matmult_opt0_naive(A.get(), B.get(), C.get(), dimension));
        break;
    case 1:
        BENCH(matmult_opt1_jk(A.get(), B.get(), C.get(), dimension));
        break;
    case 2:
        BENCH(matmult_opt2_jk_tiling(A.get(), B.get(), C.get(), dimension));
        break;
    case 3:
        BENCH(matmult_opt3_transposed(A.get(), B.get(), C.get(), dimension));
        break;
    case 4:
        BENCH(matmult_opt4_transposed_simd(A.get(), B.get(), C.get(), dimension));
        break;
    case 99:
        BENCH(matmult_opt0_naive(A.get(), B.get(), C.get(), dimension));
        BENCH(matmult_opt1_jk(A.get(), B.get(), C.get(), dimension));
        BENCH(matmult_opt2_jk_tiling(A.get(), B.get(), C.get(), dimension));
        BENCH(matmult_opt3_transposed(A.get(), B.get(), C.get(), dimension));
        BENCH(matmult_opt4_transposed_simd(A.get(), B.get(), C.get(), dimension));
        break;
    default:
        std::printf("invalid algorithm\n");
        break;
    }

    return 0;
}

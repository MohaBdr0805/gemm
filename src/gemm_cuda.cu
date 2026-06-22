// Tiled GEMM on the GPU: C = alpha*A*B + beta*C, row-major matrices.
// A 16x16 block computes one tile of C; A and B are streamed through shared
// memory to amortize global-memory traffic.
//
// This file also contains the FUSED inference epilogue (GEMM + bias + activation),
// the core pattern of an inference engine.
#include "gemm/gemm_cuda.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define TILE 16

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace gemm {

// ---------------------------------------------------------------------------
// Base GEMM. As/Bs padded to TILE+1 to avoid bank conflicts.
// ---------------------------------------------------------------------------
__global__ void gemm_kernel(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float beta, float* __restrict__ C) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int ntiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int ak = t * TILE + tx;
        const int bk = t * TILE + ty;
        As[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        const int idx = row * N + col;
        C[idx] = alpha * acc + beta * C[idx];
    }
}

// ---------------------------------------------------------------------------
// Fused GEMM + bias + activation.
// Same as the kernel above, but the epilogue adds bias[col] and applies the
// activation BEFORE writing to global memory -> one write pass, one launch.
// ACT is a template parameter (compile-time constant), so apply_act() is
// resolved at compile time: no branch in the kernel.
// ---------------------------------------------------------------------------
template <Activation ACT>
__global__ void gemm_bias_act_kernel(int M, int N, int K, float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float beta, float* __restrict__ C,
                                     const float* __restrict__ bias) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int ntiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int ak = t * TILE + tx;
        const int bk = t * TILE + ty;
        As[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        const int idx = row * N + col;
        float v = alpha * acc + beta * C[idx];
        if (bias) v += bias[col];          // per-output-column bias
        C[idx] = apply_act(ACT, v);        // fused activation
    }
}

// ---------------------------------------------------------------------------
// Separate element-wise bias+activation (the NON-fused path, for comparison).
// Represents the second global-memory pass that fusion removes.
// ---------------------------------------------------------------------------
template <Activation ACT>
__global__ void bias_act_kernel(int M, int N, float* __restrict__ C,
                                const float* __restrict__ bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        const int col = idx % N;
        float v = C[idx];
        if (bias) v += bias[col];
        C[idx] = apply_act(ACT, v);
    }
}

// ---------------------------------------------------------------------------
// Host wrapper: base GEMM.
// ---------------------------------------------------------------------------
void gemm_cuda(int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C) {
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));

    CUDA_CHECK(cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sc, cudaMemcpyHostToDevice)); // needed for the beta*C term

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// Selects the template instantiation of the fused kernel from a runtime `act`.
static void launch_fused(dim3 grid, dim3 block, int M, int N, int K, float alpha,
                         const float* dA, const float* dB, float beta,
                         float* dC, const float* dbias, Activation act) {
    switch (act) {
        case Activation::ReLU:
            gemm_bias_act_kernel<Activation::ReLU><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC, dbias); break;
        case Activation::GELU:
            gemm_bias_act_kernel<Activation::GELU><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC, dbias); break;
        default:
            gemm_bias_act_kernel<Activation::None><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC, dbias); break;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper: fused GEMM + bias + activation epilogue.
// ---------------------------------------------------------------------------
void gemm_bias_act_cuda(int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C,
                        const float* bias, Activation act) {
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);
    const size_t sbias = (size_t)N * sizeof(float);

    float *dA, *dB, *dC, *dbias = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMemcpy(dA, A, sa, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sc, cudaMemcpyHostToDevice));
    if (bias) {
        CUDA_CHECK(cudaMalloc(&dbias, sbias));
        CUDA_CHECK(cudaMemcpy(dbias, bias, sbias, cudaMemcpyHostToDevice));
    }

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    launch_fused(grid, block, M, N, K, alpha, dA, dB, beta, dC, dbias, act);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, sc, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (dbias) cudaFree(dbias);
}

// ---------------------------------------------------------------------------
// Benchmark: fusion vs two-pass. Device timing (cudaEvent), excludes transfers
// and malloc -> measures exactly what fusion changes: one launch and one global
// memory pass fewer. alpha=1, beta=0 = an inference linear layer Y = act(X*W + b).
// ---------------------------------------------------------------------------
void benchmark_fusion(int M, int N, int K, Activation act) {
    const float alpha = 1.0f, beta = 0.0f;
    const size_t sa = (size_t)M * K * sizeof(float);
    const size_t sb = (size_t)K * N * sizeof(float);
    const size_t sc = (size_t)M * N * sizeof(float);
    const size_t sbias = (size_t)N * sizeof(float);

    float *dA, *dB, *dC, *dbias;
    CUDA_CHECK(cudaMalloc(&dA, sa));
    CUDA_CHECK(cudaMalloc(&dB, sb));
    CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMalloc(&dbias, sbias));
    CUDA_CHECK(cudaMemset(dA, 0, sa));
    CUDA_CHECK(cudaMemset(dB, 0, sb));
    CUDA_CHECK(cudaMemset(dbias, 0, sbias));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    const int threads = 256;
    const int blocks = (M * N + threads - 1) / threads;

    auto run_fused = [&]{ launch_fused(grid, block, M, N, K, alpha, dA, dB, beta, dC, dbias, act); };
    auto run_two_pass = [&]{
        gemm_kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);   // 1st kernel
        switch (act) {                                                     // 2nd kernel (extra pass)
            case Activation::ReLU: bias_act_kernel<Activation::ReLU><<<blocks, threads>>>(M, N, dC, dbias); break;
            case Activation::GELU: bias_act_kernel<Activation::GELU><<<blocks, threads>>>(M, N, dC, dbias); break;
            default:               bias_act_kernel<Activation::None><<<blocks, threads>>>(M, N, dC, dbias); break;
        }
    };

    // Warm-up (context init + JIT cache).
    run_fused(); run_two_pass();
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 50;
    cudaEvent_t s, e; float ms_f = 0.f, ms_u = 0.f;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) run_fused();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_f, s, e));

    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) run_two_pass();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_u, s, e));

    const double gflop = 2.0 * M * N * K / 1e9;
    const double tf = ms_f / iters, tu = ms_u / iters;
    std::printf("  fused    : %7.3f ms/iter   %7.2f GFLOP/s\n", tf, gflop / (tf / 1e3));
    std::printf("  two-pass : %7.3f ms/iter   %7.2f GFLOP/s\n", tu, gflop / (tu / 1e3));
    std::printf("  fusion speedup : %.2fx\n", tu / tf);

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dbias);
}

} // namespace gemm

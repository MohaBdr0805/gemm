#pragma once
#include <cstddef>
#include "gemm/activation.hpp"

// All matrices are stored row-major.
//   A : M x K     B : K x N     C : M x N
// Computes: C = alpha * A * B + beta * C
namespace gemm {

// Naive reference (triple loop). Used as the correctness oracle.
void gemm_naive(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C);

// Optimized version: cache tiling + OpenMP parallelization.
void gemm_tiled(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C);

// CPU reference for the fused inference epilogue (oracle for the GPU kernel):
//   C = act( alpha*A*B + beta*C + bias[col] )
// bias is a length-N vector (one bias per output column), or nullptr.
void gemm_bias_act_ref(int M, int N, int K, float alpha,
                       const float* A, const float* B, float beta, float* C,
                       const float* bias, Activation act);

} // namespace gemm

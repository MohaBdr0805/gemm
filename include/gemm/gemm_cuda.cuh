#pragma once

// GEMM sur GPU (CUDA). Mêmes conventions que gemm_cpu.hpp :
// row-major, A (MxK), B (KxN), C (MxN), C = alpha*A*B + beta*C.
// Pas de dépendance CUDA ici -> ce header s'inclut aussi depuis du code hôte
// (tests, bench) compilé par g++/cl, sans nvcc.
namespace gemm {

void gemm_cuda(int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C);

} // namespace gemm

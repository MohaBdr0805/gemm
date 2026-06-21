#pragma once
#include <cstddef>

// Toutes les matrices sont stockées en row-major (ligne par ligne).
//   A : M x K     B : K x N     C : M x N
// Calcule : C = alpha * A * B + beta * C
namespace gemm {

// Référence naïve (triple boucle). Sert d'oracle de correction.
void gemm_naive(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C);

// Version optimisée : blocage (cache tiling) + parallélisation OpenMP.
void gemm_tiled(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C);

} // namespace gemm

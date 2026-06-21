#include "gemm/gemm_cpu.hpp"
#include <algorithm>

namespace gemm {

// --- Référence naïve : simple, lente, mais évidemment correcte ---------------
void gemm_naive(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}

// --- Version optimisée : blocage + OpenMP ------------------------------------
// Idée : on découpe C en tuiles BS x BS. Chaque thread traite une tuile (ii,jj)
// entière (toutes ses contributions kk), donc aucun thread n'écrit dans la
// tuile d'un autre -> pas de "data race", pas besoin de réduction.
// L'ordre de boucle interne i-k-j garde B et C parcourus par ligne (accès
// mémoire contigus, donc cache-friendly).
void gemm_tiled(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C) {
    constexpr int BS = 64; // taille de tuile (à régler selon le cache L1/L2)

    // 1) Pré-multiplication de C par beta (séparée pour simplifier la suite).
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * N + j] *= beta;

    // 2) Produit bloqué. collapse(2) répartit les tuiles (ii,jj) sur les threads.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            const int i_max = std::min(ii + BS, M);
            const int j_max = std::min(jj + BS, N);
            for (int kk = 0; kk < K; kk += BS) {
                const int k_max = std::min(kk + BS, K);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        const float a = alpha * A[i * K + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace gemm

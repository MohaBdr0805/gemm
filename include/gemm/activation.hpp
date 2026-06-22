#pragma once
// Activation definitions shared between the CPU reference (oracle) and the CUDA
// kernels: a SINGLE source of truth for the math, so a GPU-vs-CPU mismatch can
// never come from a formula that diverges between the two.
//
// The __host__ __device__ guard lets this header be included:
//   - from pure host code (g++/cl)  -> GEMM_HD expands to nothing
//   - from CUDA code (nvcc)         -> functions are also compiled for the device
#if defined(__CUDACC__)
  #define GEMM_HD __host__ __device__
#else
  #define GEMM_HD
#endif

#include <cmath>

namespace gemm {

enum class Activation { None = 0, ReLU = 1, GELU = 2 };

GEMM_HD inline float act_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// GELU, tanh approximation (matches PyTorch / GPT "tanh" mode).
GEMM_HD inline float act_gelu(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Applies the selected activation. When `a` is a compile-time constant (the
// kernel's template parameter) the switch is resolved at compile time -> no
// branch in the inner loop.
GEMM_HD inline float apply_act(Activation a, float x) {
    switch (a) {
        case Activation::ReLU: return act_relu(x);
        case Activation::GELU: return act_gelu(x);
        default:               return x;
    }
}

} // namespace gemm

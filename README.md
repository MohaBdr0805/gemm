# gemm-optim

Réimplémentation optimisée de **GEMM** (`C = alpha·A·B + beta·C`) en C++/CUDA,
avec analyse de performance — projet HPC.

## État
- **CPU** : référence naïve + version optimisée (blocage cache + OpenMP). ✅ testé
- **CUDA** : kernel tuilé en mémoire partagée. 🚧 à venir
- **StarPU** : ordonnancement multi-device (CPU+GPU). 🚧 à venir

## Build (CPU)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # correction
./build/bench 1024                            # benchmark
```

## Build avec CUDA (nécessite un GPU NVIDIA + nvcc)
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Structure
```
include/gemm/   en-têtes publics
src/            implémentations (CPU, CUDA)
benchmarks/     mesures de performance (GFLOP/s)
tests/          test de correction (optimisé vs naïf)
```

## Résultats (exemple)
Version tuilée + OpenMP vs naïve : gain mesuré ~×19 sur 1 cœur
(le gain croît avec le nombre de cœurs).

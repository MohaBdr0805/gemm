# gemm-optim

Réimplémentation optimisée de **GEMM** (`C = α·A·B + β·C`) en C++/OpenMP et CUDA,
avec analyse de performance. Matrices row-major, simple précision.

GEMM est l'opération au cœur du calcul dense (BLAS niveau 3) : c'est le terrain
classique pour démontrer la maîtrise de l'optimisation mémoire et du parallélisme,
sur CPU comme sur GPU. Ce dépôt part d'une référence naïve et y applique, étape
par étape, les optimisations qui comptent — en mesurant chaque gain.

## Ce que le projet démontre

- **CPU** : blocage cache (tiling) + parallélisation **OpenMP**, sans data race.
- **GPU** : kernel **CUDA tuilé en mémoire partagée**, accès **coalescés**, gestion
  des bords (tailles non multiples de la tuile), **padding anti bank-conflicts**.
- **Méthode** : test de correction systématique contre l'oracle naïf, benchmark
  GFLOP/s, et **analyse d'occupancy** pour identifier le vrai goulot d'étranglement.

## Résultats

Mesures sur **GTX 1650** (Turing, `sm_75`) + CPU 12 threads logiques, CUDA 13.3.
GFLOP/s (plus haut = mieux) ; le temps GPU inclut volontairement les transferts
H2D/D2H (perf end-to-end).

| n     | naïf | CPU tuilé (OpenMP) | GPU CUDA |
|-------|------|--------------------|----------|
| 1024  | 0.80 | **46.5**           | **104.6** |
| 2048  | —    | 21.0               | **159.4** |

- CPU tuilé vs naïf : **~58×** à n=1024 (blocage cache + 12 threads).
- GPU vs CPU tuilé : ×2.2 à n=1024, **×7.6** à n=2048 — l'avantage GPU croît avec
  la taille (plus de parallélisme, transferts amortis).
- Correction validée à chaque exécution : écart max GPU vs naïf `4.8e-6` (tolérance `1e-3`).

## Les optimisations

### CPU — `gemm_tiled`
- Découpage de `C` en tuiles `64×64` pour tenir dans le cache.
- `#pragma omp parallel for collapse(2)` sur les tuiles : chaque thread possède une
  tuile entière → **aucune data race**, pas de réduction.
- Ordre de boucle interne `i-k-j` : `B` et `C` parcourus par ligne (accès contigus).

### GPU — `gemm_kernel` (CUDA)
- Tuile `16×16` : chaque bloc calcule une tuile de `C`, en passant `A` et `B` par la
  **mémoire partagée** → le trafic vers la mémoire globale est réduit d'un facteur ~TILE.
- **Coalescing** : `threadIdx.x` indexe la colonne, donc les threads consécutifs d'un
  warp lisent des adresses contiguës en mémoire globale.
- **Bords** : tailles non multiples de 16 gérées par des gardes (chargements
  hors-bornes mis à 0, écriture finale conditionnelle).
- **Bank conflicts** : tableaux partagés paddés en `[TILE][TILE+1]` pour décaler les
  colonnes sur des bancs distincts.

## Analyse de performance (occupancy)

Stats `ptxas` du kernel sur `sm_75` : **36 registres/thread, 0 spill, 2176 o de
mémoire partagée/bloc**. Avec des blocs de 256 threads, la contrainte qui borne
est le nombre de warps par SM (32) → **100 % d'occupancy théorique** (4 blocs ×
256 = 1024 threads/SM), ni les registres ni la mémoire partagée ne limitent.

Conclusion : l'occupancy étant saturée, le kernel **n'est pas borné par l'occupancy**
mais par son **intensité arithmétique** — chaque thread ne calcule qu'un élément de
`C`, soit trop peu de réutilisation par accès mémoire (~5 % du pic FP32). Le levier
suivant n'est donc pas l'occupancy mais le **register tiling** (chaque thread calcule
un micro-bloc de `C`, p. ex. `4×4`), qui augmente la réutilisation et l'intensité
arithmétique. *(Étape suivante du projet.)*

## Build & exécution

**CPU** (Linux/GCC, ou Windows/MinGW) :
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # test de correction
./build/bench 1024                            # benchmark n=1024
```

**Avec CUDA** (GPU NVIDIA + nvcc) :
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # ajoute la vérif GPU vs naïf
./build/bench 1024                            # ajoute la ligne CUDA
```
Architectures GPU ciblées : `75;86` (Turing + Ampère), réglables dans `CMakeLists.txt`.

## Structure

```
include/gemm/   en-têtes publics (gemm_cpu.hpp, gemm_cuda.cuh)
src/            implémentations : gemm_cpu.cpp (naïf + tuilé), gemm_cuda.cu (kernel)
benchmarks/     mesures GFLOP/s
tests/          test de correction (optimisé vs naïf, bords inclus)
```

## Feuille de route

- [x] CPU : naïf + tuilé OpenMP
- [x] GPU : kernel CUDA tuilé mémoire partagée
- [ ] Kernel CUDA v2 : register tiling (micro-bloc par thread)
- [ ] Dockerfile (build reproductible CPU + CUDA)
- [ ] CI GitHub Actions (build + tests CPU)
- [ ] Variante multi-device StarPU (tâches CPU+GPU)

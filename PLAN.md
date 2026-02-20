*From my OpenClaw.*

-Step CUDA GEMM Progression (from naive to near-cuBLAS):


Step 1: Naive Global Memory Baseline
Simple triple-loop kernel: C[i][j] += A[i][k] * B[k][j]. One thread computes one output element. Profile with Nsight Compute — you'll be memory-bound (~1-5% of theoretical peak). Goal: establish your roofline starting point.

Step 2: Coalesced Access Pattern
Restructure to C[row][col] where threads in a warp access contiguous C elements. Use __restrict__ pointer qualifiers. Learn what memory coalescing actually looks like in SASS.

Step 3: Tiled Shared Memory (1D)
Load A tile into __shared__, keep B in registers. First encounter with shared memory bank conflicts (32 banks on CC 7.5). Use __syncthreads(). Target: ~20-30% peak FLOPs.

Step 4: 2D Register Blocking
Each thread computes an 8×8 or 16×16 submatrix. Accumulate in registers, prefetch next tiles into shared memory. Arithmetic intensity jumps — you're now compute-bound. Key metric: FLOPs/byte.

Step 5: Vectorized Loads (float4)
Global memory via float4/int4 — 128-bit transactions. Align allocations with cudaMallocPitch. Reduces memory instruction count 4×.

Step 6: Warp-Level Primitives
Replace __syncthreads() with __shfl_sync() for intra-warp reductions. Experiment with warp-level matrix multiply (WMMA) for Tensor Cores — mixed-precision (FP16/BF16 → FP32 accumulation).

Step 7: Double Buffering + Software Pipelining
Overlap computation of tile N with loads of tile N+1. Hide global memory latency behind math. Requires careful __syncthreads() placement.

Step 8: Auto-Tuning Grid
Grid-search tile sizes (64×64, 128×64, 256×128), thread block counts, and unroll factors for your specific GTX 1650. Benchmark vs. cuBLAS — target 70-80% of peak for a learning project.


Success metric: Single-precision GEMM hitting >1 TFLOP on your 1650 (theoretical: ~3 TFLOPS FP32).

Want me to scaffold Step 1 code?

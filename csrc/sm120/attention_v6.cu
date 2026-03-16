// attention_v6.cu — Flash Attention v6 kernel for sm_120 (Blackwell PCIe)
// Extends gau-nernst v5 with: template DIM (128/256), causal masking, GQA,
// paged KV cache, LSE output, variable sequence lengths.
// Uses only sm_80 features: cp.async, ldmatrix, mma.m16n8k16
// Compile with -arch=sm_80 (backward compatible with sm_120).

#include "common.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <float.h>
#include <math.h>

// ============================================================================
// Gathered global->shared: load rows from non-contiguous global memory
// (paged KV cache) into swizzled shared memory via cp.async.
// Each row is contiguous in head_dim, but rows may come from different pages.
// ============================================================================
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle_gathered(
    uint32_t dst,
    const nv_bfloat16 *base_ptr,    // pool base: [num_pages, page_size, num_kv_heads, DIM]
    const int *page_table,           // page indices for this request
    int page_size,                   // tokens per page
    int kv_head,                     // which KV head
    int num_kv_heads,                // total KV heads (for stride)
    int head_dim,                    // = WIDTH
    int kv_start,                    // starting logical KV position
    int kv_len,                      // total KV length for bounds check
    int tid) {

  constexpr int num_elems = 16 / sizeof(nv_bfloat16);  // 8 bf16 per 16-byte cp.async
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;       // row within tile [0, HEIGHT)
    const int col = idx % WIDTH;       // col within head_dim

    const int logical_pos = kv_start + row;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(
        dst + (row * WIDTH + col) * sizeof(nv_bfloat16));

    if (logical_pos < kv_len) {
      // Translate logical position to physical location
      const int page_idx = logical_pos / page_size;
      const int page_off = logical_pos % page_size;
      const int phys_page = page_table[page_idx];

      // Layout: [num_pages, page_size, num_kv_heads, head_dim]
      const nv_bfloat16 *src_addr = base_ptr
          + (static_cast<int64_t>(phys_page) * page_size + page_off)
            * num_kv_heads * head_dim
          + kv_head * head_dim + col;

      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
    } else {
      // Zero-fill out-of-bounds rows in shared memory
      // Use a store of zeros (cp.async doesn't support zero-fill, so we write directly)
      // We'll handle this with a predicated approach: store zeros after the cp.async group
      // For now, use a regular store (less efficient but correct)
      *reinterpret_cast<uint4*>(
          reinterpret_cast<nv_bfloat16*>(__cvta_shared_to_generic(
              reinterpret_cast<void*>(static_cast<uintptr_t>(dst_addr)))) ) =
          make_uint4(0, 0, 0, 0);
    }
  }
}

// ============================================================================
// Contiguous global->shared with bounds check (for variable-length Q/KV)
// ============================================================================
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle_bounded(
    uint32_t dst,
    const nv_bfloat16 *src,
    int src_stride,
    int valid_rows,  // number of valid rows (rest zero-filled)
    int tid) {

  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(
        dst + (row * WIDTH + col) * sizeof(nv_bfloat16));

    if (row < valid_rows) {
      const nv_bfloat16 *src_addr = src + (row * src_stride + col);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
    } else {
      *reinterpret_cast<uint4*>(
          reinterpret_cast<nv_bfloat16*>(__cvta_shared_to_generic(
              reinterpret_cast<void*>(static_cast<uintptr_t>(dst_addr)))) ) =
          make_uint4(0, 0, 0, 0);
    }
  }
}

// ============================================================================
// Paged global->shared: similar to gathered but with the standard
// sglang paged KV layout: kv_buffer[pool_size, num_kv_heads, head_dim]
// page_table maps logical positions directly to pool slot indices.
// ============================================================================
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle_paged(
    uint32_t dst,
    const nv_bfloat16 *kv_buffer,    // [pool_size, num_kv_heads, head_dim]
    const int *kv_indices,            // maps logical pos -> pool slot index
    int kv_indptr_start,              // kv_indptr[req_id]
    int kv_head,
    int num_kv_heads,
    int head_dim,
    int kv_start,                     // logical start within this request's KV
    int kv_len,                       // total KV length for this request
    int tid) {

  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const int logical_pos = kv_start + row;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(
        dst + (row * WIDTH + col) * sizeof(nv_bfloat16));

    if (logical_pos < kv_len) {
      const int pool_slot = kv_indices[kv_indptr_start + logical_pos];
      // kv_buffer layout: [pool_size, num_kv_heads, head_dim]
      const nv_bfloat16 *src_addr = kv_buffer
          + static_cast<int64_t>(pool_slot) * num_kv_heads * head_dim
          + kv_head * head_dim + col;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
    } else {
      *reinterpret_cast<uint4*>(
          reinterpret_cast<nv_bfloat16*>(__cvta_shared_to_generic(
              reinterpret_cast<void*>(static_cast<uintptr_t>(dst_addr)))) ) =
          make_uint4(0, 0, 0, 0);
    }
  }
}


// ============================================================================
// Main v6 kernel
// Template parameters:
//   BLOCK_Q:   Q tile height (64 for DIM=128, 64 for DIM=256)
//   BLOCK_KV:  KV tile height (64 for DIM=128, 32 for DIM=256)
//   DIM:       head dimension (128 or 256)
//   NUM_WARPS: warps per threadblock (4)
//   CAUSAL:    compile-time causal flag
//   PAGED_KV:  compile-time paged KV flag
// ============================================================================
template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, bool CAUSAL, bool PAGED_KV>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attention_v6_kernel(
    // Q: [total_q_tokens, num_q_heads, DIM] — packed variable-length
    const nv_bfloat16 *Q,
    // For non-paged: K,V: [total_kv_tokens, num_kv_heads, DIM]
    // For paged:     K,V: [pool_size, num_kv_heads, DIM] (token pool)
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    // O: [total_q_tokens, num_q_heads, DIM]
    nv_bfloat16 *O,
    // LSE: [total_q_tokens, num_q_heads] (log-sum-exp output, may be nullptr)
    float *Lse,
    // Sequence structure
    const int *qo_indptr,        // [num_reqs + 1] — Q token offsets per request
    const int *kv_indptr,        // [num_reqs + 1] — KV token offsets (or KV length offsets for paged)
    const int *kv_indices,       // [total_kv_pool_slots] — paged KV: logical pos -> pool slot (only if PAGED_KV)
    // Dimensions
    int num_q_heads,
    int num_kv_heads,
    int num_reqs,
    // For causal: the starting position of each request's Q in the KV timeline
    // qo_start[i] = kv_len[i] - q_len[i]  (for typical autoregressive)
    // If nullptr, assumes qo_start = 0 for all requests (prefill from position 0)
    const int *qo_start_loc      // [num_reqs] or nullptr
) {
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // MMA dimensions
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int gqa_ratio = num_q_heads / num_kv_heads;

  // ========================================================================
  // Map block to (request, q_head, q_tile)
  // Grid: sum over requests of (num_q_heads * cdiv(q_len_i, BLOCK_Q))
  // We use a linear scan to find which request this block belongs to.
  // For production, a binary search or precomputed mapping would be better,
  // but linear scan is fine for reasonable num_reqs.
  // ========================================================================
  int req_id = 0;
  int block_offset = bid;  // blocks remaining after accounting for previous requests

  // Each request contributes num_q_heads * cdiv(q_len, BLOCK_Q) blocks
  for (int r = 0; r < num_reqs; r++) {
    const int q_len_r = qo_indptr[r + 1] - qo_indptr[r];
    const int blocks_for_req = num_q_heads * cdiv(q_len_r, BLOCK_Q);
    if (block_offset < blocks_for_req) {
      req_id = r;
      break;
    }
    block_offset -= blocks_for_req;
  }

  const int q_start = qo_indptr[req_id];          // first Q token for this request
  const int q_len = qo_indptr[req_id + 1] - q_start;
  const int kv_start_global = kv_indptr[req_id];   // first KV token (or indptr start for paged)
  const int kv_len = kv_indptr[req_id + 1] - kv_start_global;

  const int num_q_blocks = cdiv(q_len, BLOCK_Q);
  const int q_head = block_offset / num_q_blocks;
  const int q_block_id = block_offset % num_q_blocks;
  const int kv_head = q_head / gqa_ratio;

  // For causal: Q position in the global KV timeline
  // Position of Q token [q_block_id * BLOCK_Q + i] in KV space = qo_start + q_block_id * BLOCK_Q + i
  int qo_base = 0;
  if constexpr (CAUSAL) {
    if (qo_start_loc != nullptr) {
      qo_base = qo_start_loc[req_id];
    }
    // qo_base = position of q_token[0] in the KV timeline
    // Default: 0 means Q[0] corresponds to KV position 0 (prefill)
  }

  // ========================================================================
  // Shared memory layout:
  //   Q load phase: Q_smem = smem[0 : BLOCK_Q * DIM]
  //   KV phase:     K_smem = smem[0 : 2 * BLOCK_KV * DIM] (double buffered)
  //                 V_smem = smem[2 * BLOCK_KV * DIM : 3 * BLOCK_KV * DIM]
  //   Q overlaps with K+V since Q is loaded once then moved to registers.
  //   smem_size = max(BLOCK_Q, BLOCK_KV * 3) * DIM * sizeof(bf16)
  // ========================================================================
  extern __shared__ nv_bfloat16 smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;  // reuse after Q->regs
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);

  // ========================================================================
  // Register arrays
  // ========================================================================
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

  // ========================================================================
  // Pre-compute ldmatrix smem addresses (with swizzle)
  // ========================================================================
  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile: rows for Q (warp-local) — ldmatrix_x4
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
        Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // B tile: rows for K — ldmatrix_x4 (loads m8n8 tiles)
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
        K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // B tile trans: rows for V — ldmatrix_x4_trans
    const int row_off = lane_id % 16;
    const int col_off = lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
        V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // ========================================================================
  // Load Q tile [BLOCK_Q, DIM] from packed Q tensor
  // Q layout: [total_q_tokens, num_q_heads, DIM]
  // This block's Q starts at token (q_start + q_block_id * BLOCK_Q), head q_head
  // ========================================================================
  {
    const int q_token_start = q_start + q_block_id * BLOCK_Q;
    const int valid_q_rows = min(BLOCK_Q, q_len - q_block_id * BLOCK_Q);

    // Q stride in elements: num_q_heads * DIM (between consecutive tokens)
    const nv_bfloat16 *Q_block = Q + static_cast<int64_t>(q_token_start) * num_q_heads * DIM
                                    + q_head * DIM;

    global_to_shared_swizzle_bounded<BLOCK_Q, DIM, TB_SIZE>(
        Q_smem, Q_block, num_q_heads * DIM, valid_q_rows, tid);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // Q shared -> registers
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      uint32_t addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
      addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  __syncthreads();  // sync before reusing smem for K

  // ========================================================================
  // KV iteration
  // ========================================================================
  const int num_kv_iter = cdiv(kv_len, BLOCK_KV);

  // For causal masking: positions of Q tokens handled by this block
  // q_pos_start = qo_base + q_block_id * BLOCK_Q
  // q_pos_end = qo_base + min((q_block_id + 1) * BLOCK_Q, q_len) - 1
  int q_pos_start, q_pos_end;
  if constexpr (CAUSAL) {
    q_pos_start = qo_base + q_block_id * BLOCK_Q;
    q_pos_end = qo_base + min((q_block_id + 1) * BLOCK_Q, q_len) - 1;
  }

  // Determine KV iteration range for causal (can skip trailing blocks)
  int kv_iter_end = num_kv_iter;
  if constexpr (CAUSAL) {
    // Last KV block we need: the one containing q_pos_end
    kv_iter_end = min(num_kv_iter, cdiv(q_pos_end + 1, BLOCK_KV));
  }

  // Lambda: load K tile into double-buffered smem
  auto load_K = [&](int kv_id, int kv_start_pos) {
    if (kv_id < kv_iter_end) {
      const uint32_t dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
      if constexpr (PAGED_KV) {
        global_to_shared_swizzle_paged<BLOCK_KV, DIM, TB_SIZE>(
            dst, K, kv_indices, kv_start_global,
            kv_head, num_kv_heads, DIM,
            kv_start_pos, kv_len, tid);
      } else {
        // Contiguous KV: K layout [total_kv_tokens, num_kv_heads, DIM]
        const int valid_rows = min(BLOCK_KV, kv_len - kv_start_pos);
        const nv_bfloat16 *K_src = K
            + static_cast<int64_t>(kv_start_global + kv_start_pos) * num_kv_heads * DIM
            + kv_head * DIM;
        global_to_shared_swizzle_bounded<BLOCK_KV, DIM, TB_SIZE>(
            dst, K_src, num_kv_heads * DIM, valid_rows, tid);
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  auto load_V = [&](int kv_id, int kv_start_pos) {
    const uint32_t dst = V_smem;
    if constexpr (PAGED_KV) {
      global_to_shared_swizzle_paged<BLOCK_KV, DIM, TB_SIZE>(
          dst, V, kv_indices, kv_start_global,
          kv_head, num_kv_heads, DIM,
          kv_start_pos, kv_len, tid);
    } else {
      const int valid_rows = min(BLOCK_KV, kv_len - kv_start_pos);
      const nv_bfloat16 *V_src = V
          + static_cast<int64_t>(kv_start_global + kv_start_pos) * num_kv_heads * DIM
          + kv_head * DIM;
      global_to_shared_swizzle_bounded<BLOCK_KV, DIM, TB_SIZE>(
          dst, V_src, num_kv_heads * DIM, valid_rows, tid);
    }
    asm volatile("cp.async.commit_group;");
  };

  // Prefetch first K tile
  load_K(0, 0);

  for (int kv_id = 0; kv_id < kv_iter_end; kv_id++) {
    const int kv_block_start = kv_id * BLOCK_KV;

    // Causal early-exit: if entire KV block is after all Q positions, skip
    if constexpr (CAUSAL) {
      if (kv_block_start > q_pos_end) break;
    }

    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // Prefetch V (need to sync to ensure V_smem from previous iter is consumed)
    __syncthreads();
    load_V(kv_id, kv_block_start);

    // K shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
        addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    // S = Q @ K^T  [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],
                       K_rmem[mma_id_kv][mma_id_d],
                       S_rmem[mma_id_q][mma_id_kv]);

    // Prefetch next K tile
    load_K(kv_id + 1, (kv_id + 1) * BLOCK_KV);

    // ====================================================================
    // Softmax + causal masking + online rescaling
    // ====================================================================
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // Apply softmax scale
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
          S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

      // ================================================================
      // Causal masking: set S[q,k] = -inf where kv_pos > q_pos
      // MMA m16n8k16 output layout per thread:
      //   reg[0]: row = lane_id/4,       col = (lane_id%4)*2
      //   reg[1]: row = lane_id/4,       col = (lane_id%4)*2 + 1
      //   reg[2]: row = lane_id/4 + 8,   col = (lane_id%4)*2
      //   reg[3]: row = lane_id/4 + 8,   col = (lane_id%4)*2 + 1
      // ================================================================
      if constexpr (CAUSAL) {
        // Q positions for this warp's MMA tile
        // Global Q position = qo_base + q_block_id * BLOCK_Q + warp_id * WARP_Q + mma_id_q * MMA_M + row_in_mma
        const int q_row_base = qo_base + q_block_id * BLOCK_Q + warp_id * WARP_Q + mma_id_q * MMA_M;
        const int q_row_0 = q_row_base + (lane_id / 4);      // rows 0..7
        const int q_row_1 = q_row_base + (lane_id / 4) + 8;  // rows 8..15

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          const int kv_col_base = kv_block_start + mma_id_kv * MMA_N;
          const int kv_col_0 = kv_col_base + (lane_id % 4) * 2;
          const int kv_col_1 = kv_col_0 + 1;

          float *regs = S_rmem[mma_id_q][mma_id_kv];
          // reg[0]: (q_row_0, kv_col_0)
          if (kv_col_0 > q_row_0) regs[0] = -FLT_MAX;
          // reg[1]: (q_row_0, kv_col_1)
          if (kv_col_1 > q_row_0) regs[1] = -FLT_MAX;
          // reg[2]: (q_row_1, kv_col_0)
          if (kv_col_0 > q_row_1) regs[2] = -FLT_MAX;
          // reg[3]: (q_row_1, kv_col_1)
          if (kv_col_1 > q_row_1) regs[3] = -FLT_MAX;
        }
      }

      // Also mask out-of-bounds Q rows (for the tail block)
      {
        const int q_local_base = q_block_id * BLOCK_Q + warp_id * WARP_Q + mma_id_q * MMA_M;
        const int q_local_0 = q_local_base + (lane_id / 4);
        const int q_local_1 = q_local_base + (lane_id / 4) + 8;

        if (q_local_0 >= q_len || q_local_1 >= q_len) {
          for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_q][mma_id_kv];
            if (q_local_0 >= q_len) { regs[0] = -FLT_MAX; regs[1] = -FLT_MAX; }
            if (q_local_1 >= q_len) { regs[2] = -FLT_MAX; regs[3] = -FLT_MAX; }
          }
        }
      }

      // Also mask out-of-bounds KV positions (for the tail KV block)
      {
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          const int kv_col_base = kv_block_start + mma_id_kv * MMA_N;
          const int kv_col_0 = kv_col_base + (lane_id % 4) * 2;
          const int kv_col_1 = kv_col_0 + 1;

          if (kv_col_0 >= kv_len || kv_col_1 >= kv_len) {
            float *regs = S_rmem[mma_id_q][mma_id_kv];
            if (kv_col_0 >= kv_len) { regs[0] = -FLT_MAX; regs[2] = -FLT_MAX; }
            if (kv_col_1 >= kv_len) { regs[1] = -FLT_MAX; regs[3] = -FLT_MAX; }
          }
        }
      }

      // Rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        if (mma_id_kv == 0) {
          this_rowmax[0] = max(regs[0], regs[1]);
          this_rowmax[1] = max(regs[2], regs[3]);
        } else {
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
        }
      }

      // Butterfly reduction within 4 threads (MMA N-group)
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // New rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      // Rescale previous O accumulator
      float rescale[2];
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // Save new rowmax
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // Rowsumexp + pack P for next MMA
      float this_rowsumexp[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = regs[0] + regs[1];
          this_rowsumexp[1] = regs[2] + regs[3];
        } else {
          this_rowsumexp[0] += regs[0] + regs[1];
          this_rowsumexp[1] += regs[2] + regs[3];
        }

        // Pack to P registers (m16n8 -> m16k16 layout for next MMA)
        // A operand layout: {A[0]=row0_k0-7, A[1]=row0_k8-15, A[2]=row8_k0-7, A[3]=row8_k8-15}
        nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_rmem[(mma_id_kv % 2)]     = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) + 2] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      // Butterfly reduction for rowsumexp
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // Accumulate total rowsumexp
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    // V shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread;
        addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
        addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
        ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    // O += P @ V  [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                       V_rmem[mma_id_kv][mma_id_d],
                       O_rmem[mma_id_q][mma_id_d]);
  }

  // ========================================================================
  // Write O and LSE output
  // ========================================================================
  const int q_token_start = q_start + q_block_id * BLOCK_Q;

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int local_row_0 = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int local_row_1 = local_row_0 + 8;
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      float *regs = O_rmem[mma_id_q][mma_id_d];

      // Divide by softmax denominator
      float denom0 = rowsumexp[mma_id_q][0];
      float denom1 = rowsumexp[mma_id_q][1];
      // Protect against div-by-zero (all masked rows)
      if (denom0 == 0.0f) denom0 = 1.0f;
      if (denom1 == 0.0f) denom1 = 1.0f;

      regs[0] /= denom0;
      regs[1] /= denom0;
      regs[2] /= denom1;
      regs[3] /= denom1;

      // O layout: [total_q_tokens, num_q_heads, DIM]
      if (local_row_0 < q_len) {
        const int global_token_0 = q_token_start + local_row_0;
        nv_bfloat16 *O_row_0 = O + static_cast<int64_t>(global_token_0) * num_q_heads * DIM
                                  + q_head * DIM + col;
        reinterpret_cast<nv_bfloat162 *>(O_row_0)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      }
      if (local_row_1 < q_len) {
        const int global_token_1 = q_token_start + local_row_1;
        nv_bfloat16 *O_row_1 = O + static_cast<int64_t>(global_token_1) * num_q_heads * DIM
                                  + q_head * DIM + col;
        reinterpret_cast<nv_bfloat162 *>(O_row_1)[0] = __float22bfloat162_rn({regs[2], regs[3]});
      }
    }

  // Write LSE: lse[token, head] = rowmax + log(rowsumexp)
  // Only one thread per row needs to write (any thread in the row's group has the same value)
  // We let the thread with lane_id%4==0 write (it has the reduced values)
  if (Lse != nullptr && (lane_id % 4) == 0) {
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      const int local_row_0 = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int local_row_1 = local_row_0 + 8;

      if (local_row_0 < q_len) {
        const int global_token_0 = q_token_start + local_row_0;
        float lse_val = rowmax[mma_id_q][0] + logf(rowsumexp[mma_id_q][0]);
        Lse[static_cast<int64_t>(global_token_0) * num_q_heads + q_head] = lse_val;
      }
      if (local_row_1 < q_len) {
        const int global_token_1 = q_token_start + local_row_1;
        float lse_val = rowmax[mma_id_q][1] + logf(rowsumexp[mma_id_q][1]);
        Lse[static_cast<int64_t>(global_token_1) * num_q_heads + q_head] = lse_val;
      }
    }
  }
}


// ============================================================================
// Compute total number of threadblocks across all requests
// ============================================================================
static int compute_num_blocks(
    const int *qo_indptr_host,  // host pointer
    int num_reqs,
    int num_q_heads,
    int block_q) {
  int total = 0;
  for (int r = 0; r < num_reqs; r++) {
    int q_len = qo_indptr_host[r + 1] - qo_indptr_host[r];
    total += num_q_heads * cdiv(q_len, block_q);
  }
  return total;
}


// ============================================================================
// C++ launch wrappers
// ============================================================================

// Unified launcher — selects template based on (DIM, CAUSAL, PAGED_KV)
void attention_v6_launch(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    float *Lse,                // may be nullptr
    const int *qo_indptr,      // device
    const int *kv_indptr,      // device
    const int *kv_indices,     // device, may be nullptr if not paged
    const int *qo_indptr_host, // host copy for grid computation
    int num_q_heads,
    int num_kv_heads,
    int num_reqs,
    int head_dim,
    bool causal,
    bool paged_kv,
    const int *qo_start_loc    // device, may be nullptr
) {
  // Select block sizes based on head_dim
  if (head_dim == 128) {
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int DIM = 128;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

    const int num_blocks = compute_num_blocks(qo_indptr_host, num_reqs, num_q_heads, BLOCK_Q);
    const int smem_size = max(BLOCK_Q, BLOCK_KV * 3) * DIM * sizeof(nv_bfloat16);

    #define LAUNCH_V6(CAUSAL_VAL, PAGED_VAL) \
      { \
        auto kernel = attention_v6_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, CAUSAL_VAL, PAGED_VAL>; \
        launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, \
            Q, K, V, O, Lse, qo_indptr, kv_indptr, kv_indices, \
            num_q_heads, num_kv_heads, num_reqs, qo_start_loc); \
      }

    if (causal && paged_kv)       LAUNCH_V6(true, true)
    else if (causal && !paged_kv) LAUNCH_V6(true, false)
    else if (!causal && paged_kv) LAUNCH_V6(false, true)
    else                          LAUNCH_V6(false, false)

    #undef LAUNCH_V6

  } else if (head_dim == 256) {
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 32;
    constexpr int DIM = 256;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

    const int num_blocks = compute_num_blocks(qo_indptr_host, num_reqs, num_q_heads, BLOCK_Q);
    const int smem_size = max(BLOCK_Q, BLOCK_KV * 3) * DIM * sizeof(nv_bfloat16);
    // smem = max(64, 96) * 256 * 2 = 49152 bytes

    #define LAUNCH_V6_256(CAUSAL_VAL, PAGED_VAL) \
      { \
        auto kernel = attention_v6_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, CAUSAL_VAL, PAGED_VAL>; \
        launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, \
            Q, K, V, O, Lse, qo_indptr, kv_indptr, kv_indices, \
            num_q_heads, num_kv_heads, num_reqs, qo_start_loc); \
      }

    if (causal && paged_kv)       LAUNCH_V6_256(true, true)
    else if (causal && !paged_kv) LAUNCH_V6_256(true, false)
    else if (!causal && paged_kv) LAUNCH_V6_256(false, true)
    else                          LAUNCH_V6_256(false, false)

    #undef LAUNCH_V6_256
  }
}

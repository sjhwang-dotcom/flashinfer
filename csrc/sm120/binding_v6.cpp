// binding_v6.cpp — PyTorch C++ extension binding for sm_120 Flash Attention v6
// Extends v5 with: head_dim 128/256, causal masking, GQA, paged KV, LSE, variable seqlen.
//
// This file is compiled alongside attention_v5.cu and attention_v6.cu.
// It provides both the original v5 forward and the new v6 forward.

#include <torch/extension.h>
#include <cuda_bf16.h>

// ============================================================================
// v5 declarations (unchanged)
// ============================================================================
void attention_v5_launch(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    int bs,
    int len_q,
    int len_kv);

// ============================================================================
// v6 declaration (from attention_v6.cu)
// ============================================================================
void attention_v6_launch(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    float *Lse,
    const int *qo_indptr,
    const int *kv_indptr,
    const int *kv_indices,
    const int *qo_indptr_host,
    int num_q_heads,
    int num_kv_heads,
    int num_reqs,
    int head_dim,
    bool causal,
    bool paged_kv,
    const int *qo_start_loc);

// ============================================================================
// v5 PyTorch wrapper (preserved for backward compat)
// Q: [bs, len_q, 128] bf16
// K: [bs, len_kv, 128] bf16
// V: [bs, len_kv, 128] bf16
// Returns: O [bs, len_q, 128] bf16
// ============================================================================
torch::Tensor sm120_fa_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {

  TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
  TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
  TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
  TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
  TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
  TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be bfloat16");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  TORCH_CHECK(Q.dim() == 3, "Q must be 3D [bs, len_q, dim]");
  TORCH_CHECK(K.dim() == 3, "K must be 3D [bs, len_kv, dim]");
  TORCH_CHECK(V.dim() == 3, "V must be 3D [bs, len_kv, dim]");

  const int bs = Q.size(0);
  const int len_q = Q.size(1);
  const int len_kv = K.size(1);
  const int dim = Q.size(2);

  TORCH_CHECK(dim == 128, "Only head_dim=128 is supported for v5, got ", dim);
  TORCH_CHECK(K.size(0) == bs, "K batch size mismatch");
  TORCH_CHECK(V.size(0) == bs, "V batch size mismatch");
  TORCH_CHECK(K.size(1) == len_kv, "K/V len_kv mismatch");
  TORCH_CHECK(V.size(1) == len_kv, "K/V len_kv mismatch");
  TORCH_CHECK(K.size(2) == dim, "K dim mismatch");
  TORCH_CHECK(V.size(2) == dim, "V dim mismatch");

  auto O = torch::empty_like(Q);

  attention_v5_launch(
      reinterpret_cast<const nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
      reinterpret_cast<const nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
      reinterpret_cast<nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
      bs, len_q, len_kv);

  return O;
}


// ============================================================================
// v6 PyTorch wrapper
//
// Q: [total_q_tokens, num_q_heads, head_dim] bf16
// K: [total_kv_tokens, num_kv_heads, head_dim] bf16  (or paged pool)
// V: [total_kv_tokens, num_kv_heads, head_dim] bf16  (or paged pool)
// qo_indptr: [num_reqs + 1] int32 — Q token offsets per request
// kv_indptr: [num_reqs + 1] int32 — KV token offsets per request
// kv_indices: [total_kv_pool_slots] int32 or empty — paged KV indices
// qo_start_loc: [num_reqs] int32 or empty — causal Q start positions
//
// Returns: (O, Lse) where:
//   O:   [total_q_tokens, num_q_heads, head_dim] bf16
//   Lse: [total_q_tokens, num_q_heads] f32
// ============================================================================
std::vector<torch::Tensor> sm120_fa_v6_forward(
    torch::Tensor Q,             // [total_q, num_q_heads, head_dim]
    torch::Tensor K,             // [total_kv, num_kv_heads, head_dim] or paged pool
    torch::Tensor V,             // same as K
    torch::Tensor qo_indptr,     // [num_reqs + 1]
    torch::Tensor kv_indptr,     // [num_reqs + 1]
    torch::Tensor kv_indices,    // [total_pool_slots] or empty
    torch::Tensor qo_start_loc,  // [num_reqs] or empty
    int num_q_heads,
    int num_kv_heads,
    bool causal,
    bool return_lse) {

  // Type checks
  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
  TORCH_CHECK(K.is_cuda(), "K must be CUDA");
  TORCH_CHECK(V.is_cuda(), "V must be CUDA");
  TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
  TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
  TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be bfloat16");
  TORCH_CHECK(qo_indptr.dtype() == torch::kInt32, "qo_indptr must be int32");
  TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must be int32");

  // Shape checks
  TORCH_CHECK(Q.dim() == 3, "Q must be 3D [total_q, num_q_heads, head_dim]");
  TORCH_CHECK(K.dim() == 3, "K must be 3D");
  TORCH_CHECK(V.dim() == 3, "V must be 3D");

  const int total_q = Q.size(0);
  const int head_dim = Q.size(2);
  const int num_reqs = qo_indptr.size(0) - 1;

  TORCH_CHECK(head_dim == 128 || head_dim == 256,
      "head_dim must be 128 or 256, got ", head_dim);
  TORCH_CHECK(Q.size(1) == num_q_heads, "Q num_heads mismatch");
  TORCH_CHECK(K.size(2) == head_dim, "K head_dim mismatch");
  TORCH_CHECK(V.size(2) == head_dim, "V head_dim mismatch");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
      "num_q_heads must be divisible by num_kv_heads");

  // Contiguity
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  // Paged KV?
  bool paged_kv = kv_indices.numel() > 0;
  if (paged_kv) {
    TORCH_CHECK(kv_indices.dtype() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA");
  }

  // qo_start_loc
  bool has_qo_start = qo_start_loc.numel() > 0;
  if (has_qo_start) {
    TORCH_CHECK(qo_start_loc.dtype() == torch::kInt32, "qo_start_loc must be int32");
    TORCH_CHECK(qo_start_loc.is_cuda(), "qo_start_loc must be CUDA");
  }

  // Allocate output
  auto O = torch::empty_like(Q);

  // Allocate LSE
  torch::Tensor Lse;
  float *lse_ptr = nullptr;
  if (return_lse) {
    Lse = torch::empty({total_q, num_q_heads},
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
    lse_ptr = Lse.data_ptr<float>();
  } else {
    Lse = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
  }

  // We need qo_indptr on the host for grid size computation.
  // Copy to host (synchronous, but only num_reqs+1 ints — negligible).
  auto qo_indptr_cpu = qo_indptr.to(torch::kCPU);
  const int *qo_indptr_host = qo_indptr_cpu.data_ptr<int>();

  attention_v6_launch(
      reinterpret_cast<const nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
      reinterpret_cast<const nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
      reinterpret_cast<nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
      lse_ptr,
      qo_indptr.data_ptr<int>(),
      kv_indptr.data_ptr<int>(),
      paged_kv ? kv_indices.data_ptr<int>() : nullptr,
      qo_indptr_host,
      num_q_heads,
      num_kv_heads,
      num_reqs,
      head_dim,
      causal,
      paged_kv,
      has_qo_start ? qo_start_loc.data_ptr<int>() : nullptr);

  return {O, Lse};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sm120_fa_forward,
        "SM120 Flash Attention v5 forward (bf16, head_dim=128)",
        py::arg("Q"), py::arg("K"), py::arg("V"));

  m.def("v6_forward", &sm120_fa_v6_forward,
        "SM120 Flash Attention v6 forward (bf16, head_dim=128/256, causal, GQA, paged KV, var seqlen)",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("qo_indptr"),
        py::arg("kv_indptr"),
        py::arg("kv_indices"),
        py::arg("qo_start_loc"),
        py::arg("num_q_heads"),
        py::arg("num_kv_heads"),
        py::arg("causal"),
        py::arg("return_lse"));
}

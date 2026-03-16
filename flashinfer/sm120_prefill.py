"""SM120 (Blackwell workstation) native FlashAttention prefill kernel.

Uses a custom CUDA kernel (v6) for paged attention with head_dim=256 on sm_120,
where flashinfer's FA2 plan() computes invalid split_kv configurations.

The kernel supports: causal, GQA, paged KV (token-level pool), variable-length,
LSE output, and uses only sm_80 features (cp.async, ldmatrix, mma.m16n8k16).
"""

import os
import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_sm120_module = None
_sm120_module_failed = False


def is_sm120_available() -> bool:
    """Check if current GPU is sm_120 (Blackwell workstation, cc=12.0)."""
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    return cc[0] == 12 and cc[1] == 0


def _get_csrc_dir():
    """Get the path to csrc/sm120/ directory."""
    # Try package data directory first (installed mode)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    data_csrc = os.path.join(pkg_dir, "data", "csrc", "sm120")
    if os.path.isdir(data_csrc):
        return data_csrc
    # Try repo root (development mode)
    repo_csrc = os.path.join(os.path.dirname(pkg_dir), "csrc", "sm120")
    if os.path.isdir(repo_csrc):
        return repo_csrc
    raise FileNotFoundError(
        f"Cannot find sm120 kernel sources. Checked:\n  {data_csrc}\n  {repo_csrc}"
    )


def _load_module():
    """JIT compile and load the sm120 FA v6 kernel module."""
    global _sm120_module, _sm120_module_failed
    if _sm120_module is not None:
        return _sm120_module
    if _sm120_module_failed:
        return None

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = _get_csrc_dir()
        sources = [
            os.path.join(csrc_dir, "binding_v6.cpp"),
            os.path.join(csrc_dir, "attention_v5.cu"),
            os.path.join(csrc_dir, "attention_v6.cu"),
        ]
        for src in sources:
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing source: {src}")

        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "flashinfer", "sm120_v6"
        )
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("JIT compiling sm120 FA v6 kernel...")
        mod = load(
            name="sm120_fa_v6",
            sources=sources,
            extra_cuda_cflags=["-arch=sm_120", "-O3", "--use_fast_math"],
            extra_include_paths=[csrc_dir],
            build_directory=cache_dir,
            verbose=False,
        )
        _sm120_module = mod
        logger.info("sm120 FA v6 kernel compiled successfully.")
        return mod
    except Exception as e:
        logger.warning(f"Failed to compile sm120 FA v6 kernel: {e}")
        _sm120_module_failed = True
        return None


def should_use_sm120_kernel(head_dim_qk: int, head_dim_vo: int) -> bool:
    """Determine if sm120 native kernel should be used."""
    return is_sm120_available() and max(head_dim_qk, head_dim_vo) >= 256


def sm120_prefill_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    page_size: int,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run sm120 native FA v6 kernel for paged prefill attention.

    Args:
        q: [total_q, num_qo_heads, head_dim]
        k_cache: [max_num_pages, page_size, num_kv_heads, head_dim] (NHD layout)
        v_cache: [max_num_pages, page_size, num_kv_heads, head_dim] (NHD layout)
        qo_indptr: [batch_size + 1]
        paged_kv_indptr: [batch_size + 1]
        paged_kv_indices: [total_pages]
        paged_kv_last_page_len: [batch_size]
        page_size: tokens per page
        causal: whether to use causal masking
        sm_scale: softmax scale (default: 1/sqrt(head_dim))
        return_lse: whether to return log-sum-exp
        out: optional pre-allocated output tensor
        lse: optional pre-allocated LSE tensor

    Returns:
        (output, lse) if return_lse else (output, None)
    """
    mod = _load_module()
    if mod is None:
        raise RuntimeError("sm120 FA v6 kernel not available")

    total_q = q.shape[0]
    head_dim = q.shape[2]
    batch_size = qo_indptr.shape[0] - 1

    # Convert paged KV to token-level indices for v6 kernel
    # v6 kernel expects flat token-level kv_indices and kv_indptr
    if page_size == 1:
        # Token-level pool: paged indices ARE token indices
        kv_indices = paged_kv_indices
        kv_indptr = paged_kv_indptr
    else:
        # Page-level pool: expand page indices to token indices
        kv_indices, kv_indptr = _expand_paged_to_token_indices(
            paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, page_size
        )

    # Reshape KV cache to token-level: [pool_size, num_kv_heads, head_dim]
    if page_size == 1:
        k_buffer = k_cache.squeeze(1)  # [pool_size, 1, H, D] -> [pool_size, H, D]
        v_buffer = v_cache.squeeze(1)
    else:
        # Flatten pages: [max_pages, page_size, H, D] -> [max_pages*page_size, H, D]
        k_buffer = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        v_buffer = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

    # Compute qo_start_loc for causal masking with prefix
    if batch_size > 0:
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        kv_lens = kv_indptr[1:] - kv_indptr[:-1]
        qo_start_loc = (kv_lens - q_lens).to(torch.int32)
    else:
        qo_start_loc = torch.empty(0, dtype=torch.int32, device=q.device)

    # Ensure contiguous int32 indices
    kv_indices = kv_indices.contiguous().to(torch.int32)
    kv_indptr = kv_indptr.contiguous().to(torch.int32)
    qo_indptr_i32 = qo_indptr.contiguous().to(torch.int32)

    # Call v6 kernel
    results = mod.v6_forward(
        q,          # [total_q, num_qo_heads, head_dim]
        k_buffer,   # [pool_size, num_kv_heads, head_dim]
        v_buffer,   # [pool_size, num_kv_heads, head_dim]
        qo_indptr_i32,
        kv_indptr,
        kv_indices,
        qo_start_loc,
        num_qo_heads,
        num_kv_heads,
        causal,
        return_lse,
    )

    output = results[0]  # [total_q, num_qo_heads, head_dim]

    # Apply sm_scale if different from kernel's default (1/sqrt(head_dim))
    if sm_scale is not None:
        default_scale = 1.0 / (head_dim ** 0.5)
        if abs(sm_scale - default_scale) > 1e-6:
            # Kernel uses 1/sqrt(d), adjust by ratio
            output = output * (sm_scale / default_scale)

    if out is not None:
        out.copy_(output)
        output = out

    result_lse = results[1] if return_lse and len(results) > 1 else None
    if lse is not None and result_lse is not None:
        lse.copy_(result_lse)
        result_lse = lse

    return output, result_lse


def _expand_paged_to_token_indices(
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand page-level indices to token-level indices.

    Converts flashinfer's (paged_kv_indptr, paged_kv_indices, last_page_len) format
    to flat (kv_indptr, kv_indices) token-level format.
    """
    batch_size = paged_kv_indptr.shape[0] - 1
    device = paged_kv_indices.device

    # Compute kv_lens per request
    paged_indptr_cpu = paged_kv_indptr.cpu()
    last_page_len_cpu = paged_kv_last_page_len.cpu()

    kv_lens = []
    all_token_indices = []
    for i in range(batch_size):
        start = paged_indptr_cpu[i].item()
        end = paged_indptr_cpu[i + 1].item()
        num_pages = end - start
        if num_pages == 0:
            kv_lens.append(0)
            continue
        last_len = last_page_len_cpu[i].item()
        kv_len = (num_pages - 1) * page_size + last_len
        kv_lens.append(kv_len)

        page_ids = paged_kv_indices[start:end]
        for p_idx in range(num_pages):
            page_id = page_ids[p_idx]
            tokens_in_page = last_len if p_idx == num_pages - 1 else page_size
            base = page_id * page_size
            for t in range(tokens_in_page):
                all_token_indices.append(base + t)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        kv_indptr[i + 1] = kv_indptr[i] + kv_lens[i]

    if all_token_indices:
        kv_indices = torch.tensor(all_token_indices, dtype=torch.int32, device=device)
    else:
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)

    return kv_indices, kv_indptr

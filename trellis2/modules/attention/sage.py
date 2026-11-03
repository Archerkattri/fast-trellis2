"""Optional inference-only SageAttention support for selected flow attention."""
import torch


def sage_attention(q, k, v, backend="sage"):
    if not q.is_cuda:
        raise ValueError("SageAttention requires CUDA tensors")
    if torch.is_grad_enabled() and any(value.requires_grad for value in (q, k, v)):
        raise ValueError("SageAttention is an inference-only backend")
    try:
        import sageattention
    except ImportError as error:
        raise ImportError(
            "Install the optional SageAttention package for your PyTorch/CUDA version "
            "before enabling Sage attention."
        ) from error
    if backend == "sage_fp16pv":
        return sageattention.sageattn_qk_int8_pv_fp16_cuda(
            q, k, v, tensor_layout="NHD", is_causal=False, pv_accum_dtype="fp32"
        )
    return sageattention.sageattn(q, k, v, tensor_layout="NHD", is_causal=False)


def set_sage_self_attention(flow_model, mode="auto"):
    """Select Sage self-attention with native cross-attention on one flow model.

    None restores native self-attention. Model weights and decoders are untouched.
    Sage uses approximate low-precision attention; validate on the target GPU.
    """
    return set_sage_attention(flow_model, mode)


def set_sage_attention(flow_model, mode="auto", include_cross=False):
    """Select flow attention kernels, with optional cross-attention acceleration."""
    if not isinstance(include_cross, bool):
        raise TypeError("include_cross must be bool")
    if mode not in (None, "auto", "fp16pv"):
        raise ValueError("mode must be None, 'auto', or 'fp16pv'")
    blocks = getattr(flow_model, "blocks", None)
    if blocks is None:
        raise TypeError("Expected a flow model with transformer blocks")
    attention = [getattr(block, "self_attn", None) for block in blocks]
    if not attention or any(module is None for module in attention):
        raise TypeError("Expected transformer blocks with self_attn modules")
    if any(getattr(module, "attn_mode", None) != "full" for module in attention):
        raise ValueError("Sage self-attention requires full-attention blocks")
    cross_attention = [getattr(block, "cross_attn", None) for block in blocks]
    if include_cross and any(module is None for module in cross_attention):
        raise TypeError("Expected transformer blocks with cross_attn modules")
    backend = None if mode is None else ("sage" if mode == "auto" else "sage_fp16pv")
    for module in attention:
        module.attn_backend = backend
    for module in cross_attention:
        if module is not None:
            if include_cross:
                module.attn_backend = backend
            elif getattr(module, "attn_backend", None) in ("sage", "sage_fp16pv"):
                module.attn_backend = None
    return len(attention)

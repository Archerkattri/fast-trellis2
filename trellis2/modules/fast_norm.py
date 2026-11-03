"""Optional compiled normalization leaves for inference on CUDA flow models."""

from functools import lru_cache
import torch
import torch.nn.functional as F


def _layer_norm(x, weight, bias, eps):
    return F.layer_norm(x.float(), x.shape[-1:], weight, bias, eps).to(x.dtype)


def _head_norm(x, gamma, scale):
    return (F.normalize(x.float(), dim=-1) * gamma * scale).to(x.dtype)


@lru_cache(maxsize=2)
def _compiled(kind):
    function = _layer_norm if kind == "layer" else _head_norm
    return torch.compile(function, fullgraph=True, dynamic=True)


def _check_inference(x):
    if not x.is_cuda:
        raise ValueError("Compiled flow normalization requires CUDA tensors")
    if torch.is_grad_enabled():
        raise ValueError(
            "Compiled flow normalization requires torch.no_grad() or inference_mode()"
        )


def compiled_layer_norm(x, weight, bias, eps):
    _check_inference(x)
    return _compiled("layer")(x, weight, bias, eps)


def compiled_head_norm(x, gamma, scale):
    _check_inference(x)
    return _compiled("head")(x, gamma, scale)


def set_compiled_norms(flow_model, enabled):
    """Enable compilation of normalization leaves; weights and blocks stay intact."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    selected = []
    for block in flow_model.blocks:
        for module in block.modules():
            name = type(module).__name__
            if name in ("LayerNorm32", "SparseMultiHeadRMSNorm", "MultiHeadRMSNorm"):
                if name == "LayerNorm32" and len(module.normalized_shape) != 1:
                    raise ValueError(
                        "Compiled flow LayerNorm requires one normalized dimension"
                    )
                selected.append(module)
    for module in selected:
        module.compile_norm = enabled
    return len(selected)

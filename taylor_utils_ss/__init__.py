# Fast-TRELLIS TaylorSeer cache for sparse-structure (SS) sampling.
#
# Instead of running the full model at every diffusion step, this cache
# approximates the predicted velocity at intermediate steps with a Taylor
# expansion around the most recent full step, and periodically refreshes with a
# full model pass. The cache schedule (first_enhance/end_enhance/interval) is
# parameterised so the sampler can adapt it to the active SS step count.
import math
from numbers import Integral
from typing import Dict

import torch


def taylor_schedule(num_steps):
    """Retain the SS warmup/interval policy, including a full terminal step."""
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    return {
        "first_enhance": max(2, round(num_steps / 3.0)) if num_steps < 20 else 2,
        "end_enhance": num_steps - 1,
        "taylor_interval": 4 if num_steps >= 20 else 3,
        "max_order": 1,
    }


def taylor_init(num_steps, taylor_interval=3, max_order=1, first_enhance=2, end_enhance=24):
    """Initialize TaylorSeer state for TRELLIS sparse-structure sampling."""
    taylor_state = {-1: {"final": {"final": {"final": {}}}}}
    taylor_dic = {
        "cache": taylor_state,
        "taylor_counter": 0,
        "taylor_interval": taylor_interval,
        "max_order": max_order,
        "first_enhance": first_enhance,
        "end_enhance": end_enhance,
        "taylor_enabled": True,
        "last_full_step": None,
    }
    current = {
        "activated_steps": [],
        "step": 0,
        "num_steps": num_steps,
    }
    return taylor_dic, current


def taylor_cal_type(taylor_dic, current, *, runtime=None, stage="sparse_structure", memory_mb=None):
    """Choose a step; the anchor is updated only after actual full computation."""
    first_steps = current["step"] < taylor_dic["first_enhance"]
    end_steps = current["step"] >= min(taylor_dic["end_enhance"], current["num_steps"] - 1)
    interval_hit = taylor_dic["taylor_counter"] >= taylor_dic["taylor_interval"] - 1
    anchor = taylor_dic["last_full_step"]
    force_full = first_steps or interval_hit or end_steps or anchor is None
    decision = None
    if runtime is not None:
        decision = runtime.decide(
            stage,
            horizon=0 if anchor is None else current["step"] - anchor,
            supported=taylor_dic["taylor_enabled"],
            memory_mb=memory_mb,
            force_full=force_full,
        )
        full = decision.method == "full"
    else:
        full = force_full or not taylor_dic["taylor_enabled"]

    if full:
        current["type"] = "full"
        taylor_dic["taylor_counter"] = 0
    else:
        taylor_dic["taylor_counter"] += 1
        current["type"] = "taylor"
    return decision


def taylor_record_full_step(taylor_dic, current):
    """Record an actual full evaluation without retaining its velocity."""
    current["activated_steps"].append(current["step"])
    taylor_dic["last_full_step"] = current["step"]


def derivative_approximation(taylor_dic: Dict, current: Dict, feature: torch.Tensor):
    # Called after the model returned its full velocity, including budget
    # fallbacks. Forecast velocities never become derivative anchors.
    taylor_record_full_step(taylor_dic, current)
    if len(current["activated_steps"]) < 2:
        difference_distance = 1.0
    else:
        difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    prev_module_cache = taylor_dic["cache"][-1][current["layer"]][current["module"]]
    current_key_factors = {0: feature}
    prev_key_cache = prev_module_cache.get("default", None) if prev_module_cache else None

    for order in range(taylor_dic["max_order"]):
        has_prev = prev_key_cache is not None and order in prev_key_cache
        is_within = current["step"] < (current["num_steps"] - taylor_dic["first_enhance"] + 1)
        if not (has_prev and is_within):
            break
        current_key_factors[order + 1] = (
            current_key_factors[order] - prev_key_cache[order]
        ) / difference_distance

    taylor_dic["cache"][-1].setdefault(current["layer"], {})
    taylor_dic["cache"][-1][current["layer"]][current["module"]] = {
        "default": current_key_factors
    }


def taylor_formula(taylor_dic: Dict, current: Dict, prev_v: torch.Tensor = None, beta=0.5):
    x = current["step"] - current["activated_steps"][-1]
    factors = taylor_dic["cache"][-1][current["layer"]][current["module"]]["default"]

    result = 0
    for order, value in factors.items():
        result = result + (1 / math.factorial(order)) * value * (x ** order)

    if prev_v is not None:
        result = beta * prev_v + (1.0 - beta) * result
    return result

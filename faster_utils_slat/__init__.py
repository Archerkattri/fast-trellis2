# Fast-TRELLIS SLaT acceleration: the "easy" delta-cache and learned-k skip
# logic for structured-latent (SLaT) sampling.
#
# The cache reuses a step's velocity delta when the input change since the last
# full step is small, and recomputes the full model output otherwise. The skip
# decision is calibrated by a learned gain `k` relating input change to output
# change. `num_steps` is supplied by the sampler so the skip logic tracks the
# actual step count of the active schedule.


def faster_init(num_steps):
    """Initialize Faster/token-cache state for TRELLIS SLAT sampling."""
    faster_state = {
        "k": None,
        "prev_x": None,
        "prev_v": None,
        "prev_prev_x": None,
        "error": None,
        "feature": None,
        "easy": None,
    }
    faster_dic = {
        "thresh": 1.0,
        "faster_counter": 0,
        "cache": faster_state,
        "faster_enabled": True,
        "first_enhance": 1,
    }
    current = {
        "type": None,
        "activated_steps": [],
        "step": 0,
        "num_steps": num_steps,
        "use_token": True,
        "is_token_active": False,
        "num_to_skip": 0,
        "cache_indices": None,
        "fast_update_indices": None,
    }
    return faster_dic, current


def faster_cal_type(faster_dic, current, input):
    faster_state = faster_dic["cache"]
    is_first_steps = current["step"] < faster_dic["first_enhance"]
    should_calc = True

    if not is_first_steps:
        has_history = (
            faster_state["prev_x"] is not None
            and faster_state["prev_v"] is not None
            and faster_state["prev_prev_x"] is not None
        )
        if has_history:
            delta_x = input - faster_state["prev_x"]
            input_change_mag = delta_x.abs().mean()

            if faster_state["k"] is not None:
                output_norm = faster_state["prev_v"].abs().mean() + 1e-6
                mag_error = faster_state["k"] * (input_change_mag / output_norm)
                faster_state["error"] += mag_error
                should_calc = faster_state["error"] >= faster_dic["thresh"]
            else:
                should_calc = True
        else:
            should_calc = True

    if should_calc:
        faster_state["error"] = 0
        current["type"] = "full"
        faster_dic["faster_counter"] = 0
        current["activated_steps"].append(current["step"])
    else:
        current["type"] = "faster"
        faster_dic["faster_counter"] += 1

    return should_calc

"""Named, explicit inference profiles for the image-to-3D pipeline."""

PROFILES = ("native", "sage", "sage-fast", "sage-fp16pv", "legacy-cache")
_CACHE_OPTIONS = {
    "cache_budget",
    "taylor_enabled",
    "faster_enabled",
    "cache_stage",
    "run_identity",
}


def configure_profile(pipeline, profile="native", pipeline_type=None):
    """Configure an idle pipeline between requests, preserving guidance/steps."""
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}; choose from {', '.join(PROFILES)}"
        )
    from . import samplers

    stages = ("sparse_structure", "shape_slat", "tex_slat")
    replacements = {}
    for stage in stages:
        previous = getattr(pipeline, f"{stage}_sampler")
        name = "FlowEulerGuidanceIntervalSampler"
        if profile == "legacy-cache":
            name += "_taylor" if stage == "sparse_structure" else "_faster"
        replacements[stage] = getattr(samplers, name)(sigma_min=previous.sigma_min)

    pipeline.enable_sage_attention(
        mode=None
        if profile in ("native", "legacy-cache")
        else ("fp16pv" if profile == "sage-fp16pv" else "auto"),
        pipeline_type=pipeline_type,
        include_cross=profile == "sage-fast",
        compile_norms=profile == "sage-fast",
    )
    for stage, sampler in replacements.items():
        previous = getattr(pipeline, f"{stage}_sampler")
        if hasattr(previous, "free_cache"):
            previous.free_cache()
        setattr(pipeline, f"{stage}_sampler", sampler)
        options = {
            key: value
            for key, value in getattr(pipeline, f"{stage}_sampler_params").items()
            if key not in _CACHE_OPTIONS
        }
        if profile == "legacy-cache":
            options[
                "taylor_enabled" if stage == "sparse_structure" else "faster_enabled"
            ] = True
        setattr(pipeline, f"{stage}_sampler_params", options)
    pipeline.enable_faster = profile == "legacy-cache"
    if hasattr(pipeline.shape_slat_sampler, "set_coords_scores"):
        pipeline.shape_slat_sampler.set_coords_scores(None)
    pipeline.inference_profile = profile
    return pipeline

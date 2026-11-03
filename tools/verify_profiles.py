"""Exercise real sampler construction and profile transitions without CUDA."""

import importlib
import os
from pathlib import Path
import sys
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SPARSE_CONV_BACKEND"] = "none"
for name, path in [
    ("trellis2", "trellis2"),
    ("trellis2.modules", "trellis2/modules"),
    ("trellis2.modules.sparse", "trellis2/modules/sparse"),
    ("trellis2.pipelines", "trellis2/pipelines"),
    ("trellis2.pipelines.samplers", "trellis2/pipelines/samplers"),
]:
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / path)]
    sys.modules[name] = package
basic = importlib.import_module("trellis2.modules.sparse.basic")
sys.modules["trellis2.modules.sparse"].SparseTensor = basic.SparseTensor
flow = importlib.import_module("trellis2.pipelines.samplers.flow_euler")
namespace = sys.modules["trellis2.pipelines.samplers"]
for name in [
    "FlowEulerGuidanceIntervalSampler",
    "FlowEulerGuidanceIntervalSampler_taylor",
    "FlowEulerGuidanceIntervalSampler_faster",
]:
    setattr(namespace, name, getattr(flow, name))
profiles = importlib.import_module("trellis2.pipelines.profiles")


class Owner:
    def __init__(self):
        for stage in ["sparse_structure", "shape_slat", "tex_slat"]:
            setattr(
                self, stage + "_sampler", flow.FlowEulerGuidanceIntervalSampler(1e-5)
            )
            setattr(
                self,
                stage + "_sampler_params",
                {
                    "steps": 12,
                    "guidance_strength": 7.5,
                    "cache_budget": object(),
                    "faster_enabled": True,
                },
            )

    def enable_sage_attention(self, **options):
        self.attention_options = options


owner = Owner()
original = owner.shape_slat_sampler_params
for profile in ["legacy-cache", "sage-fast", "native", "sage", "sage-fp16pv", "native"]:
    assert profiles.configure_profile(owner, profile) is owner
    assert owner.inference_profile == profile
    for stage in ["sparse_structure", "shape_slat", "tex_slat"]:
        options = getattr(owner, stage + "_sampler_params")
        assert options["steps"] == 12 and options["guidance_strength"] == 7.5
        assert "cache_budget" not in options
        if profile == "legacy-cache":
            assert (
                options[
                    "taylor_enabled"
                    if stage == "sparse_structure"
                    else "faster_enabled"
                ]
                is True
            )
        else:
            assert "taylor_enabled" not in options and "faster_enabled" not in options
            assert (
                type(getattr(owner, stage + "_sampler"))
                is flow.FlowEulerGuidanceIntervalSampler
            )
    assert owner.enable_faster == (profile == "legacy-cache")
    assert owner.attention_options["compile_norms"] == (profile == "sage-fast")
assert "cache_budget" in original, (
    "configuration mutated a caller-owned parameter mapping"
)
before = owner.shape_slat_sampler
try:
    profiles.configure_profile(owner, "invalid")
except ValueError:
    pass
else:
    raise AssertionError("Unknown profile accepted")
assert owner.shape_slat_sampler is before


class Flow:
    def __call__(self, x, t, cond, **kwargs):
        assert not any(key in kwargs for key in profiles._CACHE_OPTIONS)
        return x * 0.25 + t.reshape(-1, 1) / 1000


noise = torch.randn(1, 8)
options = dict(
    cond=torch.ones(1, 1),
    neg_cond=torch.zeros(1, 1),
    steps=4,
    guidance_strength=2.0,
    guidance_interval=(0.2, 0.8),
    verbose=False,
)
reference = flow.FlowEulerGuidanceIntervalSampler(1e-5).sample(Flow(), noise, **options)
actual = owner.shape_slat_sampler.sample(Flow(), noise, **options)
assert torch.equal(reference.samples, actual.samples)
print(
    "PASS: actual sampler/profile transitions, option isolation and native Euler parity"
)

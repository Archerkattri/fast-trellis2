"""Run actual sampler source with real SparseTensor arithmetic and a toy flow.

This isolates scheduling/CFG from model weights and convolution extensions.
It is NOT a TRELLIS.2 image-quality reproduction.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--hicache', type=Path, required=True)
parser.add_argument('--device', default='cpu')
parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
parser.add_argument('--phase', choices=['baseline', 'fixed'], default='baseline')
args = parser.parse_args()
sys.path[:0] = [str(args.repo.resolve()), str(args.hicache.resolve())]
os.environ['SPARSE_CONV_BACKEND'] = 'none'

def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module

# Skip eager model/renderer imports only; sampler, guidance and SparseTensor
# implementations are loaded directly from the repository unchanged.
for name, path in [('trellis2', 'trellis2'), ('trellis2.modules', 'trellis2/modules'),
                   ('trellis2.pipelines', 'trellis2/pipelines'),
                   ('trellis2.pipelines.samplers', 'trellis2/pipelines/samplers')]:
    package(name, args.repo / path)
sparse = package('trellis2.modules.sparse', args.repo / 'trellis2/modules/sparse')
basic = importlib.import_module('trellis2.modules.sparse.basic')
sparse.SparseTensor = basic.SparseTensor
samplers = importlib.import_module('trellis2.pipelines.samplers.flow_euler')
import torch

def features(x):
    return x.feats if isinstance(x, sparse.SparseTensor) else x

class ToyFlow:
    def __init__(self):
        self.times = []

    def __call__(self, x, t, cond=None, **kwargs):
        assert not any(k in kwargs for k in ('faster_enabled', 'taylor_enabled'))
        self.times.append(float(t[0]))
        # Smooth, time-dependent velocity; a stale v-x delta is not exact.
        return 0.25 * x + 0.1 * (float(t[0]) / 1000) ** 2 + 0.2 + float(cond or 0) * 0.01

records = []
for sparse_stage in [False, True]:
    kind = 'slat' if sparse_stage else 'ss'
    fast_cls = getattr(samplers, 'FlowEulerGuidanceIntervalSampler_' + ('faster' if sparse_stage else 'taylor'))
    for steps in ([12] if args.phase == 'baseline' else [1, 2, 3, 6, 12, 13, 25]):
        for guidance in ([1.0] if args.phase == 'baseline' else [1.0, 3.0]):
            torch.manual_seed(42)
            feats = torch.randn(64, 8, device=args.device, dtype=getattr(torch, args.dtype))
            coords = torch.stack([torch.zeros(64, device=args.device, dtype=torch.int32),
                                  torch.arange(64, device=args.device, dtype=torch.int32),
                                  torch.zeros(64, device=args.device, dtype=torch.int32),
                                  torch.zeros(64, device=args.device, dtype=torch.int32)], dim=1)
            noise = sparse.SparseTensor(feats, coords) if sparse_stage else feats
            reference = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
            fast = fast_cls(sigma_min=1e-5)
            options = dict(cond=1.0, neg_cond=-1.0, steps=steps, verbose=False,
                           guidance_strength=guidance, guidance_interval=(0.2, 0.8),
                           guidance_rescale=0.3, rescale_t=3.0)
            if sparse_stage:
                options['concat_cond'] = noise
            model_ref, model_fast = ToyFlow(), ToyFlow()
            expected = reference.sample(model_ref, noise, **options)
            actual = fast.sample(model_fast, noise, **options)
            error = (features(expected.samples) - features(actual.samples)).abs().max().item()
            full_steps = len(set(model_fast.times))
            record = dict(stage=kind, steps=steps, guidance=guidance,
                          full_steps=full_steps, model_calls=len(model_fast.times), max_error=error)
            records.append(record)
            if args.phase == 'fixed' and steps <= 12:
                assert full_steps == steps, record
                assert torch.equal(features(expected.samples), features(actual.samples)), record
                assert model_ref.times == model_fast.times, record
                for key in ['pred_x_t', 'pred_x_0']:
                    assert all(torch.equal(features(a), features(b)) for a, b in zip(expected[key], actual[key])), record
            if args.phase == 'fixed':
                # Explicit opt-in/out must remain available regardless of length.
                toggle = 'faster_enabled' if sparse_stage else 'taylor_enabled'
                model_off = ToyFlow()
                off = fast.sample(model_off, noise, **options, **{toggle: False})
                assert model_off.times == model_ref.times
                assert torch.equal(features(off.samples), features(expected.samples))
                model_on = ToyFlow()
                on = fast.sample(model_on, noise, **options, **{toggle: True})
                assert len(set(model_on.times)) <= steps
                assert torch.isfinite(features(on.samples)).all()
                assert fast.cache_dic is None
print(json.dumps(dict(device=args.device, dtype=args.dtype, torch=torch.__version__, phase=args.phase,
                      gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                      records=records), indent=2))

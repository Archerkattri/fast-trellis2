# Issue #4: short-schedule cache safeguard

**3 November update:** real pretrained inference and method comparisons are now
complete on a Windows RTX4060 compatibility runtime. See the
[GPU investigation](ISSUE-4-GPU-INVESTIGATION.md) for results and remaining limits.
The original record below describes the earlier safeguard validation.

## Report and scope

[Issue #4](https://github.com/Archerkattri/fast-trellis2/issues/4) includes the original
1024px stool input and a comparison of saved outputs. The comparison identifies
code `22d468fe2d34`, weights `af44b45f2e35`, seed 42, 12 updates per sampler,
`1024_cascade`, approximately 60k export faces, 1024px textures, `remesh=False`,
RTX 3090 24GB, PyTorch 2.7.1+cu128, SDPA + flex_gemm and `low_vram=True`.
It also states that both pipelines share local compatibility patches and export
code. Those patches are not attached to the issue.

The comparison shows holes in raw decoder geometry before export as well as
blotchy textured output. It reports 12 native texture evaluations versus 3
accelerated evaluations. Export settings alone cannot explain holes already
present in that raw geometry comparison.

## Confirmed scheduling behavior

The SLaT cache uses threshold 5.0. It needs three full evaluations to establish
its gain estimate, then can reuse `v = x + cached_delta` through all remaining
steps. There is no consecutive-skip bound or terminal full-evaluation guard.
The accumulated proxy is not a validated bound on model or generated-mesh error.
Fresh cascade runs do not enable token carving, so disabling carving alone does
not disable the relevant shape/texture delta cache.

An isolated regression using the actual sampler, guidance mixins and SparseTensor
source reproduced full SLaT evaluations at steps `[0, 1, 2]` out of 12. Its toy
velocity is smooth and depends on both state and time; cached deltas drift from
the stock Euler result. This reproduces the reported **scheduling pattern**, not
the reporter's image-conditioned model output. It does not identify how much of
the reported geometry damage comes from SS versus either shape pass.

## Mitigation

For schedules of at most 12 steps, default SS sampling now evaluates every step,
and default SLaT sampling delegates to the full Euler path with the same guidance.
No token is carved in that SLaT fallback. This intentionally gives up sampler
caching speedup at those defaults. The threshold is a conservative guard for the
reported short schedule, not a measured quality boundary or a guarantee for
longer schedules.

Explicit `taylor_enabled=True` and `faster_enabled=True` preserve access to the
previous approximate methods. An explicit SS cache budget selects its declared
policy unless Taylor is explicitly disabled. `False` requests full evaluation at
any length. The Taylor formula and SLaT cached-delta formula are unchanged.

## Validation (2 November 2026, America/New_York)

Windows CPU: Python 3.12, torch 2.13.0+cpu. School GPU: Windows 11 Enterprise,
RTX 4060 8188 MiB, isolated Python 3.10.11, torch 2.6.0+cu124. The shared runtime
checkout was `78c4fdfc7bfe0b6b614e12c36f9b5d9669a156b3`; its budget.py is unchanged
from the README's pinned `8fd47f9cfd2daca72a2679de69563a11b4f80f3b`.

`tools/verify_short_schedule.py` loads actual sampler/guidance/SparseTensor source,
skipping eager package-root model and renderer imports. Sparse storage uses the
`none` convolution backend. The flow is synthetic: no learned model, attention,
sparse convolution, decoder, image preprocessing, or mesh exporter runs.

- Before: 12-step GPU float32 SLaT used 3 full evaluations, with max absolute
  final feature error 0.17551255 against full Euler on the toy flow. SS used 7
  full evaluations, with error 0.01082587. These values are synthetic feature
  differences, not image/mesh quality metrics.
- After: full and guided sampling checked at 1, 2, 3, 6, 12, 13 and 25 steps,
  with rescale_t=3, guidance interval [0.2, 0.8], and CFG rescale 0.3.
  For defaults at up to 12 steps, every intermediate prediction and the final
  sample were exactly equal to stock Euler. Model-call timesteps matched too.
- Explicit disabled mode matched stock at all tested lengths. Explicit enabled
  mode remained callable and produced finite results. Reusing a sampler across
  default/disabled/enabled calls passed the same checks.
- The checks passed on CPU float32 and GPU float32 and bfloat16.

Example after installing the documented runtime and torch/numpy/tqdm/easydict:

```bash
python tools/verify_short_schedule.py --repo . --hicache /path/to/hicache-plus-plus --phase fixed --device cuda --dtype bfloat16
```

`--phase baseline` is for the original source checkout and prints the 12-step
toy-flow comparison without requiring the new `faster_enabled` option.

## Full image reproduction remains outstanding

The original attachments were downloaded and inspected. The available GPU is an
8GB Windows RTX 4060; upstream documents Linux and at least 24GB VRAM. WSL and
Virtual Machine Platform were disabled. Both normal WSL installation and the
official 3.0.3 MSIX installation were attempted. DISM failed with error 740
(elevation required), and package registration failed with 0x80073D28 (administrator
privileges required to install the packaged service). Native Windows CUDA sampler
checks succeeded, but this is not the dependency-complete TRELLIS.2 pipeline.

No stool model inference, new GLB, image-quality recovery, acceleration-quality
claim, or performance result is established here. Issue #4 should remain open for
end-to-end validation on the reporter's environment or a supported machine.

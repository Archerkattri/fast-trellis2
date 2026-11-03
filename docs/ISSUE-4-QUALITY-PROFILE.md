# Issue #4: cause and resolution

The quality-preserving acceleration path uses fresh native Euler evaluations
with faster attention/normalization kernels. The `sage-fast` profile is
1.45–1.47× faster in sampling on the uploaded stool, two seeds and the available
RTX 4060 runtime. It does not achieve the aggressive cache's 58s latency at
native quality. See [complete results and figures](BENCHMARKS.md).

## Why the initial port lost quality

Auditing original TRELLIS, Fast-TRELLIS and TRELLIS.2 found no new sign error in
the native Euler velocity conversion. The inherited reuse rule
`v_new = x_new + (v_anchor - x_anchor)` assumes a particular state response; its
gain/history proxy is not a calibrated bound on time-dependent error.

TRELLIS.2 adds state-dependent guidance rescaling and separate shape/texture
flows. Reusing the final guided velocity skips fresh rescaling, while shape
errors also change the conditioning for texture. Guidance-branch refreshes
improved geometry in an ablation but did not recover the full native output.
Sparse-structure forecasting could increase downstream token counts and decoding
work, reducing the intended speed benefit. This is a transfer limitation of the
approximate method, not evidence that the original cache can be made lossless
by reversing an arithmetic sign.

The source audit also found avoidable FFT score work on cascade runs and stale
scores on reused pipelines. These were bypassed/cleared, with coordinate and RNG
regression checks. The earlier short-schedule safeguard restores full evaluation
by default at 12 steps or fewer; `legacy-cache` explicitly opts into approximation.

## What changed

`sage` changes final-flow self-attention while preserving all native steps.
`sage-fast` also accelerates cross-attention and compiles normalization leaves.
Coarse cascade flows retain native kernels. These methods accelerate each fresh
evaluation, without velocity forecasting or changed model weights.

The public kernel implementation matched both measured seed-42 shape/texture
latents exactly in a repeat; sampling took 133.84s. The release CLI separately
completed inference and textured GLB export. Geometry and base color remain close
to native; some metallic/alpha attributes deviate more, so retain plain `sage`
for a more conservative accelerated choice. Neither profile is lossless.

## Evidence and remaining validation

The [benchmark protocol](../benchmarks/README.md) records pinned sources, weights,
hashes, timing scopes, actual four-way renders and all result data. It includes
new original TRELLIS / Fast-TRELLIS learned runs with matched V1 settings.
Cross-generation models/resolutions differ; compare speed within each pair.

The [original detailed quality record](archive/ISSUE-4-QUALITY-PROFILE.md) and
[full experiment log](archive/ISSUE-4-GPU-INVESTIGATION.md) preserve the ablations,
failed candidates and source audit references. They are historical evidence;
current profile/CLI instructions are in [usage](USAGE.md).

The original reporter's RTX 3090/SDPA stack and a clean Linux source install
remain untested. Single-image/two-seed fidelity is not a dataset-wide guarantee,
ground-truth accuracy result or a claim of peer-reviewed publication readiness.

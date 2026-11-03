# Issue 4: real-model acceleration investigation, 3 November 2026

## Outcome

The uploaded stool input now runs through the learned models and raw mesh decoder
on the available RTX4060. The original acceleration reproduces visibly damaged
geometry and mottled texture compared with full Euler in this runtime. No tested
replacement has demonstrated both native quality and the original speed. The
short-schedule full-evaluation safeguard remains a mitigation, not the requested
quality-preserving acceleration fix. Issue4 remains unresolved.

## Environment and controls

The school host has Windows11, RTX4060 8GB and16GB RAM. WSL installation requires
administrator access, so a separate native Windows environment was installed:
Python3.11.9, torch2.8.0+cu128, FlashAttention2.8.3, and the prebuilt packages and
8GB decoder offloading from [the StableProjectorz fork](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz)
at `d5d38f1e033a881e2c00e3137822ef8aa4e193de`. The release ZIP SHA256 was
`62be7caefaf12e396763dfec4b5e688fdd83c9450920090dfdb8082bd43811be`.

TRELLIS.2 weights were pinned to the issue's `af44b45f2e35` revision. Downloaded
weight files were checked against HuggingFace LFS SHA256 metadata. The stool input,
seed42,12steps,1024_cascade and low-VRAM mode were shared across comparisons.
The compatibility runtime converts BF16 model weights to FP16 and uses flash
attention rather than the reporter's patched SDPA. This is a controlled
reproduction of the failure class, not bitwise reproduction of the RTX3090 run.

One model stage was loaded at a time. Meta construction avoided discarded random
parameter initialization; each of the four flow models matched its original
loader's saved first forward exactly, and the subsequent full native run matched
all final reference shape/texture coordinates and features exactly. Missing
learned checkpoint tensors were fatal; only deterministic rotary phase buffers
were reconstructed. CPU/CUDA RNG states were preserved during model loading.

## Mechanisms found

1. The SLaT rule `v_new = x_new + (v_anchor - x_anchor)` assumes identity state
   response and omits explicit time dependence. A constant-velocity counterexample
   drifts under this rule despite a zero velocity-change gain proxy. The rule is
   exact for a constant `v-x` delta, a narrower condition.
2. Shape guidance changes from7.5 to1 at step9. Legacy full evaluations stop at
   steps0,1,2 on this run, so guided predictions are reused across that transition.
   A guidance-boundary counterexample confirms this failure mechanism. Texture
   guidance is1 throughout, so this cannot explain the texture failure alone.
3. Skipped predictions overwrite history used by the gain proxy. Its state-change
   estimate is not a measured bound on skipped-step prediction error. There is no
   consecutive-skip bound or forced terminal full SLaT evaluation.
4. SS forecasting is independently damaging: the SS-only ablation below uses full
   shape/texture evaluations yet still changes geometry substantially. Changed
   token/voxel workload can also make downstream sampling and decoding slower.

These are source observations, controlled counterexamples and stage ablations;
they do not allocate an exact percentage of the reporter's damage to each cause.

## Method comparison

Full learned-model x/v/t/guidance traces screened delta, velocity, clean-sample,
noise, linear-velocity-in-rescaled-time, linear-clean-sample, Hermite and DMD
forecasts at3/4/5/6full-step budgets. Clean-sample/noise holds reduce algebraically
to velocity reuse on their own Euler trajectories. Replay used integrated candidate
states with prescribed reference velocities; it only screened candidates and did
not substitute for closed-loop inference.

Actual closed-loop runs included six forecast variants, signed scalar state-gain
and state/time fits, independent stage ablations and a warmup hybrid, plus full
and original-cache controls. Full-call placement was optimized on this stool's
reference trace, so these are exploratory results and may overfit this input.
All candidates were decoded. Large raw meshes exceeded the rasterizer triangle
limit in several cases; saved meshes were rendered at a shared60kface target,
without remeshing, rather than repeating generation.

| Configuration | Sampler seconds | Generation seconds | Raw vertex Fscore@.005 | Whole-view color PSNR | Upper-frame color PSNR |
|---|---:|---:|---:|---:|---:|
| Original acceleration |58.37|126.45|.359|15.41|19.75|
| Shifted delta refresh |65.12|149.14|.637|12.49|24.76|
| Velocity reuse |65.31|156.60|.808|16.82|23.61|
| Linear velocity |65.85|159.04|.810|9.42|20.91|
| Linear clean sample |66.34|181.43|.779|13.60|19.88|
| Hermite |65.72|177.50|.747|17.71|21.10|
| DMD on rescaled-time grid |67.18|166.15|.808|16.82|23.59|
| Scalar state response |65.28|176.33|.784|13.11|20.79|
| Joint state/time response |66.08|186.46|.804|13.23|19.76|
| SS-only Taylor |218.62|371.85|.608|13.80|24.11|
| Texture-only6evals |165.91|247.32|1.000|25.78|30.56|
| Shape-only6evals |135.03|218.68|.937|11.64|22.06|
| Warmup hybrid |104.20|186.73|.937|10.98|22.16|
| Full Euler |195.19|271.24|1.000|33.25|34.24|

Sampler times synchronize CUDA around each stage and exclude model loading.
Generation includes lazy stage loading and raw decoding, but excludes rendering
and GLB export. These are single exploratory timings, not repeated speed claims.
The main forecast comparisons keep SSfull and shape/texture model-call budgets
equal to the legacy cache; this adds SS cost and isolates SLaT behavior.

The warmup hybrid uses full SS, velocity reuse in shape at steps0,1,4,7,9,11,
and time-linear velocity texture forecasts at0,1,4,7,10,11. Shape-only has the same
shape policy and full texture; texture-only has full SS/shape. SS-only uses the
unchanged Taylor formula at0,1,2,5,8,10,11, resetting at the guidance change.

## Measurement limits

Fscore compares sampled raw vertices with nearest reference vertices at threshold
.005; it is reference fidelity, not ground-truth surface accuracy. No similarity
alignment was applied. Six matched-camera base-color views compare common geometry
pixels after60kface decimation. A generated floor plane changes color and can
dominate whole-view PSNR, so the upper65% frame crop is also reported. That crop
is an explicit diagnostic, not semantic object segmentation. All original scores
were retained. Decimation itself varies slightly: identical native raw outputs
score33.25dB whole-view rather than infinite PSNR. SSAO shading is not used for
texture ranking. Raw boundary/nonmanifold counts are nonmonotonic diagnostics and
were not used to choose a winner.

No tested candidate meets both near-native fidelity and original latency. Velocity
reuse is the strongest fast geometry/texture compromise among these experiments;
the warmup hybrid improves geometry more at a substantial latency cost. Neither
is promoted to a production default or described as resolving the issue. No
held-out seed/image validation or finalist timing repetitions were run because
none met the requested acceptance condition on the reproduction input.

Raw traces, configs, logs, decoded tensors and evaluation scripts are retained in
the research workspace `projects/personal-projects/fast-trellis2-issue-4` and on
the school host under `C:\Users\kr881817\fast-trellis2-issue-4`. This document is
an investigation record, not a self-contained supported Windows installation.

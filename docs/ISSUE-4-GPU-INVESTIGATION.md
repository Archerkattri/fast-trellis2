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

## Internal-feature follow-up, 3 November 2026

The next experiment reused the residual of the middle transformer blocks rather
than forecasting the final velocity. Input projection, timestep embedding, first
block, last block and output projection ran at every step. Each guidance branch
had an independent cache; a guidance transition discarded the cache and forced
refresh. Coordinate support was checked before reuse. Time modulation inside
the skipped middle blocks was not recalculated.

Full-refresh hook controls matched the saved FP16 native final latents exactly
for both shape1024 and texture1024 (maximum difference zero). Candidate runs
used their own evolving states, not teacher-state replay.

| Stage candidate | Middle refresh steps | Stage seconds | Final latent NRMSE | Decoded geometry Fscore |
|---|---|---:|---:|---:|
| Shape full control |all12|105.77|0|1.000|
| Shape residual hold4 |0,4,9,11|35.53|.734|.959|
| Shape residual hold6 |0,1,4,7,9,11|54.18|.383|.998|
| Shape linear residual4 |0,1,9,11|35.90|.941|.915|
| Texture full control |all12|60.90|0|native shape held fixed|
| Texture residual hold3 |0,7,11|18.75|.360|native shape held fixed|
| Texture residual hold6 |0,1,4,7,10,11|32.54|.120|native shape held fixed|
| Texture linear residual3 |0,1,11|18.53|.353|native shape held fixed|

Shape stage ablations start from the saved native512 output/support and native
1024 noise. Texture stage ablations hold native geometry fixed. The linear
candidate uses two true residual anchors in actual time, with extrapolation
ratio bounded to0..1.5. Its refresh schedules differ from the hold candidates;
these rows do not isolate forecast basis independently of refresh placement.

The texture hold6 ablation scored32.02dB whole-view and32.53dB fixed-crop PSNR;
hold3 scored21.81/28.67dB and linear3 scored23.60/25.16dB. Native render comparison
noise is33.25/34.24dB. Large shape latent error did not imply equally large mesh
error: hold6 reached raw Fscore.997877, Chamfer.00051715 and maskIoU.99032.
Decoded geometry and downstream texture therefore remain essential checks.

### Complete pipeline check

The combined FP16 candidate kept full sparse structure and full shape512, used
shape1024 hold6 and texture1024 hold6, and generated texture conditioned on its
own new shape. It retained all12 Euler steps; only middle-block computation was
reused. Shape1024 executed the middle at10 of21 guidance-branch invocations;
texture executed it at6 of12. It is not a reduction to six model invocations.

| Configuration | Sampler seconds | Generation seconds | Raw geometry Fscore | Whole/crop PSNR |
|---|---:|---:|---:|---:|
| Original acceleration |58.37|126.45|.359|15.41/19.75|
| Full native |195.19|271.24|1.000|33.25/34.24|
| Internal-feature hold6 |116.08|196.70|.998|27.43/28.44|

The combined output has11.649M faces, Chamfer.00051715 and maskIoU.99041. Its
individual stage times were14.48s SS,15.47s shape512,53.88s shape1024 and32.25s
texture1024; sampling wall162.73s plus33.97s raw decoding gives196.70s generation.
These are exploratory same-input timings, not held-out or repeated end-to-end
benchmarks. Stage timings agree closely with the isolated shape/texture runs.

This is a substantial fidelity improvement over the old acceleration, but its
sampler takes roughly twice the old acceleration's time. The original latency
requirement is still unmet. The lower whole/crop texture score in the combined
run includes geometry displacement and changed conditioning as well as texture
approximation; the isolated texture score must not be presented as combined
pipeline fidelity. No production default was changed.

To separate some of those effects, full texture inference was rerun using the
candidate's exact shape, original image conditioning and original texture noise.
The cached/full texture comparison on that same geometry scored29.84dB whole-view
and29.67dB fixed-crop PSNR, with raw geometry identical (Chamfer zero). Original
native-reference metrics were retained separately. Geometry alignment explains
part of the combined comparison, but a texture approximation difference remains.
This is not evidence of lossless texture reuse. No held-out image/seed validation
or repeated end-to-end timing was claimed, because original-speed acceptance
still failed on the reproduction input.

An independent rerun of cached texture with the candidate shape reproduced the
combined pipeline's texture latent exactly (coordinates equal, maximum feature
difference zero), checking that the conditioning/noise reconstruction used for
the stage comparison matches the combined run.

### Rejected FP8 implementation

An isolated RTX4060 kernel probe supported E4M3 FP8 with FP16 output. For a
16384x1536 by1536x6144 matrix multiplication, FP16 took10.14ms, prequantized FP8
5.02ms, and FP8 including activation quantization6.33ms. This did not translate
into real-model acceleration. Quantizing the210 large transformer-block linears
gave full texture62.16s versus60.90s FP16, and cached texture32.96s versus32.54s.
The cached FP8 shape run produced non-finite output. This implementation was
rejected; its kernel result is not an inference speedup claim. These results do
not establish the behavior of other FP8 implementations or other GPUs.

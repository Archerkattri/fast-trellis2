# Methods and validation scope

## Native and accelerated inference

TRELLIS.2 generates sparse structure, coarse/refined shape latents and material
latents, then decodes geometry and PBR attributes. The native Euler sampler
evaluates the learned velocity on every scheduled step, including current
guidance and state-dependent guidance rescaling.

The Sage profiles preserve those evaluations. `sage` changes self-attention on
the final shape/texture flows. `sage-fast` additionally changes cross-attention
and compiles LayerNorm/head-normalization leaves. Coarse cascade flows remain
native. There is no model retraining, checkpoint quantization or denoising-output
cache in these profiles. SageAttention internally uses low-precision attention
arithmetic, and compiler fusion can change floating-point rounding.

The experimental legacy port reproduces the Fast-TRELLIS velocity-delta cache,
Taylor SS forecast and token carving. Its reuse rule assumes a particular state
response and its gain/history proxy is not a validated error bound. A short
schedule can reuse stale outputs across time/guidance changes. Separate texture
conditioning makes shape-latent changes consequential even when meshes look close.

## Image and runtime controls

All comparison images derive from the original stool attachment in issue #4.
The saved preprocessed RGB image is shared. V1 uses its native DINOv2 conditioner;
V2 uses DINOv3. Native and fast variants within a generation share weights, seed,
conditioning and sampler configuration. V1 uses 25 SS/SLaT steps and its 256-grid
mesh decoder; V2 uses 12 steps and `1024_cascade`. Do not interpret the different
native latencies as an acceleration comparison across model generations.

The available host is Windows/RTX 4060 8GB. Stage loading and compatibility
packages make these experiments possible. This differs from the original
reporter's RTX 3090/SDPA stack. Model revisions, source revisions, calculated
hashes, timing data and render settings are in [benchmarks/issue4](../benchmarks/issue4/).

## Timing

Sampler times synchronize CUDA around learned sampling stages and exclude model
loading, image conditioning and decoding. V2 generation includes lazy loading
and raw decoding, but excludes initial preprocessing, rendering and GLB export.
V1 generation excludes its cached DINO conditioning and includes its own loading,
decoding and artifact save. Generation totals are meaningful within each pair;
their scopes are not identical across V1/V2. CLI metadata states its own scope.

V2 fast measurements use warmed compiler caches. First-use compilation is
reported separately. No cache was cleared to manufacture a performance result.
Single-image runs do not estimate dataset-wide speed or uncertainty.

For times `T_native` and `T_fast`:

```text
speed multiplier = T_native / T_fast
percent faster   = 100 * (T_native / T_fast - 1)
time reduction   = 100 * (1 - T_fast / T_native)
```

Thus 1.45× speed is about 45% faster, or about 31% less waiting time.

## Fidelity and figures

V2 geometry fidelity samples raw mesh vertices and measures nearest-reference
distance at threshold 0.005 without alignment. It is not a surface-based
ground-truth metric. Six-view base-color PSNR uses matched cameras and 60k-face
render proxies; decimation introduces noise even for identical native outputs.
The upper-65% crop is a fixed diagnostic, not semantic foreground segmentation.
PBR attributes are also compared on exact shared voxel coordinates, with coverage
and maximum errors reported. Unshared coordinates are excluded from that check.

The release four-way figures are freshly rendered from **raw, undecimated**
meshes, common cameras and common clay lighting. Appearance is interpolated
vertex color for V1 and volumetric base color for V2, not a shared texture decoder
or a photorealistic PBR comparison. These raw figure renders are distinct from
the earlier 60k-proxy PSNR evaluation. SVG/PDF exports preserve figure labels;
their scene pixels remain rasterized model renders.

## Software checks

CPU checks exercise actual sampler source/profile transitions, attention packing,
batch isolation, normalization selection and cache/score lifecycle without model
weights. Their bootstrap skips eager CUDA extension imports. Real CUDA tests
exercise Sage/compiled normalization. Learned-model generation, raw decoding,
renders, material diagnostics and public-implementation latent parity provide
separate model-level evidence. Passing the CPU checks alone does not establish
GPU compatibility or output quality.

Historical investigation records are preserved under [archive/](archive/).
Unsupported historical benchmark claims are not used in the current release.

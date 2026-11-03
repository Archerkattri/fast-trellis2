# Issue #4 reproduction data

This directory contains release measurements and provenance, not pretrained
weights. Read [methods](../docs/METHODS.md) and [results](../docs/BENCHMARKS.md)
before reusing the numbers. The rendered input is the stool attached to
[issue #4](https://github.com/Archerkattri/fast-trellis2/issues/4).

## Evidence

- `issue4/results.json`: four models, pinned sources/weights, seed, input hash and times.
- `issue4/v1-pair-metrics.json`: V1 geometry fidelity and input/conditioning-weight hashes.
- `issue4/v1-model-sources.json`: downloaded checkpoint files and SHA256 digests.
- `issue4/render-manifest.json`: cameras, renderer, resolution and raw mesh counts.
- `issue4/cli-validation.json`: real CLI/export metadata, with private paths removed.
- `issue4/v2-pairs.json`: both V2 seed pairs and fidelity diagnostics.
- `issue4/v2-materials.json`: shared-coordinate PBR diagnostics.

The release evidence ZIP adds all 48 raw color/clay renders and PNG/SVG/PDF
figures. Model weights and the reporter's input attachment are not redistributed
in the bundle. Obtain the input from the issue and verify its SHA256. The saved
preprocessed RGB hash is recorded separately; a different crop/background changes
the inference and invalidates a strict matched comparison.

## Rebuild figures

```bash
python -m pip install matplotlib==3.11.1 pillow==12.3.0
python benchmarks/build_figures.py --renders /path/to/release-comparison
```

Figures are written to `dist/figures` as PNG, SVG and PDF. To package a new
comparison bundle, pass that directory with `build_bundle.py --figures dist/figures
--renders /path/to/release-comparison --output dist/evidence.zip`.

The plotting script reads recorded numbers and actual rendered images. It does
not generate or edit model outputs. Scene pixels remain raster in SVG/PDF.

## Repeat inference

For V2, install the source dependencies, obtain the pinned model revision listed
in `results.json` (for example with `huggingface-cli download --revision`), and
pass that local model directory to `tools/generate.py --model`. Run native and
Sage profiles with the same preprocessed image, seed and pipeline. Run a warmup
before measuring compiled profiles and retain the first-use timings separately.
The public CLI reports complete generation/export time; sampler numbers require
CUDA-synchronized boundaries around each sampling call as described in methods.

For V1, use the two pinned upstream source commits and the same pinned
`microsoft/TRELLIS-image-large` weights. Use 25 SS and SLaT steps in both variants,
guidance strength 5, interval [.5, 1], time rescale 3, seed 42 and mesh format.
Fast-TRELLIS's example defaults differ; do not compare those defaults against a
25-step native run. Use each generation's own conditioner. Sample 100k raw
vertices per mesh and compute bidirectional nearest-neighbor fidelity at .005,
without alignment, for the V1 diagnostic.

The observed Windows setup used separate spconv-cu126 2.3.8 packages for V1,
stage-wise residency, local pinned DINOv2 code/weights, the upstream pure-Python
Kaolin tensor validator and an explicit unavailable-rembg stub (already
preprocessed inputs only). V2 used the community Windows runtime at the recorded
commit plus this project's acceleration modules. A weighted-gather compatibility
fallback was used for rendering/export. These adapters are environment-specific;
the source install is not a turnkey reproduction of that Windows runtime.

Render all raw meshes with `MeshRenderer`, 512px, ssaa 1, near .1, far 10;
yaw = pi/4 + view*pi/3, pitch .3, radius 2, FOV 40 degrees, view = 0..5.
Use V1 RGB vertex attributes or V2 base-color volume, mask-composite on .96 gray.
Clay uses .76 gray times (.35 + .65 * abs(dot(normal, normalized[.3,-.6,.75]))).
The exact render source is supplied in `benchmarks/render_meshes.py`.

Do not report these single-image results as a dataset-wide speed or quality
improvement. No ground-truth 3D reference or reporter RTX 3090 validation exists.

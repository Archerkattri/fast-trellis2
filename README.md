<p align="center">
  <img src="assets/banner.png" alt="fast-trellis2" width="680">
</p>

# fast-trellis2

Faster image-to-3D generation with [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2), using SageAttention and native sampling. The original Fast-TRELLIS cache port remains available as an experimental profile.

[Install](docs/INSTALLATION.md) · [Usage](docs/USAGE.md) · [Benchmarks](docs/BENCHMARKS.md) · [Releases](https://github.com/Archerkattri/fast-trellis2/releases)

## Results

On the uploaded [issue #4 stool](https://github.com/Archerkattri/fast-trellis2/issues/4), RTX 4060 8GB, two seeds:

| Profile | Sampling | Generation¹ | Geometry Fscore² |
|---|---:|---:|---:|
| Native TRELLIS.2 | 195–199s | 271–290s | reference |
| `sage` | 154–156s | 228–244s | .9999–1.0000 |
| `sage-fast` | **135s** | **214–224s** | **.9998–1.0000** |

`sage-fast` gives **1.45–1.47× sampling speed**, or **31–32% less waiting time**, while retaining all 12 sampling steps. These warm-cache results cover one input. Some material channels change more than with plain `sage`; neither profile is lossless.

¹ Stage loading and raw decoding included; initial preprocessing, rendering and GLB export excluded. ² Vertex fidelity against the matched native mesh at distance .005, without alignment.

![TRELLIS, Fast-TRELLIS, TRELLIS.2 and fast-trellis2 on the same input](docs/images/four-way-appearance.png)

[Full comparison, methods and raw data](docs/BENCHMARKS.md). V1 and V2 use different models and resolutions; speed comparisons are within each native/fast pair.

## Install

Linux, an NVIDIA GPU and a matching CUDA Toolkit are required for the standard source install. Upstream recommends at least 24GB VRAM. The measured 8GB Windows setup used a separate compatibility runtime; see [platform details](docs/INSTALLATION.md).

```bash
git clone --recursive https://github.com/Archerkattri/fast-trellis2.git
cd fast-trellis2
bash setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec
conda activate trellis2
python -m pip install sageattention==2.2.0 --no-build-isolation
```

## Generate

```bash
python tools/generate.py --profile sage --image input.png --output result.glb
python tools/generate.py --profile sage-fast --image input.png --output result-fast.glb
```

The CLI writes a textured GLB and a `.run.json` with settings, device and timings. Defaults: `1024_cascade`, native step counts, about 60k faces, 1K textures, no remeshing. Use a GLB viewer that supports WebP textures.

| Profile | Behavior |
|---|---|
| `native` | Native Euler sampling and attention; default |
| `sage` | Native sampling, final-flow Sage self-attention |
| `sage-fast` | Also accelerates cross-attention and compiles normalization |
| `sage-fp16pv` | Sage self-attention with FP16 value products |
| `legacy-cache` | Approximate forecasting/token caching; can damage output |

`sage-fast` compiles kernels on first use. Choose `sage` for a more conservative speed/quality tradeoff. [CLI options and Python API](docs/USAGE.md) · [Troubleshooting](docs/TROUBLESHOOTING.md).

## Credits

Models and base pipeline: [TRELLIS.2](https://github.com/microsoft/TRELLIS.2). Legacy cache method: [Fast-TRELLIS](https://github.com/wlfeng0509/Fast-SAM3D/tree/Fast-TRELLIS). Attention kernels: [SageAttention](https://github.com/thu-ml/SageAttention). This repository provides integration, profiles and validation.

MIT source license; weights and dependencies retain their own terms. [License](LICENSE) · [Third-party notices](NOTICE) · [Citation](CITATION.cff) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md).

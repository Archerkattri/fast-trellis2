# fast-trellis2

Measured inference profiles for [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2): native sampling, SageAttention acceleration, and an explicitly experimental Fast-TRELLIS cache port.

[Installation](docs/INSTALLATION.md) · [Usage](docs/USAGE.md) · [Benchmarks](docs/BENCHMARKS.md) · [Methods](docs/METHODS.md) · [Release notes](CHANGELOG.md)

## Results

On the uploaded issue #4 stool, an RTX 4060 8GB compatibility runtime, and two seeds:

| Profile | Sampling | Generation¹ | Geometry fidelity² |
|---|---:|---:|---:|
| Native TRELLIS.2 | 195–199s | 271–290s | reference |
| `sage` | 154–156s | 228–244s | 0.9999–1.0000 |
| `sage-fast` | **135s** | **214–224s** | **0.9998–1.0000** |

`sage-fast` uses 31–32% less sampling time, equivalent to **1.45–1.47× speed**, versus native TRELLIS.2. It retains all 12 sampling steps. These are warm-compiler-cache measurements on one image, not a dataset-wide benchmark. Some material attributes differ more than with `sage`; neither profile is lossless.

¹ Includes stage loading and raw decoding; excludes initial image preprocessing, rendering and GLB export. The CLI records its own complete timing scope. ² Raw vertex Fscore at distance 0.005 against the matched native mesh, without alignment; not ground-truth accuracy.

![Four model families on the same input](docs/images/four-way-appearance.png)

The [four-way comparison](docs/BENCHMARKS.md#four-way-comparison) includes actual TRELLIS, Fast-TRELLIS, TRELLIS.2 and fast-trellis2 outputs. V1 and V2 use different models, native resolutions and sampling schedules. Speedups are reported within each pair.

## Install

The standard source installation targets Linux with an NVIDIA GPU and CUDA. Upstream TRELLIS.2 recommends at least 24GB VRAM. Our 8GB Windows validation used a separate compatibility runtime; it is not a general 8GB installation guarantee.

```bash
git clone --recursive https://github.com/Archerkattri/fast-trellis2.git
cd fast-trellis2
bash setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec
conda activate trellis2
# Optional, in an environment matching SageAttention's CUDA requirements:
python -m pip install sageattention==2.2.0 --no-build-isolation
```

See [installation and platform support](docs/INSTALLATION.md) before building extensions. Model weights download separately under their upstream terms.

## Generate a GLB

```bash
python tools/generate.py --profile sage --image input.png --output output.glb
python tools/generate.py --profile sage-fast --image input.png --output output-fast.glb
```

`sage` is the conservative accelerated choice. `sage-fast` adds cross-attention acceleration and compiled normalization; first-use compilation adds latency. Use `native` for the reference implementation. Export defaults to approximately 60k faces, 1K textures and no remeshing.

Each command writes a `.run.json` sidecar with the profile, seed, settings, device and synchronized wall times. Existing output files require `--overwrite`. `example.py` and `example_faster.py` remain compatible entry points; `--attention` is retained as an alias.

```python
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.configure_profile("sage", pipeline_type="1024_cascade")
pipeline.cuda()  # cuda() returns None; do not chain into assignment.
mesh = pipeline.run(Image.open("input.png"), seed=42)[0]
```

## Profiles

| Name | Sampling | Attention / normalization | Use |
|---|---|---|---|
| `native` | Full Euler | Native | Reference and maximum fidelity |
| `sage` | Full Euler | Final-flow Sage self-attention | Conservative acceleration |
| `sage-fast` | Full Euler | Final-flow Sage self/cross attention, compiled norms | More speed, more material deviation |
| `sage-fp16pv` | Full Euler | Sage self-attention with FP16 value product | Alternative attention precision |
| `legacy-cache` | Approximate Taylor / velocity-delta caching | Native | Experimental reproduction only |

The old cache can be faster while producing holes and blotchy textures. It is not the recommended quality profile. The cache classes themselves default to full evaluation on schedules of 12 steps or fewer; selecting `legacy-cache` explicitly opts into approximation.

## Validation and limits

CPU regression tests, real CUDA attention/normalization checks, actual pretrained inference and paired renders support the published results. [Methods](docs/METHODS.md) defines what those checks establish. The reporter's RTX 3090/SDPA environment remains untested. Compiled kernels require a working CUDA/Triton toolchain; performance varies by device and startup state.

This release supplies source, profiles, reproducibility data and a CLI. It does not certify every CUDA combination, provide a hosted service, or establish publication acceptance. See [troubleshooting](docs/TROUBLESHOOTING.md), [contributing](CONTRIBUTING.md), and [release validation](docs/RELEASING.md).

## Attribution and citation

TRELLIS.2 and its checkpoints are Microsoft's work. The legacy cache method comes from [Fast-TRELLIS](https://github.com/wlfeng0509/Fast-SAM3D/tree/Fast-TRELLIS). Sage kernels come from [SageAttention](https://github.com/thu-ml/SageAttention). This repository contributes integration, safeguards, profiles, tooling and the reported validation.

MIT source license; dependencies and model weights retain their own licenses. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff). Historical development notes and unverified prior benchmark claims are preserved in [the archive](docs/archive/), outside the current release evidence.

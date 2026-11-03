# Installation and platform support

## Requirements

The standard source path follows upstream TRELLIS.2: Linux, NVIDIA CUDA and
enough GPU memory for the selected resolution. Upstream recommends at least
24GB VRAM. Install a CUDA Toolkit compatible with your PyTorch build before
building FlashAttention, FlexGEMM, CuMesh and the rendering/export extensions.

The measured release runtime is Python 3.11.9, PyTorch 2.8.0+cu128,
FlashAttention 2.8.3 and SageAttention 2.2.0 on Windows/RTX 4060 8GB using a
community compatibility runtime and stage loading. This validates the reported
experiments, not a universal native-Windows or 8GB installation path. A clean
Linux installation and the reporter's RTX 3090 stack were not exercised here.

## Source installation

```bash
git clone --recursive https://github.com/Archerkattri/fast-trellis2.git
cd fast-trellis2
bash setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec
conda activate trellis2
```

The installer builds native extensions; it can take significant time. Review
`bash setup.sh --help` for individual components. Existing model/download caches
are reusable. Keep the CUDA/PyTorch/compiler versions consistent when rebuilding.

`--new-env` creates a Python 3.11 conda environment with PyTorch 2.8.0/cu128.
`--basic` pins Transformers 4.57.3, matching the measured runtime. The script
does not install system packages or configure a CUDA compiler. CuMesh/FlexGEMM
and renderutils are fetched from upstream branches; their exact build commits
are printed, and build directories are retained. This is a source installer,
not a fully locked binary distribution. The [recorded environment](../benchmarks/issue4/release-environment.json)
is evidence for the measurements, not a cross-platform dependency lockfile.

The installer activates its new environment within the installer process;
activate `trellis2` in your terminal afterward as shown above. For an existing
environment, activate it first and omit `--new-env`. Model weights are downloaded by
`from_pretrained`; the default model is `microsoft/TRELLIS.2-4B`. Download/access
conditions and licenses belong to the upstream model providers, including the
image conditioner and background-removal model.

## Optional Sage profiles

Follow the [SageAttention installation requirements](https://github.com/thu-ml/SageAttention#installation):

```bash
python -m pip install sageattention==2.2.0 --no-build-isolation
```

SageAttention2++ on Ada requires a suitable CUDA toolchain; its upstream
requirements distinguish CUDA versions and GPU architectures. A working native
profile is useful before enabling optional acceleration.

`sage-fast` also uses `torch.compile` for normalization leaves. Linux needs a
working PyTorch/Triton compiler environment. Windows validation used
`triton-windows` and matching Python development headers/import libraries.
First-use compilation can add startup latency; see [benchmark startup costs](BENCHMARKS.md).

The tested Windows Sage wheel was the community build
`sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl` from
[woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows.post3).
SHA256: `7dabcd00e63229b28f046c5a69ec37cf4756afb375dbadd1975dadec045ae21c`.
Use a wheel matching your own environment; do not install this build into an
unmatched PyTorch/CUDA installation.

## Optional legacy cache runtime

Only the experimental Taylor budget adapter needs the shared runtime:

```bash
python -m pip install --no-deps "git+https://github.com/Archerkattri/hicache-plus-plus.git@8fd47f9cfd2daca72a2679de69563a11b4f80f3b"
```

The pinned source supports the `taylor` budget token; older package releases may
not. Native/Sage profiles do not require this optional runtime.

## Verify

```bash
python tools/generate.py --help
python tools/generate.py --profile native --image input.png --output native.glb
python tools/generate.py --profile sage --image input.png --output sage.glb
```

Do not delete caches to diagnose a general speed problem. Check the selected
profile, first-use compilation, GPU memory pressure and measured stage times.

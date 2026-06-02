<div align="center">

# 🛠 fast-trellis2

**[Fast-TRELLIS](https://github.com/wlfeng0509/Fast-SAM3D/tree/Fast-TRELLIS)'s training-free acceleration, faithfully ported onto [TRELLIS.2](https://github.com/microsoft/TRELLIS).**

`TRELLIS.2-4B` · `1024_cascade` (mesh + texture) · training-free · single RTX 5090 · MIT

</div>

> **This is a port, not a new method.** It re-implements the acceleration introduced by
> **Fast-TRELLIS** (built for TRELLIS v1) on the **v2** sampler stack, so you get the same
> speedup on TRELLIS.2. All credit for the acceleration design belongs to the Fast-TRELLIS
> authors; all credit for the base model belongs to microsoft/TRELLIS.2. For our *own*
> acceleration method on v2, see the sibling repo **faster-trellis2** (HiCache + Adaptive
> Guidance).

`fast-trellis2` wires Fast-TRELLIS's cross-step caching into TRELLIS.2's three flow-matching
stages (sparse structure + shape SLaT + texture SLaT). Microsoft TRELLIS.2 model / decoder /
o-voxel code is left untouched.

---

## At a glance

| config | CD ↓ | F1@0.05 ↑ | vIoU ↑ | latency | speedup |
|---|:--:|:--:|:--:|:--:|:--:|
| vanilla (stock TRELLIS.2) | 0.197 | 0.370 | 0.037 | 15.79 s | 1.00× |
| **fast-trellis2** (this port) | 0.221 | 0.334 | 0.022 | 6.46 s | **2.44×** |

<sub>20 Toys4K objects, `1024_cascade`, RTX 5090. CD ↓ better; F1/vIoU ↑ better;
`vIoU` = surface-shell occupancy IoU.</sub>

> **Honest tradeoff.** This port is the **fastest** (2.44×) but **degrades geometry** —
> highest CD, lowest F1, vIoU roughly halved vs vanilla. That is the original authors' speed /
> quality tradeoff faithfully reproduced, **not a regression introduced here**. If you want
> speed *and* quality on v2, use **faster-trellis2**.

---

## Quickstart

```bash
git clone --recursive https://github.com/Archerkattri/fast-trellis2
cd fast-trellis2
bash setup.sh --new-env --basic --flash-attn --o-voxel --flexgemm --nvdiffrast --nvdiffrec
```

Enable acceleration by swapping in the ported samplers; the pipeline auto-detects them and
flips `enable_faster` on:

```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline, samplers

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B").cuda()

pipeline.sparse_structure_sampler = samplers.FlowEulerGuidanceIntervalSampler_taylor(
    sigma_min=pipeline.sparse_structure_sampler.sigma_min)
pipeline.shape_slat_sampler = samplers.FlowEulerGuidanceIntervalSampler_faster(
    sigma_min=pipeline.shape_slat_sampler.sigma_min)
pipeline.tex_slat_sampler = samplers.FlowEulerGuidanceIntervalSampler_faster(
    sigma_min=pipeline.tex_slat_sampler.sigma_min)
# pipeline.enable_faster auto-enables once the *_faster / *_taylor samplers are set.

mesh = pipeline.run(image)[0]
```

See `example.py` (stock) and `example_faster.py` (accelerated).

<details>
<summary><b>Blackwell (RTX 50-series / sm_120) note</b></summary>

spconv's implicit-GEMM path can SIGFPE on sm_120; force the native backend:

```bash
SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native python example_faster.py
```

`SPCONV_ALGO` is read from the environment (`trellis2/modules/sparse/conv/config.py`).
</details>

---

## What the port replicates

Fast-TRELLIS's three components, wired into the TRELLIS.2 samplers (all credit: Fast-TRELLIS):

| component | what it does | where |
|---|---|---|
| **TaylorSeer on SS** | sparse-structure stage caches the final velocity, Taylor-extrapolates on skipped steps | `taylor_utils_ss/` |
| **SLaT delta-cache** | shape/texture SLaT reuse a cached velocity delta, gated by a learned sensitivity `k` + cosine-direction error | `faster_utils_slat/` |
| **Token carving** | voxels ranked by 3D high-frequency energy; low-freq tokens skipped on a fraction of steps, restored from cache | `token_slat/`, `fft/fft3d.py` |

<details>
<summary><b>v1 → v2 port notes</b> (API / schedule adaptations only — no logic changes)</summary>

- v1 `cfg_strength` / `cfg_interval` → v2 `guidance_strength` / `guidance_interval`.
- v1's single CFG-in-interval mixin → v2's split MRO (`GuidanceIntervalSamplerMixin`,
  `ClassifierFreeGuidanceSamplerMixin`, base).
- v1's fixed 25-step cache schedule → parameterised to v2's shorter (~12-step) schedule. With
  v1's raw constants the SS stage collapsed to 0 voxels; the schedule is now scaled to the
  actual step count (warm-up steps + an always-full final step). See the comments in
  `trellis2/pipelines/samplers/flow_euler.py`.
- Token carving auto-disables on the cascade-upsampling and texture stages (carved indices no
  longer align once coords are re-derived / a concat conditioning tensor is present); the easy
  delta-cache still applies. Logged at runtime.

Ported code: `trellis2/pipelines/samplers/flow_euler.py`, `.../samplers/__init__.py`,
`trellis2/pipelines/trellis2_image_to_3d.py` (`enable_faster`, `coords_scores`), and the util
packages `taylor_utils_ss/`, `faster_utils_slat/`, `token_slat/`, `fft/`.
</details>

---

## Credits & license

This repo **reproduces the method of, and depends on, two MIT-licensed projects** — both are
credited because this work reproduces their contributions:

| | |
|---|---|
| **microsoft/TRELLIS.2** | the base image-to-3D model, pipeline, and decoders |
| **Fast-TRELLIS** | [wlfeng0509/Fast-SAM3D (Fast-TRELLIS branch)](https://github.com/wlfeng0509/Fast-SAM3D/tree/Fast-TRELLIS) — the training-free acceleration this repo ports to v2 |

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The port wiring © 2026 Krishi Attri; the
acceleration design © the Fast-TRELLIS authors; the base model © Microsoft.

**Krishi Attri** · krishiattriwork@gmail.com · [github.com/Archerkattri](https://github.com/Archerkattri)

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@misc{attri2026fasttrellis2,
  title  = {fast-trellis2: Fast-TRELLIS acceleration ported to TRELLIS.2},
  author = {Krishi Attri}, year = {2026},
  howpublished = {\url{https://github.com/Archerkattri/fast-trellis2}}
}
@article{trellis2,
  title   = {Native and Compact Structured Latents for 3D Generation},
  author  = {Microsoft TRELLIS.2 Team},
  journal = {arXiv preprint arXiv:2512.14692}, year = {2025}
}
@misc{fasttrellis,
  title  = {Fast-TRELLIS}, author = {wlfeng0509},
  howpublished = {\url{https://github.com/wlfeng0509/Fast-SAM3D/tree/Fast-TRELLIS}}
}
```
</details>

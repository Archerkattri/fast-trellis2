# Generation and profiles

## Command line

```bash
python tools/generate.py --profile sage --image input.png --output result.glb
python tools/generate.py --profile sage-fast --image input.png --output result-fast.glb
```

The default profile is `native`. Export defaults to 60,000 target faces, a
1024-pixel texture and `remesh=False`. Target face counts are approximate.

| Option | Meaning |
|---|---|
| `--profile` | `native`, `sage`, `sage-fast`, `sage-fp16pv`, `legacy-cache` |
| `--pipeline` | `512`, `1024`, `1024_cascade` (default), `1536_cascade` |
| `--seed` | Reproducible sampling seed; default 42 |
| `--steps` | Override all three sampler schedules; otherwise use model configuration |
| `--faces` / `--texture-size` | Export target, separate from sampling resolution |
| `--preprocessed` | Use an input with the intended background/crop already applied |
| `--remesh` | Explicitly enable remeshing during export |
| `--overwrite` | Replace an existing GLB/run metadata file |
| `--model` | Model repository or local pretrained model directory |

`example.py` and `example_faster.py` call the same CLI. The previous `--attention`
option remains a compatibility alias; prefer `--profile` for new scripts. The
examples now generate a GLB and metadata rather than automatically rendering a
video. Rendering/export durations should not be confused with sampler latency.
GLB textures use the `EXT_texture_webp` extension; use a viewer that supports it.

The `.run.json` sidecar records the model identifier, profile, seed, device,
settings and CUDA-synchronized loading/generation/export times. Its generation
time includes image conditioning, preprocessing/decoding and any first-use
compilation. It is not the sampler-only time in the published benchmark table.

## Python API

```python
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.configure_profile("sage-fast", pipeline_type="1024_cascade")
pipeline.cuda()
meshes = pipeline.run(Image.open("input.png"), seed=42, pipeline_type="1024_cascade")
```

`configure_profile` selects samplers and acceleration together, preserves guidance
and step counts, clears cache-specific sampler options and releases old sampler
caches. Configure an idle pipeline between requests. A pipeline instance is not
a concurrent request scheduler; use a separate worker/instance for concurrent jobs.

For kernel-only control, the lower-level API remains available:

```python
pipeline.enable_sage_attention(mode="auto", include_cross=True, compile_norms=True)
pipeline.enable_sage_attention(mode=None)  # Restore native attention/normalization.
```

The lower-level method does not replace a caller's samplers. Named profiles are
the safer choice when switching away from legacy cache settings.

## Choosing a profile

- `native`: fresh Euler evaluations with native attention; reference behavior.
- `sage`: fresh evaluations with final-flow self-attention acceleration.
- `sage-fast`: adds final-flow cross-attention and compiled normalization; faster
  in the measured runs, with greater deviation in some material channels.
- `sage-fp16pv`: Sage self-attention with an FP16 value product and FP32 accumulation.
- `legacy-cache`: explicit approximate forecasting/token caching; can damage
  geometry and textures. See [legacy controls](LEGACY-CACHE.md).

Sage and compiled kernels are approximate numerically. Use paired comparisons
on your input/GPU when fidelity is important. Model weights remain unchanged.

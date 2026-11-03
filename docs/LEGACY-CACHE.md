# Experimental legacy cache

The `legacy-cache` profile reproduces the approximate acceleration ported from
Fast-TRELLIS. It is retained for research and comparison, not recommended as a
quality-preserving replacement for native inference.

```bash
python tools/generate.py --profile legacy-cache --image input.png --output legacy.glb
```

This explicitly enables Taylor SS forecasting and SLaT velocity-delta caching,
including at 12 steps. In contrast, direct use of the cache sampler classes
defaults to full evaluation on schedules of 12 steps or fewer. Longer schedules
retain approximate defaults. Token carving requires aligned scores; cascade and
texture stages cannot reuse nonmatching coordinate scores.

The optional Taylor adapter uses the [pinned shared runtime](INSTALLATION.md#optional-legacy-cache-runtime).
It accepts a `CacheBudget` with backend `taylor` or `full`, allowed stage exactly
`('sparse_structure',)` and fallback `full`. Warmup and terminal steps run fully;
horizon is measured from the last real evaluation. `max_memory_mb` bounds retained
cache payload, not total VRAM. No calibrated forward-error estimator exists;
error caps are rejected. Audit metadata is not an autonomous quality controller.

Unknown weight/input/source digests remain empty in default run identities.
Caller-supplied identities are declarations, not automatically verified hashes.
No cache persists between completed runs. The recorded `activated_steps` count
sampler steps, not conditional/unconditional branch calls.

The forecast formula remains the upstream Taylor baseline. The
[original CPU contract](https://github.com/Archerkattri/fast-trellis2/blob/v0.2.0/docs/archive/ACC16-CPU-CONTRACT.md)
records the budget integration and its tests.

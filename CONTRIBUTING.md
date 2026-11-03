# Contributing

Open an issue or pull request with a concrete problem, reproduction and expected
behavior. Keep changes focused and preserve upstream model/decoder attribution.

For an inference change, include the input/seed, native control, model/source
revisions, GPU/runtime, raw outputs and a precise timing scope. Separate model
loading, sampling, decoding, rendering and export. Record first-use compilation
separately. Report failures and quality regressions alongside speedups.

Run the relevant checks:

```bash
python tools/verify_cascade_scores.py
python tools/verify_sage_attention.py
python tools/verify_fast_norm.py
python tools/verify_profiles.py
python -m compileall -q trellis2 tools
```

GPU changes also need CUDA/kernel and learned-model evidence; CPU source
validation is not a substitute. Do not change the legacy Taylor formula under a
baseline-control claim. Add a separate named method/profile for a new algorithm.

Use plain documentation, explicit profile names and measured claims. Avoid
formatting unrelated vendored code. Configure a pipeline between requests rather
than mutating a shared instance during concurrent inference.

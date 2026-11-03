# Release and publication checklist

Releases are source releases with measured validation, not certification for
every GPU/OS combination or a claim of peer-review acceptance.

## Maintainer checks

1. Keep the source tree clean; increment `trellis2/_version.py`, CHANGELOG and citation version.
2. Run the CPU validation suite, lint/format checks and compile checks in CI.
3. For changed GPU behavior, run real CUDA checks and paired learned inference.
4. Record exact model/source versions, input hashes, runtime, seeds, warmup state and timing scopes.
5. Review raw geometry, appearance and PBR attributes; preserve failed experiments.
6. Validate CLI argument handling and a real GLB export. State any compatibility adapter used.
7. Build figures from actual renders and recorded JSON. Inspect PNG and vector exports.
8. Build a release evidence bundle and SHA256 manifest; exclude model weights,
   private paths, credentials, caches and unlicensed external assets.
9. Push/tag the tested commit, wait for CI, then publish GitHub notes and evidence assets.

## Validation boundary

Version 0.2.0 includes CPU regression checks, real CUDA attention/normalization
checks and actual inference/render evidence on the documented Windows RTX 4060
compatibility runtime. Standard Linux source installation and the original
reporter's RTX 3090 remain external validation targets. These limits should
remain visible in release notes and any publication using the results.

## Reproducing figures

```bash
python -m pip install matplotlib pillow
python benchmarks/build_figures.py --renders /path/to/release-comparison
```

The [benchmark protocol](../benchmarks/README.md) explains source/weight revisions
and render inputs. The data support within-pair comparisons, not a claim that
the smaller V1 model is an acceleration of the larger V2 model.

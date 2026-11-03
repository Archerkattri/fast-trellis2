# Changelog

## 0.2.0 — 2026-11-03

- Added explicit `native`, `sage`, `sage-fast`, `sage-fp16pv` and `legacy-cache` profiles.
- Added SageAttention integration and optional compiled normalization while retaining native sampling schedules.
- Added a unified GLB CLI, argument validation and run metadata; retained example entry points and the `--attention` alias.
- Made short cache schedules conservative by default; kept approximate legacy behavior as an explicit research profile.
- Removed unused cascade FFT scoring and stale score reuse across generations.
- Added CPU/CUDA checks, paired learned-model evidence, material diagnostics and a four-way model comparison.
- Reorganized documentation, archived development records, and published reproducibility data/figures.

Measured on the documented RTX 4060 runtime: V2 `sage-fast` sampling is 31–32%
lower latency than native on the uploaded stool at two seeds. This is 1.45–1.47×
speed, with near-native geometry/base color and some PBR deviations. Compilation
startup costs and hardware limits are documented.

Migration: examples now write GLB plus `.run.json` rather than rendering a video
automatically. Prefer `--profile`; the older `--attention` values remain aliases.
Native remains the default. Experimental caching requires explicit selection.

## 0.1.1 — 2026-09-18

Historical baseline-lock/CI release. See the GitHub release record for its original scope.

## 0.1.0 — 2026-06-11

Initial Fast-TRELLIS acceleration port to TRELLIS.2.

# ACC16 bounded Taylor runtime — 2026-10-13

## Task record

Initial observed state: clean `wip/release-readiness-2026-10-11` at
`d313a4203243abcabf5654665261a39555f79a11`, matching the remote WIP branch.
No repository-local AGENTS.md was found; research workspace instructions apply.

Objective: integrate the shared method-neutral budget with the existing SS
Taylor formula, preserve warmup/terminal full steps, enforce full fallback, and
release sampler cache references on success and failure. CPU-only contracts
must exercise actual sampler source without importing the eager TRELLIS root,
`example_faster.py`, weights, asset downloaders, or GPU code.

Observed environment: Python 3.12.10, torch 2.13.0+cpu, numpy 2.5.2,
pytest 9.1.1; easydict is absent. The CPU harness can supply only its small
attribute-dictionary surface and the unused sparse module namespace. It will
load the actual base sampler, guidance mixins, Taylor utilities, and sampler.
This is an isolated contract seam, not a dependency-complete pipeline import.

Implemented on the existing WIP branch. The adapter accepts only the Taylor
or full backend, only the sparse-structure stage, and full fallback; an
uncalibrated predicted-error cap is rejected instead of being treated as
enforced. The shared HiCache++ budget recognizes Taylor as a method token but
does not implement the Taylor formula. The example is import-safe; loading
weights and CUDA is confined to `main()`.

Verification (Windows, Python 3.12.10, torch 2.13.0+cpu):

- `python -m pytest -q tests/test_taylor_contract.py tests/test_example_import.py -p no:cacheprovider` with the HiCache++ checkout on `PYTHONPATH`: **21 passed, 36 subtests passed**.
- The CPU harness loads the actual sampler, guidance mixins, Taylor functions,
  and shared runtime. It counts actual model calls and tests schedule edges,
  full-step anchors after budget fallbacks, bounded horizon, exact disabled and
  full-backend parity, stage/memory fallback, CFG, repeat-run isolation,
  import safety, and cleanup on failures. It replaces only the package-root
  import namespace and the absent `easydict` surface; it is not a full TRELLIS.2
  dependency/runtime test.
- `python -m compileall -q taylor_utils_ss trellis2 example_faster.py tests`
  and `git diff --check`: **passed**.
- HiCache++ `python tests/run_tests.py`: **all modules passed**, including the
  new budget runtime tests, existing DIT forecast/checkpoint tests, and five
  H1 adapter-contract tests. `compileall` and `git diff --check` passed.

The temporary unit-test files used for this verification were removed after
the run, consistent with the owner's repository-wide instruction not to retain
unit tests. No new test files are included in the proposed source changes.

Evidence limits: these are CPU contract checks, not a dependency-complete
TRELLIS.2 import, GPU timing, memory-peak measurement, generated-asset quality
evaluation, or a comparison against the upstream acceleration. The shared
budget's memory value is explicitly the Taylor factor and previous-velocity
tensor payload only; it excludes model weights, sparse coordinates, activations,
temporary tensors, and device peak. This adapter has no validated forecast
error estimator and rejects `max_predicted_error`.

Integration prerequisite: install a HiCache++ source/wheel that contains the
Taylor method token before running the accelerated sparse-structure sampler.
The repository's existing standalone example does not configure that shared
runtime dependency. Do not describe the sampler integration as a supported
standalone install until the dependency is versioned and its setup is tested.

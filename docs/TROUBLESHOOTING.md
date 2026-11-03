# Troubleshooting

## Missing CUDA or extension imports

Generation requires a CUDA GPU. Check `torch.cuda.is_available()` and the installed
PyTorch/CUDA build before importing CuMesh, FlexGEMM or nvdiffrast. Rebuild binary
extensions after changing PyTorch/CUDA. A CPU-only source test passing does not
validate GPU dependencies. Consult the extension's upstream installation guide
for your platform.

## Out of memory

Start with the native 512 pipeline and upstream low-VRAM settings. Sampling
resolution, token count, raw decoding and GLB texture baking all affect memory.
The 8GB validation used stage loading in a separate compatibility runtime;
installing this source tree alone does not provide that adapter. Preserve model
and kernel caches. Deleting caches is not a general memory or speed fix.

## First Sage run is slower

`sage-fast` compiles normalization kernels on first use. A new shape/device or
toolchain can trigger compilation again. Compare warm runs and record cold-start
time separately. Plain `sage` avoids normalization compilation. Kernel speed
does not imply the same end-to-end speedup; loading and export still cost time.

## Sage import or compiler failure

Install a package matching PyTorch, CUDA and GPU architecture. On Windows, the
tested community wheel also needed matching Python headers/import libraries for
Triton. Use `native` or plain `sage` while diagnosing optional compilation.
Do not silently substitute another precision/backend in a reported benchmark.

## Holes, blotchy textures or material changes

Check the selected profile. `legacy-cache` can introduce substantial artifacts.
Compare with `native` using the same input, seed and schedule. Disable remeshing
while isolating sampling/decoder differences. `sage-fast` changes some material
attributes more than plain `sage`; close base-color renders do not guarantee
identical metallic/alpha values or specular appearance.

## Input/background problems

An RGBA input with a meaningful alpha channel can avoid background removal.
Use `--preprocessed` only when the intended crop/background is already applied.
The image conditioner and background remover have independent model dependencies.
Keep their identifiers and preprocessing settings in a reproducibility record.

## Reporting a problem

Attach the CLI `.run.json`, profile, complete error, versions/GPU, input image if
shareable, and matched native/accelerated outputs. Remove private filesystem
details if necessary. Report security issues using [SECURITY.md](../SECURITY.md).

"""
Smoke test for the Fast-TRELLIS acceleration ported onto TRELLIS.2.

Loads TRELLIS.2-4B, swaps in the *_taylor (SS) and *_faster (SLaT) samplers,
runs the 512 pipeline on one masked RGBA crop, and confirms:
  * the pipeline loads,
  * the cache-accelerated samplers run (they print cache type + "Activated steps"),
  * a mesh with vertices is produced.

Run SOLO on one GPU. nvdiffrast/nvdiffrec are render-only and stubbed.
"""
import os, sys, types, time


def _stub_render_deps():
    for m in ("nvdiffrast", "nvdiffrast.torch", "nvdiffrec"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["nvdiffrast"].torch = sys.modules["nvdiffrast.torch"]


def main():
    _stub_render_deps()
    here = os.path.dirname(os.path.abspath(__file__))
    # fast-trellis2 root holds the top-level util packages (taylor_utils_ss,
    # faster_utils_slat, token_slat, fft) imported by the ported samplers.
    sys.path.insert(0, here)

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import samplers as samplers_mod

    ckpt = os.path.join(here, "ckpts", "TRELLIS.2-4B")
    img_path = os.path.join(here, "assets", "example_image", "T.png")

    t0 = time.perf_counter()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(ckpt)
    pipe.to("cuda")
    print(f"[test] loaded pipeline in {time.perf_counter()-t0:.1f}s")

    # Swap to the Fast-TRELLIS accelerated samplers (faithful v1 port on v2).
    sm = samplers_mod
    pipe.sparse_structure_sampler = sm.FlowEulerGuidanceIntervalSampler_taylor(
        sigma_min=pipe.sparse_structure_sampler.sigma_min)
    pipe.shape_slat_sampler = sm.FlowEulerGuidanceIntervalSampler_faster(
        sigma_min=pipe.shape_slat_sampler.sigma_min)
    pipe.tex_slat_sampler = sm.FlowEulerGuidanceIntervalSampler_faster(
        sigma_min=pipe.tex_slat_sampler.sigma_min)
    pipe.enable_faster = True
    print("[test] samplers:",
          type(pipe.sparse_structure_sampler).__name__,
          type(pipe.shape_slat_sampler).__name__,
          type(pipe.tex_slat_sampler).__name__)

    # Prefer a pre-masked alpha (no rembg); fall back to in-pipeline rembg
    # preprocessing when the asset carries no usable alpha channel.
    image = Image.open(img_path).convert("RGBA")
    has_alpha = image.split()[-1].getextrema()[0] < 255
    if has_alpha:
        image = pipe.preprocess_image(image)
        do_preprocess = False
    else:
        print("[test] no alpha mask on asset; using in-pipeline rembg preprocess")
        do_preprocess = True

    torch.manual_seed(42)
    t1 = time.perf_counter()
    out_mesh = pipe.run(image, seed=42, preprocess_image=do_preprocess, pipeline_type="512")
    print(f"[test] run() finished in {time.perf_counter()-t1:.1f}s")

    mesh = out_mesh[0]
    nv = int(mesh.vertices.shape[0])
    nf = int(mesh.faces.shape[0])
    print(f"[test] MESH vertices={nv} faces={nf}")
    assert nv > 0, "mesh has no vertices"
    print("FAST_TRELLIS2_PORT_OK")


if __name__ == "__main__":
    main()

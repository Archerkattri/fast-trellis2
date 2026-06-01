"""
fast-trellis2 example: image -> 3D with Fast-TRELLIS acceleration enabled.

This mirrors example.py but swaps the stock TRELLIS.2 samplers for the ported
Fast-TRELLIS cache-accelerated samplers (TaylorSeer on the SS final velocity +
easy delta-cache / token-carving on SLaT). ~2.4x faster end to end on RTX 5090,
at a faithful-to-the-original geometry-quality tradeoff (see README).

Blackwell (sm_120) env, set before launch:
    SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native python example_faster.py
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines import samplers
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Environment map for PBR video render.
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load pipeline.
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Swap in the ported Fast-TRELLIS accelerated samplers.
#    SS stage -> TaylorSeer cache; shape/tex SLaT stages -> easy delta-cache
#    (with token carving on the shape stage). enable_faster wires the
#    coords_scores plumbing the carver needs.
pipeline.sparse_structure_sampler = samplers.FlowEulerGuidanceIntervalSampler_taylor(
    sigma_min=pipeline.sparse_structure_sampler.sigma_min)
pipeline.shape_slat_sampler = samplers.FlowEulerGuidanceIntervalSampler_faster(
    sigma_min=pipeline.shape_slat_sampler.sigma_min)
pipeline.tex_slat_sampler = samplers.FlowEulerGuidanceIntervalSampler_faster(
    sigma_min=pipeline.tex_slat_sampler.sigma_min)
pipeline.enable_faster = True

# 4. Load image & run.
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast vertex limit

# 5. Render video.
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample_faster.mp4", video, fps=15)

# 6. Export to GLB.
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample_faster.glb", extension_webp=True)

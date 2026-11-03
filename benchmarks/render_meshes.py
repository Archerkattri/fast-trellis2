"""Render four trusted raw mesh artifacts using the release camera protocol.

Requires the TRELLIS.2 rendering dependencies. PyTorch artifacts must be obtained
from trusted sources. No simplification or image synthesis is performed.
"""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description=__doc__)
for name in ("trellis", "fast-trellis", "trellis2", "fast-trellis2"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

import numpy as np
from PIL import Image
import torch
from trellis2.representations import Mesh, MeshWithVoxel
from trellis2.renderers import MeshRenderer
from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

args.output.mkdir(parents=True, exist_ok=True)
record = {
    "seed": 42,
    "render": "nvdiffrast CUDA, 512px, raw undecimated meshes, base color + clay",
    "camera": {
        "yaw_start": math.pi / 4,
        "yaw_step": math.pi / 3,
        "pitch": 0.3,
        "radius": 2,
        "fov_degrees": 40,
    },
    "cases": [],
}
renderer = MeshRenderer({"resolution": 512, "near": 0.1, "far": 10, "ssaa": 1})
with torch.no_grad():
    for label in ("trellis", "fast-trellis", "trellis2", "fast-trellis2"):
        data = torch.load(
            getattr(args, label.replace("-", "_")),
            weights_only=True,
            map_location="cpu",
        )
        if label in ("trellis", "fast-trellis"):
            mesh = Mesh(
                data["vertices"].cuda(),
                data["faces"].cuda(),
                data["vertex_attrs"].cuda(),
            )
            appearance = "vertex color"
        else:
            mesh = MeshWithVoxel(
                data["vertices"].cuda(),
                data["faces"].cuda(),
                data["origin"].tolist(),
                data["voxel_size"],
                data["coords"].cuda(),
                data["attrs"].cuda(),
                torch.Size(data["voxel_shape"]),
                {key: slice(*value) for key, value in data["layout"].items()},
            )
            appearance = "volumetric PBR base color"
        for view in range(6):
            extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(
                view * math.pi / 3 + math.pi / 4, 0.3, 2, 40
            )
            rendered = renderer.render(
                mesh, extr, intr, return_types=["attr", "mask", "normal"]
            )
            mask = rendered["mask"].reshape(1, 512, 512).clamp(0, 1)
            color = (
                rendered["attr"][:3] if "attr" in rendered else rendered["base_color"]
            ).clamp(0, 1)
            normal = rendered["normal"] * 2 - 1
            light = torch.tensor([0.3, -0.6, 0.75], device="cuda")
            light = light / light.norm()
            shade = (
                0.35 + 0.65 * (normal * light[:, None, None]).sum(0, keepdim=True).abs()
            )
            for suffix, value in (
                ("color", color),
                ("clay", shade.expand_as(color) * 0.76),
            ):
                pixels = value * mask + (1 - mask) * 0.96
                pixels = np.clip(pixels.permute(1, 2, 0).cpu().numpy() * 255, 0, 255)
                Image.fromarray(pixels.astype(np.uint8)).save(
                    args.output / f"{label}-view-{view}-{suffix}.png"
                )
        record["cases"].append(
            {
                "label": label,
                "raw_vertices": len(mesh.vertices),
                "raw_faces": len(mesh.faces),
                "appearance": appearance,
            }
        )
        del mesh, data, rendered
        torch.cuda.empty_cache()
(args.output / "render-manifest.json").write_text(json.dumps(record, indent=2) + "\n")

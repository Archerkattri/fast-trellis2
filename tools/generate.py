"""Generate a textured GLB with an explicit, reproducible inference profile."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILES = ("native", "sage", "sage-fast", "sage-fp16pv", "legacy-cache")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=ROOT / "assets/example_image/T.png"
    )
    parser.add_argument("--output", type=Path, default=Path("sample.glb"))
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument(
        "--attention",
        choices=("native", "sage", "sage-fast", "sage_fp16pv"),
        help="Compatibility alias; prefer --profile",
    )
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument(
        "--pipeline",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="1024_cascade",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        type=positive_int,
        help="Override all three native sampler step counts",
    )
    parser.add_argument("--faces", type=positive_int, default=60000)
    parser.add_argument(
        "--texture-size", type=int, choices=(512, 1024, 2048, 4096), default=1024
    )
    parser.add_argument(
        "--remesh", action="store_true", help="Opt into remeshing during export"
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Input already has the intended crop/background",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile and args.attention:
        parser.error("choose --profile or the --attention compatibility alias")
    profile = args.profile or (args.attention or "native").replace(
        "sage_fp16pv", "sage-fp16pv"
    )
    if not args.image.is_file():
        parser.error(f"input image does not exist: {args.image}")
    if args.output.suffix.lower() != ".glb":
        parser.error("--output must have a .glb extension")
    metadata_path = args.output.with_suffix(".run.json")
    if not args.overwrite and (args.output.exists() or metadata_path.exists()):
        parser.error(
            "output or run metadata already exists; use --overwrite to replace it"
        )
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")

    import torch
    from PIL import Image
    import o_voxel
    from trellis2 import __version__
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if not torch.cuda.is_available():
        parser.error("generation requires a CUDA GPU; see docs/INSTALLATION.md")
    start = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.configure_profile(profile, pipeline_type=args.pipeline)
    pipeline.cuda()
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    step_options = {} if args.steps is None else {"steps": args.steps}
    image = Image.open(args.image)
    mesh = pipeline.run(
        image,
        seed=args.seed,
        pipeline_type=args.pipeline,
        preprocess_image=not args.preprocessed,
        sparse_structure_sampler_params=step_options,
        shape_slat_sampler_params=step_options,
        tex_slat_sampler_params=step_options,
    )[0]
    torch.cuda.synchronize()
    generated = time.perf_counter()
    raw_vertices, raw_faces = len(mesh.vertices), len(mesh.faces)
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.faces,
        texture_size=args.texture_size,
        remesh=args.remesh,
        verbose=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(args.output), extension_webp=True)
    torch.cuda.synchronize()
    record = {
        "version": __version__,
        "profile": profile,
        "model": args.model,
        "pipeline": args.pipeline,
        "seed": args.seed,
        "step_override": args.steps,
        "image": str(args.image.resolve()),
        "output": str(args.output.resolve()),
        "preprocessed": args.preprocessed,
        "raw_vertices": raw_vertices,
        "raw_faces": raw_faces,
        "export_faces_target": args.faces,
        "texture_size": args.texture_size,
        "remesh": args.remesh,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "load_seconds": loaded - start,
        "generation_seconds": generated - loaded,
        "export_seconds": time.perf_counter() - generated,
        "timing_scope": "CUDA-synchronized CLI wall time; generation includes preprocessing/conditioning/decoding and any first-use compilation",
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

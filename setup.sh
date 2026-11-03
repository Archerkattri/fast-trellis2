#!/usr/bin/env bash
# Linux/CUDA source installer. Build directories and caches are retained.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash setup.sh [OPTIONS]
  -h, --help       Show help (no environment changes)
  --new-env        Create/activate conda environment trellis2 (Python 3.11)
  --basic          Install Python dependencies
  --flash-attn     Install FlashAttention 2.8.3
  --cumesh         Build CuMesh
  --o-voxel        Build bundled o-voxel
  --flexgemm       Build FlexGEMM
  --nvdiffrast     Build nvdiffrast v0.4.0
  --nvdiffrec      Build renderutils
CUDA Toolkit, a compatible compiler, git and conda (--new-env only) are required.
The new environment uses PyTorch 2.8.0 / CUDA 12.8. SageAttention is optional;
see docs/INSTALLATION.md. Linux source installation is not CI GPU-certified.
Existing caches/environments are never deleted. Extension build trees are kept
under ${TMPDIR:-/tmp}/fast-trellis2-build.* for inspection and reuse.
EOF
}

if (($# == 0)); then usage; exit 0; fi
declare -A selected=()
for option in "$@"; do
    case "$option" in
        -h|--help) usage; exit 0 ;;
        --new-env|--basic|--flash-attn|--cumesh|--o-voxel|--flexgemm|--nvdiffrast|--nvdiffrec)
            selected["${option#--}"]=1 ;;
        *) echo "Unknown option: $option" >&2; exit 2 ;;
    esac
done
[[ $(uname -s) == Linux ]] || { echo 'Use Linux with CUDA; see docs/INSTALLATION.md for Windows limits.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/nvidia-smi is required.' >&2; exit 1; }
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_dir"
if [[ ${selected[new-env]:-0} == 1 ]]; then
    command -v conda >/dev/null || { echo 'conda is required for --new-env.' >&2; exit 1; }
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda create -y -n trellis2 python=3.11
    conda activate trellis2
    python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
fi
if [[ ${selected[basic]:-0} == 1 ]]; then
    python -m pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
        transformers==4.57.3 gradio==6.0.1 tensorboard pandas lpips zstandard pillow kornia timm
    python -m pip install 'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8'
fi
if [[ ${selected[flash-attn]:-0} == 1 ]]; then
    python -m pip install packaging psutil
    python -m pip install flash-attn==2.8.3 --no-build-isolation
fi

build_dir=''
build_extension() {
    local name=$1 url=$2 ref=$3
    command -v nvcc >/dev/null || { echo 'CUDA Toolkit (nvcc) is required to build extensions.' >&2; exit 1; }
    if [[ -z $build_dir ]]; then
        build_dir=$(mktemp -d "${TMPDIR:-/tmp}/fast-trellis2-build.XXXXXXXX")
        echo "Retaining extension sources at $build_dir"
    fi
    if [[ $ref == HEAD ]]; then
        git clone --recursive "$url" "$build_dir/$name"
    else
        git clone --recursive --branch "$ref" "$url" "$build_dir/$name"
    fi
    echo "$name source: $(git -C "$build_dir/$name" rev-parse HEAD)"
    python -m pip install "$build_dir/$name" --no-build-isolation
}
[[ ${selected[nvdiffrast]:-0} == 0 ]] || build_extension nvdiffrast https://github.com/NVlabs/nvdiffrast.git v0.4.0
[[ ${selected[nvdiffrec]:-0} == 0 ]] || build_extension nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git renderutils
[[ ${selected[cumesh]:-0} == 0 ]] || build_extension CuMesh https://github.com/JeffreyXiang/CuMesh.git HEAD
[[ ${selected[flexgemm]:-0} == 0 ]] || build_extension FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git HEAD
if [[ ${selected[o-voxel]:-0} == 1 ]]; then
    command -v nvcc >/dev/null || { echo 'CUDA Toolkit (nvcc) is required for o-voxel.' >&2; exit 1; }
    python -m pip install "$repo_dir/o-voxel" --no-build-isolation
fi
echo 'Selected installation stages completed.'

"""Build publication figures from recorded measurements and actual rendered PNGs."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("benchmarks/issue4/results.json"))
parser.add_argument("--renders", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("dist/figures"))
args = parser.parse_args()
data = json.loads(args.data.read_text())
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}
)
cases = data["four_way"]
for suffix, title in [("color", "Appearance"), ("clay", "Geometry")]:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.25), facecolor="white")
    for ax, case in zip(axes, cases):
        ax.imshow(Image.open(args.renders / f"{case['id']}-view-0-{suffix}.png"))
        ax.set_title(case["title"] + "\n" + case["configuration"], fontsize=12, pad=10)
        ax.axis("off")
        ax.text(
            0.5,
            -0.035,
            f"Sampling: {case['sampler_seconds']:.2f} s",
            ha="center",
            transform=ax.transAxes,
        )
    fig.suptitle(
        title + " comparison · uploaded stool · seed 42", fontsize=19, x=0.5, y=0.99
    )
    fig.text(
        0.5,
        0.035,
        "Different model generations and native resolutions. Compare speed within each native/fast pair only.",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.83, bottom=0.11, wspace=0.02)
    name = "four-way-appearance" if suffix == "color" else "four-way-geometry"
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(args.output / f"{name}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")
for ax, pair, title in zip(
    axes, [cases[:2], cases[2:]], ["TRELLIS v1 · 25 steps", "TRELLIS.2 · 12 steps"]
):
    values = [case["sampler_seconds"] for case in pair]
    ax.barh([1, 0], values, color=["#8192a6", "#176b62"], height=0.55)
    ax.set_yticks([1, 0], [case["title"] for case in pair])
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("Sampling time (s) ↓")
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    for y, value in zip([1, 0], values):
        ax.text(value + max(values) * 0.025, y, f"{value:.2f}", va="center")
    speed = values[0] / values[1]
    ax.text(
        0.98,
        0.98,
        f"{speed:.2f}× speed",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
    )
fig.suptitle("Measured within-pair sampling latency · RTX 4060 · seed 42", fontsize=16)
fig.text(
    0.5,
    0.025,
    "One input and one matched run per family; V2 fast profile uses warm compiler caches. No cross-version speed claim.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.93], w_pad=3)
for extension in ["png", "svg", "pdf"]:
    fig.savefig(
        args.output / f"sampling-speedups.{extension}", dpi=200, facecolor="white"
    )
plt.close(fig)
for path in args.output.glob("*.svg"):
    path.write_text(
        "\n".join(
            line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
print("Created PNG/SVG/PDF comparison and latency figures")

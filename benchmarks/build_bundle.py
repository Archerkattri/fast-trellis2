"""Bundle recorded evidence and all actual render views; never include weights."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--renders", type=Path, required=True)
parser.add_argument("--figures", type=Path, default=ROOT / "docs/images")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error("output already exists; choose a new bundle path")
expected = {
    f"{model}-view-{view}-{kind}.png"
    for model in ("trellis", "fast-trellis", "trellis2", "fast-trellis2")
    for view in range(6)
    for kind in ("color", "clay")
}
assert expected == {path.name for path in args.renders.glob("*.png")}
entries = {}
for path in (ROOT / "benchmarks").rglob("*"):
    if path.is_file() and path.suffix in (".py", ".json", ".md"):
        entries[path.relative_to(ROOT).as_posix()] = path
for name in ("LICENSE", "NOTICE", "CITATION.cff", "CHANGELOG.md"):
    entries[name] = ROOT / name
for path in (ROOT / "docs").rglob("*.md"):
    entries[path.relative_to(ROOT).as_posix()] = path
for path in (ROOT / "docs").glob("*.json"):
    entries[path.relative_to(ROOT).as_posix()] = path
for path in args.figures.glob("*"):
    if path.suffix in (".png", ".svg", ".pdf"):
        entries["docs/images/" + path.name] = path
for name in expected | {"render-manifest.json"}:
    entries["release-comparison/" + name] = args.renders / name
manifest = {
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "scope": "Recorded issue4 evidence; input image, pretrained weights and caches excluded",
    "files": {
        name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in sorted(entries.items())
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name, path in sorted(entries.items()):
        archive.write(path, name)
    archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
with zipfile.ZipFile(args.output) as archive:
    assert archive.testzip() is None
    for name, record in manifest["files"].items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == record["sha256"]
print(f"Verified {len(entries)} files, including all 48 render PNGs: {args.output}")

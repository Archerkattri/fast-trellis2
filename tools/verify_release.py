"""Check release metadata, local documentation targets and cheap CLI failures."""

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
version = ast.literal_eval(
    next(
        node.value
        for node in ast.parse((ROOT / "trellis2/_version.py").read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )
)
assert f"version: {version}" in (ROOT / "CITATION.cff").read_text()
assert f"## {version}" in (ROOT / "CHANGELOG.md").read_text()
if len(sys.argv) > 1:
    assert sys.argv[1] == "v" + version, "tag and source version differ"

documents = [ROOT / name for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md")]
documents += [path for path in (ROOT / "docs").glob("*.md")]
documents += [ROOT / "benchmarks/README.md"]
for document in documents:
    for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0])
        assert (document.parent / target).exists(), f"{document.name}: missing {target}"

data = json.loads((ROOT / "benchmarks/issue4/results.json").read_text())
assert len(data["four_way"]) == 4
for row in data["four_way"]:
    assert row["sampler_seconds"] > 0 and row["raw_faces"] > 0

for options in (
    ["--seed", "-1"],
    ["--steps", "0"],
    ["--output", "invalid.obj"],
    ["--profile", "sage", "--attention", "native"],
    ["--image", "definitely-missing-input.png"],
):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/generate.py"), *options],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
print(
    f"PASS: {version} metadata, documentation targets, measurements and CLI validation"
)

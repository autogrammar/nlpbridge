#!/usr/bin/env python3
"""Bundle sources, examples, verification and optional local native binaries."""

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    "target",
    "state",
    ".venv",
    "build",
    "dist",
    ".git",
    ".worktrees",
    ".subactor",
    "__pycache__",
    ".ruff_cache",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../artifacts/nlbridge-0.1.0.zip")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(p in EXCLUDED or p.endswith(".egg-info") for p in rel.parts):
            continue
        if (
            path == output
            or rel.name in {"SHA256SUMS.txt"}
            or path.suffix in {".pyc", ".zip"}
        ):
            continue
        files.append((path, rel))
    sums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel.as_posix()}\n"
        for path, rel in files
    )
    manifest = ROOT / "SHA256SUMS.txt"
    manifest.write_text(sums)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, rel in [*files, (manifest, Path("SHA256SUMS.txt"))]:
            archive.write(path, "nlbridge/" + rel.as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(digest + "  " + output.name + "\n")
    print(
        f"{output}: {len(files) + 1} files, {output.stat().st_size} bytes, sha256={digest}"
    )


if __name__ == "__main__":
    main()

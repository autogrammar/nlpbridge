#!/usr/bin/env python3
"""Collect the actual resolved Cargo license/notice files for binary distribution."""

import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
cargo = os.environ.get("NLBRIDGE_CARGO", "cargo")
env = os.environ.copy()
if os.path.isabs(cargo):
    env["PATH"] = str(Path(cargo).parent) + os.pathsep + env.get("PATH", "")
metadata = json.loads(
    subprocess.check_output(
        [
            cargo,
            "metadata",
            "--locked",
            "--offline",
            "--format-version",
            "1",
            "--filter-platform",
            "x86_64-unknown-linux-gnu",
        ],
        cwd=ROOT,
        env=env,
    )
)
sections = [
    "Resolved Cargo dependencies and their upstream notices.\nApplies to the included Linux core and optional ONNX plugin.\nONNX Runtime and model weights are separate downloads and are not redistributed here.\n"
]
missing = []
for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
    if not package["source"]:
        continue
    root = Path(package["manifest_path"]).parent
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.name.upper().startswith(("LICENSE", "NOTICE", "COPYING"))
            or (package.get("license_file") and p == root / package["license_file"])
        )
    )
    sections.append(
        "\n"
        + "=" * 60
        + "\n"
        + f"{package['name']} {package['version']}\n{package.get('license')}\n{package.get('repository')}\n"
    )
    if not files:
        missing.append(package["name"])
    for path in files:
        sections.append(
            "\n"
            + path.relative_to(root).as_posix()
            + "\n"
            + path.read_text(errors="replace")
            + "\n"
        )
(ROOT / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(sections))
print(
    json.dumps(
        {"packages": len(metadata["packages"]), "missing_license_files": missing}
    )
)

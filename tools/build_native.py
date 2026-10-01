#!/usr/bin/env python3
"""Build a portable-for-this-platform release. No target-cpu=native by default."""

import os
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx", action="store_true", help="also build the optional embedding plugin"
)
args = parser.parse_args()
cargo = os.environ.get("NLBRIDGE_CARGO", "cargo")
env = os.environ.copy()
if os.path.isabs(cargo):
    env["PATH"] = str(Path(cargo).parent) + os.pathsep + env.get("PATH", "")
packages = ["core", "onnx"] if args.onnx else ["core"]
command = [cargo, "build", "--release", "--locked"]
for package in packages:
    command.extend(["-p", "nlbridge-" + package])
subprocess.run(command, cwd=root, env=env, check=True)
dest = root / "python" / "nlbridge" / "_native"
dest.mkdir(parents=True, exist_ok=True)
for package in packages:
    stem = "nlbridge_" + package
    name = (
        stem + ".dll"
        if sys.platform == "win32"
        else "lib" + stem + (".dylib" if sys.platform == "darwin" else ".so")
    )
    shutil.copy2(root / "target" / "release" / name, dest / name)
    print(dest / name)

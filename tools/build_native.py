#!/usr/bin/env python3
"""Build a portable-for-this-platform release. No target-cpu=native by default."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
cargo = os.environ.get("NLBRIDGE_CARGO", "cargo")
env = os.environ.copy()
if os.path.isabs(cargo):
    env["PATH"] = str(Path(cargo).parent) + os.pathsep + env.get("PATH", "")
subprocess.run([cargo, "build", "--release", "--locked"], cwd=root, env=env, check=True)
name = (
    "nlbridge_core.dll"
    if sys.platform == "win32"
    else "libnlbridge_core.dylib"
    if sys.platform == "darwin"
    else "libnlbridge_core.so"
)
dest = root / "python" / "nlbridge" / "_native"
dest.mkdir(parents=True, exist_ok=True)
shutil.copy2(root / "target" / "release" / name, dest / name)
print(dest / name)

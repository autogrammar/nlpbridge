#!/usr/bin/env python3
"""Run reproducible checks and save actual stdout/stderr and exit codes."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cargo", default=os.environ.get("NLBRIDGE_CARGO", "cargo"))
    p.add_argument(
        "--onnx",
        action="store_true",
        help="require optional ONNX dependencies/plugin and test both Rust crates",
    )
    ns = p.parse_args()
    env = os.environ.copy()
    if os.path.isabs(ns.cargo):
        env["PATH"] = str(Path(ns.cargo).parent) + os.pathsep + env.get("PATH", "")
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Regression tests with scripted LLM and tiny generated ONNX graphs. Real downloaded-model measurements are separate in onnx-benchmark.json.",
        "checks": [],
        "not_run": [
            "real LLM inference or representative multilingual quality benchmark",
            "full NL->DSL latency with real LLM",
            "Docker build/run",
            "remote GitHub Actions",
            "macOS or Windows build",
        ],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["jsonschema", "referencing", "rpds-py", "attrs"]
        },
    }
    checks = [
        (
            "rust-tests",
            [ns.cargo, "test", "--locked"] + (["--workspace"] if ns.onnx else []),
            {},
        ),
        (
            "python-native",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            {"NLBRIDGE_TEST_BACKEND": "rust"},
        ),
        (
            "python-fallback",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            {
                "NLBRIDGE_TEST_BACKEND": "python",
                "NLBRIDGE_NATIVE": str(ROOT / "state/not-installed.so"),
            },
        ),
        ("example", [sys.executable, "examples/in_process.py"], {}),
    ]
    if ns.onnx:
        checks.insert(
            0,
            (
                "onnx-required",
                [
                    sys.executable,
                    "-c",
                    "import onnx, onnxruntime, tokenizers, numpy; from nlbridge.onnx_provider import native_library; print(native_library())",
                ],
                {},
            ),
        )
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    for name, cmd, overrides in checks:
        start = time.perf_counter()
        run = subprocess.run(
            cmd, cwd=ROOT, env={**env, **overrides}, capture_output=True, text=True
        )
        (out / (name + ".log")).write_text(run.stdout + run.stderr)
        entry = {
            "name": name,
            "exit_code": run.returncode,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "log": name + ".log",
        }
        report["checks"].append(entry)
        print(json.dumps(entry), flush=True)
    report["all_passed"] = all(c["exit_code"] == 0 for c in report["checks"])
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

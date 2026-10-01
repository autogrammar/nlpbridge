#!/usr/bin/env python3
"""Kernel and no-LLM pipeline benchmark. Never a language quality benchmark."""

import argparse
import json
import os
import platform
from pathlib import Path
import random
import statistics
import tempfile
from time import perf_counter
from nlbridge.catalog import CatalogStore, load_catalog
from nlbridge.contracts import Policy
from nlbridge.native import VectorIndex, load_native
from nlbridge.runtime import Runtime


def measure(fn, trials):
    fn()
    values = []
    for _ in range(trials):
        start = perf_counter()
        fn()
        values.append((perf_counter() - start) * 1000)
    values.sort()
    return {
        "p50_ms": round(statistics.median(values), 4),
        "p95_ms": round(values[min(len(values) - 1, int(0.95 * len(values)))], 4),
        "trials": trials,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/benchmark.json")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    p.add_argument("--dim", type=int, default=384)
    args = p.parse_args()
    if args.trials < 3:
        p.error("at least three trials required")
    report = {
        "scope": "synthetic vectors; exact masked top-k; warm process; NO embeddings inference, LLM inference, HTTP, model quality or concurrent load in these timings",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_env": {
            k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        },
        "native_available": load_native() is not None,
        "kernel": [],
        "pipeline": [],
    }
    cpu = Path("/proc/cpuinfo")
    if cpu.exists():
        report["cpu"] = next(
            (
                l.split(":", 1)[1].strip()
                for l in cpu.read_text().splitlines()
                if l.startswith("model name")
            ),
            "unknown",
        )
    for rows in args.sizes:
        rng = random.Random(42)
        vectors = [[rng.uniform(-1, 1) for _ in range(args.dim)] for _ in range(rows)]
        query = [rng.uniform(-1, 1) for _ in range(args.dim)]
        mask = [i % 4 != 0 for i in range(rows)]
        expected = None
        for backend in ["python", "rust", "numpy"]:
            if backend == "rust" and not load_native():
                continue
            try:
                index = VectorIndex(vectors, args.dim, backend)
            except ImportError:
                continue
            hits = index.search(query, mask, 8)
            ids = [i for i, _ in hits]
            if expected is None:
                expected = ids
            elif ids != expected:
                raise AssertionError("backend top-k parity failed")
            result = {
                "backend": backend,
                "rows": rows,
                "dim": args.dim,
                "k": 8,
                "allowed_fraction": 0.75,
                **measure(lambda: index.search(query, mask, 8), args.trials),
            }
            report["kernel"].append(result)
            print(json.dumps(result), flush=True)
    records = load_catalog(
        Path(__file__).resolve().parents[1] / "examples/catalog.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        for backend in ["python", "rust"]:
            if backend == "rust" and not load_native():
                continue
            store = CatalogStore(Path(tmp) / (backend + ".sqlite"), backend)
            store.sync(records)
            for cache in [0, 128]:
                runtime = Runtime(store, policy=Policy(), cache_size=cache)
                result = {
                    "backend": backend,
                    "path": "exact URI + explicit args (no LLM)",
                    "result_cache": bool(cache),
                    **measure(
                        lambda: runtime.compile(
                            "proc://demo/text/word-count/v1",
                            args={"text": "Zażółć gęślą jaźń"},
                        ),
                        max(100, args.trials),
                    ),
                }
                report["pipeline"].append(result)
            store.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

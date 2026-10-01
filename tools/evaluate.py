#!/usr/bin/env python3
"""Live, per-language end-to-end eval against application-owned golden cases.

Reads JSONL {language,text,expected:{status,steps:[{uri,args}]},context?}.
Disable result cache so repeated requests do not hide inference cost.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from time import perf_counter
from nlbridge.common import BridgeError, strict_json
from nlbridge.config import open_runtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cases", default="examples/eval.jsonl")
    p.add_argument("--output", default="reports/live-eval.json")
    ns = p.parse_args()
    runtime, _ = open_runtime(ns.config)
    if runtime.model is None:
        p.error("live evaluation requires a configured chat model")
    runtime.cache_size = 0
    rows = []
    try:
        for line in Path(ns.cases).read_text().splitlines():
            if not line.strip():
                continue
            case = strict_json(line)
            start = perf_counter()
            try:
                actual = runtime.compile(case["text"], case.get("context"))
                expected = case["expected"]
                projected = {"status": actual["status"]}
                if actual["status"] == "ready":
                    projected["steps"] = [
                        {"uri": s["uri"], "args": s["args"]}
                        for s in actual["plan"]["steps"]
                    ]
                row = {
                    "language": case["language"],
                    "ok": projected == expected,
                    "expected": expected,
                    "actual": projected,
                    "metrics": actual["metrics"],
                }
            except BridgeError as e:
                row = {"language": case["language"], "ok": False, "error": str(e)}
            row["elapsed_ms"] = (perf_counter() - start) * 1000
            rows.append(row)
    finally:
        runtime.store.close()
    groups = defaultdict(list)
    for row in rows:
        groups[row["language"]].append(row)
    summary = {}
    for language, group in groups.items():
        times = sorted(r["elapsed_ms"] for r in group)
        summary[language] = {
            "n": len(group),
            "exact_plan_accuracy": sum(r["ok"] for r in group) / len(group),
            "p50_ms": statistics.median(times),
            "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
        }
    result = {
        "model": runtime.model.descriptor,
        "embedding": getattr(runtime.embedder, "descriptor", None),
        "cold_first_request_included": True,
        "languages": summary,
        "cases": rows,
    }
    out = Path(ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

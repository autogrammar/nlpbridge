#!/usr/bin/env python3
"""Real local embedding parity/latency; no LLM or model-quality claims."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import tempfile
from time import perf_counter
import numpy as np
from nlbridge.cards import selection_card
from nlbridge.catalog import CatalogStore, index_text, load_catalog
from nlbridge.common import canonical
from nlbridge.contracts import operation, summary
from nlbridge.onnx_provider import ONNXEmbedder, RustONNXEmbedder

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "proc://demo/text/uppercase/v1",
    "proc://demo/text/word-count/v1",
    "proc://media/images/svg-to-png/v1",
    "proc://media/images/png-to-svg/v1",
]
QUERIES = {
    "pl": [
        "Zamień podany tekst na wielkie litery.",
        "Policz słowa oddzielone spacjami w tekście.",
        "Zamień logo.svg na logo.png i zachowaj oryginał.",
        "Zamień logo.png na logo.svg i zachowaj oryginał.",
    ],
    "en": [
        "Convert the supplied text to uppercase.",
        "Count whitespace-separated words in the text.",
        "Convert logo.svg to logo.png and keep the original.",
        "Convert logo.png to logo.svg and keep the original.",
    ],
    "de": [
        "Wandle den Text in Großbuchstaben um.",
        "Zähle die durch Leerzeichen getrennten Wörter im Text.",
        "Konvertiere logo.svg in logo.png und behalte das Original.",
        "Konvertiere logo.png in logo.svg und behalte das Original.",
    ],
    "fr": [
        "Convertis le texte en majuscules.",
        "Compte les mots séparés par des espaces dans le texte.",
        "Convertis logo.svg en logo.png et conserve l’original.",
        "Convertis logo.png en logo.svg et conserve l’original.",
    ],
    "es": [
        "Convierte el texto a mayúsculas.",
        "Cuenta las palabras separadas por espacios en el texto.",
        "Convierte logo.svg a logo.png y conserva el original.",
        "Convierte logo.png a logo.svg y conserva el original.",
    ],
}


def distribution(values):
    return {
        "n": len(values),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", default="models/multilingual-e5-small/manifest.json"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", default="reports/onnx-benchmark.json")
    args = parser.parse_args()
    if args.trials < 20:
        parser.error("at least 20 trials required")
    manifest = json.loads(Path(args.bundle).read_text())
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": args.threads,
        "scope": "Warm CPU embedding calls including tokenizer, pooling, normalization and Python/FFI transfer. No LLM, HTTP or full NL->DSL. Retrieval smoke is 20 hand-written queries against four operations, not a representative quality benchmark.",
        "manifest": manifest,
        "backends": {},
    }
    report["cpu"] = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    models = {}
    for name, cls in [("python_onnx", ONNXEmbedder), ("rust_onnx", RustONNXEmbedder)]:
        start = perf_counter()
        models[name] = cls(args.bundle, threads=args.threads)
        report["backends"][name] = {
            "load_ms": round((perf_counter() - start) * 1000, 3),
            "descriptor": models[name].descriptor,
            "info": models[name].info,
            "samples": [],
        }
    cases = [
        (lang, text, TARGETS[i])
        for lang, texts in QUERIES.items()
        for i, text in enumerate(texts)
    ]
    for model in models.values():
        for _, text, _ in cases:
            model.embed_query(text)
    order = list(range(args.trials))
    random.Random(17).shuffle(order)
    vectors = {name: {} for name in models}
    for n, item in enumerate(order):
        index = item % len(cases)
        lang, text, _ = cases[index]
        names = list(models) if n % 2 else list(reversed(models))
        for name in names:
            start = perf_counter()
            vector = models[name].embed_query(text)
            elapsed = (perf_counter() - start) * 1000
            vectors[name][index] = vector
            report["backends"][name]["samples"].append(
                {"query_index": index, "lang": lang, "ms": round(elapsed, 5)}
            )
    records = [operation(r) for r in load_catalog(ROOT / "examples/catalog.json")]
    docs = [index_text(r) for r in records]
    doc_vectors = {
        name: np.array(m.embed_documents(docs)) for name, m in models.items()
    }
    differences = [
        float(
            np.max(
                np.abs(np.array(vectors["python_onnx"][i]) - vectors["rust_onnx"][i])
            )
        )
        for i in range(len(cases))
    ]
    report["max_abs_query_difference"] = max(differences)
    report["max_abs_document_difference"] = float(
        np.max(np.abs(doc_vectors["python_onnx"] - doc_vectors["rust_onnx"]))
    )
    report["parity_passed"] = (
        max(report["max_abs_query_difference"], report["max_abs_document_difference"])
        <= 1e-5
    )
    for name, model in models.items():
        result = report["backends"][name]
        result["latency"] = distribution([s["ms"] for s in result["samples"]])
        result["by_language"] = {}
        result["retrieval_cases"] = []
        for i, (lang, text, target) in enumerate(cases):
            rank = np.argsort(-(doc_vectors[name] @ vectors[name][i]), kind="stable")
            uris = [records[j]["uri"] for j in rank]
            result["retrieval_cases"].append(
                {
                    "lang": lang,
                    "query": text,
                    "target": target,
                    "ranking": uris,
                    "hit1": target == uris[0],
                    "hit2": target in uris[:2],
                }
            )
        for lang in QUERIES:
            group = [c for c in result["retrieval_cases"] if c["lang"] == lang]
            result["by_language"][lang] = {
                **distribution(
                    [s["ms"] for s in result["samples"] if s["lang"] == lang]
                ),
                "retrieval_smoke_recall1": sum(c["hit1"] for c in group) / len(group),
                "retrieval_smoke_recall2": sum(c["hit2"] for c in group) / len(group),
            }
        # Real indexing verifies that the same model identity reuses all stored vectors.
        with tempfile.TemporaryDirectory() as tmp:
            store = CatalogStore(Path(tmp) / "cache.sqlite")
            try:
                start = perf_counter()
                result["first_index"] = store.sync(records, model)
                result["first_index_ms"] = round((perf_counter() - start) * 1000, 3)
                result["second_index"] = store.sync(records, model)
                if result["second_index"]["embedded"] != 0:
                    raise RuntimeError("vector cache was not reused")
            finally:
                store.close()
    full = len(canonical([summary(r) for r in records]).encode())
    compact = len(canonical([selection_card(r) for r in records]).encode())
    report["selection_cards"] = {
        "full_bytes": full,
        "compact_bytes": compact,
        "reduction_percent": round((1 - compact / full) * 100, 2),
        "scope": "operation payload only; bytes are not model tokens",
    }
    report["ranking_parity"] = [
        c["ranking"] for c in report["backends"]["python_onnx"]["retrieval_cases"]
    ] == [c["ranking"] for c in report["backends"]["rust_onnx"]["retrieval_cases"]]
    models["rust_onnx"].close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "parity_passed": report["parity_passed"],
                "ranking_parity": report["ranking_parity"],
                "max_abs_query_difference": report["max_abs_query_difference"],
                "selection_cards": report["selection_cards"],
                "latency": {
                    name: result["latency"]
                    for name, result in report["backends"].items()
                },
            },
            indent=2,
        )
    )
    if not report["parity_passed"] or not report["ranking_parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

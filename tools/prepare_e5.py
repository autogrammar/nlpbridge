#!/usr/bin/env python3
"""Opt-in download of the pinned model used in this release's CPU benchmark."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen
from nlbridge.model_bundle import create_bundle

REPO = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
FILES = {
    "model.onnx": "onnx/model_qint8_avx512_vnni.onnx",
    "tokenizer.json": "onnx/tokenizer.json",
    "MODEL_CARD.md": "README.md",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="models/multilingual-e5-small")
    args = p.parse_args()
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, source in FILES.items():
        target = root / name
        # Explicit command owns these outputs. Never trust an unverified partial file.
        temp = root / (name + ".part")
        with (
            urlopen(
                f"https://huggingface.co/{REPO}/resolve/{REVISION}/{source}", timeout=60
            ) as response,
            temp.open("wb") as out,
        ):
            total = 0
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > 200 << 20:
                    raise RuntimeError("model artifact exceeds download budget")
                out.write(chunk)
        temp.replace(target)
        print(f"{target}: {total} bytes")
    provenance = {"repo": REPO, "revision": REVISION, "onnx": FILES["model.onnx"]}
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(create_bundle(root, pad_id=1, provenance=provenance))


if __name__ == "__main__":
    main()

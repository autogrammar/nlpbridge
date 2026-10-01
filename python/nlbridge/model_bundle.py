"""Content-addressed ONNX bundles; absolute paths do not enter cache identities."""

from copy import deepcopy
import hashlib
from pathlib import Path
import re
from .common import BridgeError, canonical, strict_json

FORMAT = "nlbridge/embedding-bundle-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise BridgeError("bundle artifact path must be relative")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise BridgeError("bundle artifact escapes its directory")
    return target


def load_bundle(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    if len(data) > 1 << 20:
        raise BridgeError("bundle manifest exceeds 1 MiB")
    manifest = strict_json(data)
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        - {
            "format",
            "files",
            "model",
            "tokenizer",
            "dim",
            "preprocessing",
            "provenance",
        }
        or manifest.get("format") != FORMAT
    ):
        raise BridgeError("invalid embedding bundle format")
    files = manifest.get("files")
    if not isinstance(files, dict) or not 2 <= len(files) <= 256:
        raise BridgeError("bundle needs model and tokenizer hashes")
    for key in ("model", "tokenizer"):
        if manifest.get(key) not in files:
            raise BridgeError("bundle is missing a required artifact")
    if type(manifest.get("dim")) is not int or not 1 <= manifest["dim"] <= 65536:
        raise BridgeError("invalid embedding dimension")
    for name, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise BridgeError("invalid artifact SHA-256")
        if file_sha256(local_path(path.parent, name)) != digest:
            raise BridgeError("bundle artifact checksum mismatch: " + name)
    if not graph_artifacts(path.parent, manifest["model"]) <= set(files):
        raise BridgeError("bundle omits external ONNX weight hashes")
    prep = manifest.get("preprocessing")
    required = {
        "query_prefix",
        "document_prefix",
        "max_length",
        "pooling",
        "normalize",
        "output_name",
        "pad_id",
    }
    if not isinstance(prep, dict) or set(prep) != required:
        raise BridgeError("invalid preprocessing specification")
    if (
        not all(
            isinstance(prep[k], str)
            for k in ("query_prefix", "document_prefix", "output_name")
        )
        or not prep["output_name"]
    ):
        raise BridgeError("invalid prefixes/output name")
    if prep["pooling"] not in {"mean", "cls", "none"} or prep["normalize"] is not True:
        raise BridgeError("supported pooling: mean/cls/none with L2 normalization")
    if type(prep["max_length"]) is not int or not 2 <= prep["max_length"] <= 8192:
        raise BridgeError("invalid token limit")
    if type(prep["pad_id"]) is not int or not 0 <= prep["pad_id"] < 2**31:
        raise BridgeError("invalid padding token ID")
    # Relative names matter: swapping two external tensor files changes semantics.
    # Absolute deployment directories do not participate in the identity.
    identity = {
        "format": FORMAT,
        "model_sha256": files[manifest["model"]],
        "tokenizer_sha256": files[manifest["tokenizer"]],
        "artifacts": dict(sorted(files.items())),
        "dim": manifest["dim"],
        "preprocessing": deepcopy(prep),
    }
    return manifest, path.parent, identity


def graph_artifacts(root, model):
    """Inspect all graph messages without loading external weights."""
    try:
        import onnx
    except ImportError as e:
        raise BridgeError("install nlbridge[onnx] to validate the model bundle") from e
    model_path = local_path(root, model)
    try:
        graph = onnx.load_model(model_path, load_external_data=False)
    except Exception as e:
        raise BridgeError("invalid ONNX graph") from e
    paths = {model}

    def visit(message):
        if (
            isinstance(message, onnx.TensorProto)
            and message.data_location == onnx.TensorProto.EXTERNAL
        ):
            info = {entry.key: entry.value for entry in message.external_data}
            external = local_path(model_path.parent, info["location"])
            if not external.is_relative_to(root):
                raise BridgeError("external ONNX weights escape the bundle")
            paths.add(external.relative_to(root).as_posix())
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                for child in value if field.is_repeated else [value]:
                    visit(child)

    visit(graph)
    return paths


def create_bundle(
    directory,
    model="model.onnx",
    tokenizer="tokenizer.json",
    dim=384,
    query_prefix="query: ",
    document_prefix="passage: ",
    max_length=512,
    pooling="mean",
    output_name="last_hidden_state",
    pad_id=0,
    provenance=None,
):
    """Build a manifest including graph, tokenizer and external weight hashes."""
    root = Path(directory).resolve()
    paths = graph_artifacts(root, model) | {tokenizer}
    manifest = {
        "format": FORMAT,
        "files": {p: file_sha256(local_path(root, p)) for p in sorted(paths)},
        "model": model,
        "tokenizer": tokenizer,
        "dim": dim,
        "preprocessing": {
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "max_length": max_length,
            "pooling": pooling,
            "normalize": True,
            "output_name": output_name,
            "pad_id": pad_id,
        },
        "provenance": provenance or {},
    }
    target = root / "manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(canonical(manifest) + "\n")
    temporary.replace(target)
    return target

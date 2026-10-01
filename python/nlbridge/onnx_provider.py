"""Local CPU embeddings with the same bundle contract in Python and Rust."""

import ctypes as C
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import threading
from .common import BridgeError, ModelError, canonical, strict_json
from .model_bundle import file_sha256, load_bundle, local_path


def find_runtime_library():
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None:
        raise BridgeError("install nlpbridge[onnx] or supply runtime_library")
    root = Path(spec.origin).parent / "capi"
    candidates = sorted(
        p
        for p in root.iterdir()
        if p.name.startswith("libonnxruntime.so.")
        or p.name in {"onnxruntime.dll", "libonnxruntime.dylib"}
    )
    if len(candidates) != 1:
        raise BridgeError("cannot locate a unique ONNX Runtime library")
    return candidates[0]


def native_library():
    name = (
        "nlbridge_onnx.dll"
        if sys.platform == "win32"
        else "libnlbridge_onnx.dylib"
        if sys.platform == "darwin"
        else "libnlbridge_onnx.so"
    )
    path = Path(__file__).parent / "_native" / name
    if not path.is_file():
        raise BridgeError(
            "build the optional plugin: python tools/build_native.py --onnx"
        )
    return path


class BundleEmbedder:
    def __init__(self, bundle, threads=2, batch_size=8):
        if (
            type(threads) is not int
            or not 1 <= threads <= 64
            or type(batch_size) is not int
            or not 1 <= batch_size <= 32
        ):
            raise BridgeError("threads must be 1..64 and batch_size 1..32")
        self.manifest, self.root, identity = load_bundle(bundle)
        self.dim = self.manifest["dim"]
        self.prep = self.manifest["preprocessing"]
        self.threads, self.batch_size = threads, batch_size
        self.descriptor = {
            "bundle": identity,
            "threads": threads,
            "batch_size": batch_size,
            "implementation": "nlbridge/onnx-v1",
        }
        self.lock = threading.RLock()

    def _encode(self, texts, prefix):
        if not isinstance(texts, (list, tuple)) or any(
            not isinstance(t, str) for t in texts
        ):
            raise ModelError("embedding input must be a list of strings")
        result = []
        with self.lock:
            for pos in range(0, len(texts), self.batch_size):
                batch = [prefix + t for t in texts[pos : pos + self.batch_size]]
                if sum(len(t.encode()) for t in batch) > 1 << 20:
                    raise ModelError("embedding batch exceeds 1 MiB")
                result.extend(self._batch(batch))
        return result

    def embed_documents(self, texts):
        return self._encode(texts, self.prep["document_prefix"])

    def embed_query(self, text):
        return self._encode([text], self.prep["query_prefix"])[0]


class ONNXEmbedder(BundleEmbedder):
    """Reference backend; ONNX Runtime already runs inference in native code."""

    def __init__(self, bundle, threads=2, batch_size=8):
        super().__init__(bundle, threads, batch_size)
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise BridgeError("install nlpbridge[onnx]") from e
        self.np = np
        self.tokenizer = Tokenizer.from_file(
            str(local_path(self.root, self.manifest["tokenizer"]))
        )
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        try:
            self.session = ort.InferenceSession(
                str(local_path(self.root, self.manifest["model"])),
                options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise BridgeError("cannot open ONNX model: " + str(e)) from e
        self.inputs = self.session.get_inputs()
        if not {"input_ids", "attention_mask"} <= {i.name for i in self.inputs} or any(
            i.name not in {"input_ids", "attention_mask", "token_type_ids"}
            or i.type not in {"tensor(int64)", "tensor(int32)"}
            or len(i.shape) != 2
            for i in self.inputs
        ):
            raise BridgeError(
                "ONNX graph requires rank-2 int32/int64 input_ids, attention_mask and optional token_type_ids"
            )
        output = next(
            (
                o
                for o in self.session.get_outputs()
                if o.name == self.prep["output_name"]
            ),
            None,
        )
        if (
            output is None
            or output.type != "tensor(float)"
            or len(output.shape) != (2 if self.prep["pooling"] == "none" else 3)
        ):
            raise BridgeError("ONNX output name/type/rank does not match pooling")
        self.descriptor.update(
            provider="onnx",
            runtime_version=ort.__version__,
            runtime_sha256=file_sha256(find_runtime_library()),
            tokenizer_version=importlib.metadata.version("tokenizers"),
            numpy_version=np.__version__,
        )
        self.info = {
            "runtime": ort.__version__,
            "threads": threads,
            "dim": self.dim,
            "inputs": [{"name": i.name, "dtype": i.type} for i in self.inputs],
        }

    def _batch(self, texts):
        np = self.np
        encoded = [self.tokenizer.encode(t, add_special_tokens=True) for t in texts]
        if any(not e.ids or len(e.ids) > self.prep["max_length"] for e in encoded):
            raise ModelError(
                "input exceeds embedding token budget or tokenizes to empty; no text was truncated"
            )
        shape = (len(encoded), max(len(e.ids) for e in encoded))
        ids = np.full(shape, self.prep["pad_id"], dtype=np.int64)
        mask = np.zeros(shape, dtype=np.int64)
        types = np.zeros(shape, dtype=np.int64)
        for b, e in enumerate(encoded):
            ids[b, : len(e.ids)] = e.ids
            mask[b, : len(e.ids)] = e.attention_mask
            types[b, : len(e.ids)] = e.type_ids
        arrays = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        feed = {
            i.name: arrays[i.name].astype(
                np.int32 if i.type == "tensor(int32)" else np.int64, copy=False
            )
            for i in self.inputs
        }
        try:
            out = self.session.run([self.prep["output_name"]], feed)[0]
        except Exception as e:
            raise ModelError("ONNX inference failed: " + str(e)) from e
        expected = (
            (shape[0], self.dim)
            if self.prep["pooling"] == "none"
            else (*shape, self.dim)
        )
        if out.shape != expected:
            raise ModelError("ONNX output dimensions mismatch")
        if self.prep["pooling"] == "none":
            rows = out.astype(np.float64)
        elif self.prep["pooling"] == "cls":
            rows = out[:, 0, :].astype(np.float64)
        else:
            weights = (mask != 0)[..., None]
            counts = weights.sum(axis=1)
            if np.any(counts == 0):
                raise ModelError("empty attention mask")
            rows = np.where(weights, out, 0).sum(axis=1, dtype=np.float64) / counts
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ModelError("embedding is non-finite or zero")
        return (rows / norms).astype(np.float32).tolist()


class RustONNXEmbedder(BundleEmbedder):
    def __init__(
        self, bundle, threads=2, batch_size=8, runtime_library=None, native_path=None
    ):
        super().__init__(bundle, threads, batch_size)
        runtime = (
            Path(runtime_library).resolve()
            if runtime_library
            else find_runtime_library()
        )
        plugin = Path(native_path).resolve() if native_path else native_library()
        self.lib = C.CDLL(str(plugin))
        self.handle = None
        self.lib.nlb_onnx_abi_version.restype = C.c_uint32
        if self.lib.nlb_onnx_abi_version() != 1:
            raise BridgeError("incompatible ONNX plugin ABI")
        self.lib.nlb_onnx_open.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.nlb_onnx_open.restype = C.c_void_p
        self.lib.nlb_onnx_embed.argtypes = [C.c_void_p, C.c_char_p]
        self.lib.nlb_onnx_embed.restype = C.c_void_p
        self.lib.nlb_onnx_info.argtypes = [C.c_void_p]
        self.lib.nlb_onnx_info.restype = C.c_void_p
        self.lib.nlb_onnx_free.argtypes = [C.c_void_p]
        self.lib.nlb_onnx_free.restype = None
        self.lib.nlb_onnx_string_free.argtypes = [C.c_void_p]
        self.lib.nlb_onnx_string_free.restype = None
        config = {
            k: self.prep[k] for k in ("output_name", "pooling", "max_length", "pad_id")
        }
        config.update(
            runtime_library=str(runtime),
            model=str(local_path(self.root, self.manifest["model"])),
            tokenizer=str(local_path(self.root, self.manifest["tokenizer"])),
            dim=self.dim,
            threads=threads,
        )
        error = C.c_void_p()
        self.handle = self.lib.nlb_onnx_open(canonical(config).encode(), C.byref(error))
        if not self.handle:
            message = (
                self._take(error.value)
                if error.value
                else {"error": "ONNX plugin initialization failed"}
            )
            raise BridgeError(message.get("error"))
        self.info = self._take(self.lib.nlb_onnx_info(self.handle))
        self.descriptor.update(
            provider="rust_onnx",
            runtime_sha256=file_sha256(runtime),
            plugin_sha256=file_sha256(plugin),
            runtime_info=self.info["runtime"],
        )

    def _take(self, pointer):
        if not pointer:
            raise ModelError("null ONNX plugin response")
        try:
            return strict_json(C.string_at(pointer))
        finally:
            self.lib.nlb_onnx_string_free(pointer)

    def _batch(self, texts):
        if not self.handle:
            raise ModelError("ONNX plugin is closed")
        result = self._take(
            self.lib.nlb_onnx_embed(self.handle, canonical(texts).encode())
        )
        if "error" in result:
            raise ModelError(result["error"])
        return result["vectors"]

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.nlb_onnx_free(self.handle)
                self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()

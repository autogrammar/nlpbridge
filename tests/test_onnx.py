from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from nlbridge.common import BridgeError, ModelError, fingerprint
from nlbridge.model_bundle import create_bundle, file_sha256, load_bundle
from nlbridge.onnx_provider import ONNXEmbedder, RustONNXEmbedder, native_library

AVAILABLE = all(
    importlib.util.find_spec(p) for p in ("onnx", "onnxruntime", "tokenizers", "numpy")
)


def fixture(root, dtype="int64", external=False):
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper
    from tokenizers import Tokenizer, models, pre_tokenizers

    root.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "hello": 2, "świat": 3, "bonjour": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(
        max_length=2
    )  # Providers must override this, never silently truncate.
    tokenizer.save(str(root / "tokenizer.json"))
    weights = numpy_helper.from_array(
        np.arange(20, dtype=np.float32).reshape(5, 4) + 1, "weights"
    )
    dt = onnx.TensorProto.INT32 if dtype == "int32" else onnx.TensorProto.INT64
    graph = h.make_graph(
        [h.make_node("Gather", ["weights", "input_ids"], ["last_hidden_state"])],
        "fixture",
        [
            h.make_tensor_value_info("input_ids", dt, ["b", "s"]),
            h.make_tensor_value_info("attention_mask", dt, ["b", "s"]),
        ],
        [
            h.make_tensor_value_info(
                "last_hidden_state", onnx.TensorProto.FLOAT, ["b", "s", 4]
            )
        ],
        [weights],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    onnx.save_model(
        model,
        root / "model.onnx",
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    return create_bundle(root, dim=4, query_prefix="", document_prefix="", max_length=8)


@unittest.skipUnless(AVAILABLE, "install nlpbridge[onnx] for optional ONNX tests")
class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_identity_portable_and_preprocessing_sensitive(self):
        path = fixture(self.root / "one")
        shutil.copytree(path.parent, self.root / "two")
        a = load_bundle(path)[2]
        b = load_bundle(self.root / "two/manifest.json")[2]
        self.assertEqual(fingerprint(a), fingerprint(b))
        manifest = json.loads(path.read_text())
        manifest["preprocessing"]["query_prefix"] = "different: "
        path.write_text(json.dumps(manifest))
        self.assertNotEqual(fingerprint(a), fingerprint(load_bundle(path)[2]))

    def test_tokenizer_tampering_detected(self):
        path = fixture(self.root)
        with (self.root / "tokenizer.json").open("a") as f:
            f.write(" ")
        with self.assertRaisesRegex(BridgeError, "checksum"):
            load_bundle(path)

    def test_artifact_names_are_bound_to_hashes(self):
        path = fixture(self.root)
        manifest = json.loads(path.read_text())
        for name, value in [("a.bin", b"aaa"), ("b.bin", b"bbb")]:
            (self.root / name).write_bytes(value)
            manifest["files"][name] = file_sha256(self.root / name)
        path.write_text(json.dumps(manifest))
        before = fingerprint(load_bundle(path)[2])
        for name, value in [("a.bin", b"bbb"), ("b.bin", b"aaa")]:
            (self.root / name).write_bytes(value)
            manifest["files"][name] = file_sha256(self.root / name)
        path.write_text(json.dumps(manifest))
        self.assertNotEqual(before, fingerprint(load_bundle(path)[2]))

    def test_external_weights_must_be_hashed(self):
        path = fixture(self.root, external=True)
        manifest, _, _ = load_bundle(path)
        self.assertIn("weights.bin", manifest["files"])
        del manifest["files"]["weights.bin"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(BridgeError, "omits external"):
            load_bundle(path)

    def test_path_escape_rejected(self):
        path = fixture(self.root / "bundle")
        manifest = json.loads(path.read_text())
        manifest["files"]["../outside"] = "a" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(BridgeError, "escapes"):
            load_bundle(path)

    def test_python_token_budget_and_padding(self):
        import numpy as np

        emb = ONNXEmbedder(fixture(self.root))
        single = emb.embed_query("hello")
        batch = emb.embed_documents(["hello", "hello świat bonjour"])
        np.testing.assert_allclose(single, batch[0], atol=1e-7)
        with self.assertRaisesRegex(ModelError, "no text was truncated"):
            emb.embed_query("hello " * 9)

    def test_rust_matches_python_int64_int32_and_threads(self):
        import numpy as np

        try:
            native_library()
        except BridgeError:
            self.skipTest("build native ONNX plugin")
        for dtype in ("int64", "int32"):
            with self.subTest(dtype=dtype):
                path = fixture(self.root / dtype, dtype)
                py, rs = ONNXEmbedder(path), RustONNXEmbedder(path)
                self.addCleanup(rs.close)
                texts = ["hello", "hello świat bonjour"]
                np.testing.assert_allclose(
                    py.embed_documents(texts), rs.embed_documents(texts), atol=1e-7
                )
                with ThreadPoolExecutor(max_workers=4) as pool:
                    vectors = list(pool.map(rs.embed_query, texts * 4))
                for i, vector in enumerate(vectors):
                    np.testing.assert_allclose(
                        vector, py.embed_query(texts[i % 2]), atol=1e-7
                    )
                with self.assertRaisesRegex(ModelError, "no text was truncated"):
                    rs.embed_query("hello " * 9)
                rs.close()
                with self.assertRaisesRegex(ModelError, "closed"):
                    rs.embed_query("hello")

    def test_output_dimension_failure(self):
        path = fixture(self.root)
        manifest = json.loads(path.read_text())
        manifest["dim"] = 7
        path.write_text(json.dumps(manifest))
        providers = [ONNXEmbedder]
        try:
            native_library()
            providers.append(RustONNXEmbedder)
        except BridgeError:
            pass
        for provider in providers:
            emb = provider(path)
            if hasattr(emb, "close"):
                self.addCleanup(emb.close)
            with self.assertRaisesRegex(ModelError, "dimensions mismatch"):
                emb.embed_query("hello")


if __name__ == "__main__":
    unittest.main()

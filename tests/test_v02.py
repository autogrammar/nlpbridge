import copy
import http.client
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import threading
import unittest

from nlbridge.cards import schema_hint, selection_card
from nlbridge.catalog import CatalogStore
from nlbridge.common import BridgeError, canonical
from nlbridge.contracts import operation, summary
from nlbridge.runtime import Runtime
from nlbridge.server import make_server
from nlbridge.vector_codec import ENCODING, decode_vector, encode_vector
from test_bridge import RECORDS, UP, POLICY, ScriptedModel, TestEmbedder, ready, select


class CompactTests(unittest.TestCase):
    def test_constraints_preserved(self):
        schema = {
            "type": "object",
            "properties": {
                "source": {"type": "string", "pattern": "\\.svg$"},
                "target_format": {"const": "png"},
                "keep_source": {"type": "boolean", "const": True},
            },
            "required": ["source"],
            "additionalProperties": False,
        }
        hint = schema_hint(schema)
        self.assertEqual(hint["object"]["target_format"], {"const": "png"})
        self.assertEqual(hint["object"]["source"]["pattern"], "\\.svg$")
        self.assertTrue(hint["object"]["keep_source"]["const"])
        self.assertEqual(hint["required"], ["source"])
        self.assertFalse(hint["additionalProperties"])
        self.assertEqual(
            schema_hint({"anyOf": [{"type": "string"}, {"enum": [1, 2]}]}),
            {"anyOf": [{"type": "string"}, {"enum": [1, 2]}]},
        )

    def test_no_silent_description_truncation(self):
        op = copy.deepcopy(RECORDS[0])
        op["desc"] = "A" * 2000 + " Do not delete the source."
        op = operation(op)
        self.assertEqual(selection_card(op)["desc"], op["desc"])
        op["selection_description"] = "Explicit author summary"
        self.assertEqual(
            selection_card(operation(op))["desc"], "Explicit author summary"
        )

    def test_smaller_selection_full_binding_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CatalogStore(Path(tmp) / "c.sqlite", "python")
            self.addCleanup(store.close)
            store.sync(RECORDS)
            self.assertLess(
                len(canonical([selection_card(o) for o in store.snapshot.operations])),
                len(canonical([summary(o) for o in store.snapshot.operations])),
            )
            model = ScriptedModel(select(UP), ready({"text": "Zażółć"}))
            runtime = Runtime(store, model, policy=POLICY)
            result = runtime.compile("Użyj uppercase dla Zażółć")
            first = json.loads(model.calls[0][0][1]["content"])
            second = json.loads(model.calls[1][0][1]["content"])
            self.assertIn("in", first["operations"][0])
            self.assertEqual(second["operation"], summary(store.snapshot.by_uri[UP]))
            self.assertEqual(
                set(result["metrics"]["prompt_bytes_by_phase"]), {"select", "bind"}
            )
            self.assertEqual(
                result["metrics"]["prompt_bytes"],
                sum(result["metrics"]["prompt_bytes_by_phase"].values()),
            )


class CacheHealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "c.sqlite"

    def store(self):
        s = CatalogStore(self.path, "python")
        self.addCleanup(s.close)
        return s

    def test_portable_vector_bytes_and_invalid_blobs(self):
        self.assertEqual(
            encode_vector([1.0, -0.5]), b"\x00\x00\x80\x3f\x00\x00\x00\xbf"
        )
        self.assertEqual(
            list(decode_vector(struct.pack("<2f", 1.0, 2.0), 2)), [1.0, 2.0]
        )
        for blob, dim in [
            (b"abc", 1),
            (struct.pack("<f", float("nan")), 1),
            (struct.pack("<f", 0.0), 1),
            (b"", 0),
        ]:
            with self.subTest(blob=blob), self.assertRaises(BridgeError):
                decode_vector(blob, dim)

    def test_legacy_and_corrupt_cache_rebuilt(self):
        c = sqlite3.connect(self.path)
        c.execute(
            "CREATE TABLE vectors(model TEXT,doc_hash TEXT,dim INTEGER,data BLOB,PRIMARY KEY(model,doc_hash))"
        )
        c.execute("INSERT INTO vectors VALUES('old','old',1,?)", (b"garbage",))
        c.commit()
        c.close()
        store = self.store()
        emb = TestEmbedder()
        store.sync(RECORDS, emb)
        self.assertEqual(
            store.conn.execute("SELECT DISTINCT encoding FROM vectors").fetchall(),
            [(ENCODING,)],
        )
        store.conn.execute(
            "UPDATE vectors SET data=? WHERE rowid=(SELECT min(rowid) FROM vectors)",
            (b"bad",),
        )
        store.conn.commit()
        stats = store.sync(RECORDS, emb)
        self.assertEqual(stats["cache_repaired"], 1)
        self.assertEqual(stats["embedded"], 1)
        self.assertEqual(store.sync(RECORDS, emb)["embedded"], 0)

    def test_reads_and_health_during_failed_reindex(self):
        store = self.store()
        emb = TestEmbedder()
        store.sync(RECORDS, emb)
        old = store.snapshot
        entered, release = threading.Event(), threading.Event()

        class Blocked(TestEmbedder):
            def embed_documents(self, texts):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test synchronization timeout")
                raise BridgeError("controlled model failure")

        failures = []

        def run():
            try:
                store.sync(RECORDS, Blocked("new"))
            except BridgeError as e:
                failures.append(e)

        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertTrue(store.health()["syncing"])
            self.assertTrue(store.health()["ready"])
            self.assertIs(store.snapshot, old)
            self.assertEqual(
                Runtime(store, embedder=emb).compile(UP, args={"text": "ok"})["status"],
                "ready",
            )
        finally:
            release.set()
            worker.join(5)
        self.assertEqual(len(failures), 1)
        self.assertIs(store.snapshot, old)
        self.assertEqual(store.health()["status"], "degraded")
        self.assertEqual(store.health()["generation"], 1)
        self.assertEqual(store.health()["active_vectors"], len(RECORDS))
        store.sync(RECORDS, emb)
        self.assertIsNone(store.health()["last_error"])

    def test_readiness_http_does_not_depend_on_llm_configuration(self):
        store = self.store()
        runtime = Runtime(store)
        server = make_server(runtime, "127.0.0.1", 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            for endpoint, status in [("/health", 200), ("/ready", 503)]:
                conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
                conn.request("GET", endpoint)
                response = conn.getresponse()
                self.assertEqual(response.status, status)
                self.assertFalse(json.loads(response.read())["ready"])
                conn.close()
            store.sync(RECORDS)
            self.assertTrue(runtime.health()["ready"])
            self.assertFalse(runtime.health()["model_configured"])
            runtime.embedder = TestEmbedder()
            self.assertFalse(runtime.health()["ready"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations
from array import array
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import threading
from .common import BridgeError, canonical, fingerprint, strict_json
from .contracts import operation
from .native import VectorIndex


def index_text(op):
    return canonical(
        {k: op[k] for k in ("uri", "desc", "input_schema", "output_schema")}
    )


def words(text):
    return set(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE))


@dataclass(frozen=True)
class Snapshot:
    revision: str
    operations: tuple
    by_uri: dict
    index: VectorIndex | None
    lexical: tuple
    embedding_id: str

    def search(self, query, policy, limit, vector=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise BridgeError("limit must be 1..100")
        mask = [policy.allows(op) for op in self.operations]
        exact = self.by_uri.get(query.strip())
        if exact:
            return [exact] if policy.allows(exact) else []
        tokens = words(query)
        lex = sorted(
            (
                (i, len(tokens & self.lexical[i]))
                for i in range(len(mask))
                if mask[i] and tokens & self.lexical[i]
            ),
            key=lambda x: (-x[1], x[0]),
        )[: max(32, limit * 4)]
        dense = (
            self.index.search(vector, mask, max(32, limit * 4))
            if self.index and vector is not None
            else []
        )
        if not self.index and sum(mask) <= limit:
            ranked = [i for i, _ in lex]
            ranked.extend(i for i, x in enumerate(mask) if x and i not in ranked)
            return [self.operations[i] for i in ranked]
        if not dense and not lex:
            # With no embedding model, a small catalog can still be routed by the LLM.
            eligible = [i for i, x in enumerate(mask) if x]
            if not self.index and len(eligible) <= limit:
                return [self.operations[i] for i in eligible]
            return []
        ranks = {}
        for ranking in (dense, lex):
            for rank, (i, _) in enumerate(ranking, 1):
                ranks[i] = ranks.get(i, 0.0) + 1.0 / (60 + rank)
        chosen = sorted(ranks, key=lambda i: (-ranks[i], self.operations[i]["uri"]))[
            :limit
        ]
        return [self.operations[i] for i in chosen]


class CatalogStore:
    def __init__(self, path, backend="auto"):
        self.path = str(path)
        self.backend = backend
        self.lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS operations(uri TEXT PRIMARY KEY,record TEXT NOT NULL,doc_hash TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS vectors(model TEXT NOT NULL,doc_hash TEXT NOT NULL,dim INTEGER NOT NULL,data BLOB NOT NULL,PRIMARY KEY(model,doc_hash));
        """)
        self.snapshot = None

    def close(self):
        self.conn.close()

    def sync(self, records, embedder=None):
        with self.lock:
            ops = sorted((operation(r) for r in records), key=lambda x: x["uri"])
            if len({o["uri"] for o in ops}) != len(ops):
                raise BridgeError("duplicate operation URI")
            if len(ops) > 100000:
                raise BridgeError("catalog exceeds 100,000 operations")
            descriptor = embedder.descriptor if embedder else {"provider": "none"}
            model_id = fingerprint(descriptor)
            old = {
                u: (r, h)
                for u, r, h in self.conn.execute(
                    "SELECT uri,record,doc_hash FROM operations"
                )
            }
            encoded = {o["uri"]: canonical(o) for o in ops}
            docs = {o["uri"]: index_text(o) for o in ops}
            hashes = {
                u: fingerprint({"template": "nlbridge/index-v1", "text": t})
                for u, t in docs.items()
            }
            blobs = {}
            pending = {}
            dim = embedder.dim if embedder else 0
            if embedder:
                for h in set(hashes.values()):
                    row = self.conn.execute(
                        "SELECT dim,data FROM vectors WHERE model=? AND doc_hash=?",
                        (model_id, h),
                    ).fetchone()
                    if row and row[0] == dim:
                        v = array("f")
                        v.frombytes(row[1])
                        blobs[h] = v
                missing = sorted(set(hashes.values()) - set(blobs))
                text_by_hash = {hashes[u]: t for u, t in docs.items()}
                if missing:
                    vectors = embedder.embed_documents(
                        [text_by_hash[h] for h in missing]
                    )
                    if len(vectors) != len(missing):
                        raise BridgeError("embedding count mismatch")
                    for h, v in zip(missing, vectors):
                        if len(v) != dim:
                            raise BridgeError("embedding dimension changed")
                        a = array("f", v)
                        blobs[h] = a
                        pending[h] = a.tobytes()
            # Build/validate before opening a write transaction; failure leaves prior state intact.
            index = (
                VectorIndex([blobs[hashes[o["uri"]]] for o in ops], dim, self.backend)
                if embedder
                else None
            )
            revision = fingerprint({"format": "nlbridge/catalog-v1", "operations": ops})
            new_snapshot = Snapshot(
                revision,
                tuple(ops),
                {o["uri"]: o for o in ops},
                index,
                tuple(words(index_text(o)) for o in ops),
                model_id,
            )
            written = 0
            removed = set(old) - set(encoded)
            with self.conn:
                for h, data in pending.items():
                    self.conn.execute(
                        "INSERT INTO vectors VALUES(?,?,?,?) ON CONFLICT(model,doc_hash) DO UPDATE SET dim=excluded.dim,data=excluded.data",
                        (model_id, h, dim, data),
                    )
                for u in removed:
                    self.conn.execute("DELETE FROM operations WHERE uri=?", (u,))
                for u, raw in encoded.items():
                    if old.get(u) != (raw, hashes[u]):
                        self.conn.execute(
                            "INSERT INTO operations VALUES(?,?,?) ON CONFLICT(uri) DO UPDATE SET record=excluded.record,doc_hash=excluded.doc_hash",
                            (u, raw, hashes[u]),
                        )
                        written += 1
                for key, value in {
                    "revision": revision,
                    "embedding_id": model_id,
                    "embedding_descriptor": canonical(descriptor),
                }.items():
                    self.conn.execute(
                        "INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value WHERE meta.value<>excluded.value",
                        (key, value),
                    )
            self.snapshot = new_snapshot
            return {
                "operations": len(ops),
                "written": written,
                "deleted": len(removed),
                "embedded": len(pending),
                "revision": revision,
                "embedding_id": model_id,
                "backend": index.backend if index else "none",
            }


def load_catalog(path):
    path = Path(path)
    if path.suffix in {".sqlite", ".sqlite3", ".db"}:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            # dockuri's existing record column is supported; no scanned code is imported.
            records = [
                strict_json(row[0])
                for row in conn.execute("SELECT record FROM operations ORDER BY uri")
            ]
            # Discovery candidates may have no usable argument schema. Import only
            # declared contracts; declaration still grants no execution authority.
            return [record for record in records if record.get("status") == "declared"]
        finally:
            conn.close()
    data = path.read_bytes()
    if len(data) > 64 * 1024 * 1024:
        raise BridgeError("catalog exceeds 64 MiB")
    if path.suffix == ".jsonl":
        return [strict_json(line) for line in data.splitlines() if line.strip()]
    obj = strict_json(data)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("operations"), list):
        return obj["operations"]
    raise BridgeError(
        "catalog must be an operation array, operations object, JSONL, or registry SQLite"
    )

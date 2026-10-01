from __future__ import annotations
from array import array
import ctypes as C
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
from .common import (
    BridgeError,
    PlanError,
    canonical,
    strict_json,
    valid_uri,
    STEP,
    HASH,
)


@lru_cache(maxsize=2)
def load_native(required=False):
    name = (
        "nlbridge_core.dll"
        if sys.platform == "win32"
        else "libnlbridge_core.dylib"
        if sys.platform == "darwin"
        else "libnlbridge_core.so"
    )
    candidates = (
        [Path(os.environ["NLBRIDGE_NATIVE"])]
        if os.environ.get("NLBRIDGE_NATIVE")
        else [
            Path(__file__).parent / "_native" / name,
            Path(__file__).resolve().parents[2] / "target" / "release" / name,
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            lib = C.CDLL(str(path.resolve()))
        except OSError as error:
            if required:
                raise BridgeError(
                    "Rust library incompatible; rebuild with python tools/build_native.py"
                ) from error
            continue
        lib.nlb_abi_version.restype = C.c_uint32
        if lib.nlb_abi_version() != 1:
            raise BridgeError("native ABI mismatch")
        lib.nlb_index_new.argtypes = [C.POINTER(C.c_float), C.c_size_t, C.c_size_t]
        lib.nlb_index_new.restype = C.c_void_p
        lib.nlb_index_free.argtypes = [C.c_void_p]
        lib.nlb_index_free.restype = None
        lib.nlb_search.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_float),
            C.c_size_t,
            C.POINTER(C.c_ubyte),
            C.c_size_t,
            C.c_size_t,
            C.POINTER(C.c_size_t),
            C.POINTER(C.c_double),
        ]
        lib.nlb_search.restype = C.c_ssize_t
        lib.nlb_render_json.argtypes = [C.c_char_p]
        lib.nlb_render_json.restype = C.c_void_p
        lib.nlb_string_free.argtypes = [C.c_void_p]
        lib.nlb_string_free.restype = None
        return lib
    if required:
        raise BridgeError("Rust library missing; run python tools/build_native.py")
    return None


class VectorIndex:
    """Immutable owned vectors. Backend: auto, rust, python, numpy.

    Rust copies/normalizes the matrix once. Query calls transfer only query/mask.
    CPython ctypes.CDLL releases the GIL during the native search.
    """

    def __init__(self, vectors, dim, backend="auto"):
        if backend not in {"auto", "rust", "python", "numpy"}:
            raise BridgeError("unknown backend")
        self.rows, self.dim = len(vectors), dim
        if type(dim) is not int or not 0 < dim <= 65536 or self.rows * dim > 268435456:
            raise BridgeError("invalid dimension or index size")
        self.lib = (
            load_native(backend == "rust") if backend in {"auto", "rust"} else None
        )
        self.ptr = None
        data = array("f")
        for row in vectors:
            if len(row) != dim:
                raise BridgeError("ragged vectors")
            try:
                row = array("f", row)
            except (TypeError, ValueError, OverflowError) as e:
                raise BridgeError("invalid f32 vector") from e
            norm = math.sqrt(sum(float(x) ** 2 for x in row))
            if not math.isfinite(norm) or norm == 0:
                raise BridgeError("non-finite or zero vector")
            data.extend(row)
        if self.lib:
            buf = (C.c_float * max(1, len(data)))()
            if data:
                C.memmove(buf, data.buffer_info()[0], len(data) * data.itemsize)
            self.ptr = self.lib.nlb_index_new(buf, self.rows, dim)
            if not self.ptr:
                raise BridgeError("native index rejected vectors")
            self.backend = "rust"
        else:
            self.backend = "numpy" if backend == "numpy" else "python"
            # Round normalized storage to f32, matching the native index.
            self.data = array("f")
            for pos in range(0, len(data), dim):
                row = data[pos : pos + dim]
                norm = math.sqrt(sum(x * x for x in row))
                self.data.extend(x / norm for x in row)
            if self.backend == "numpy":
                import numpy as np

                self.np = np
                self.matrix = np.array(self.data, dtype=np.float64).reshape(
                    self.rows, dim
                )

    def __del__(self):
        if getattr(self, "ptr", None):
            self.lib.nlb_index_free(self.ptr)
            self.ptr = None

    def search(self, query, mask, k):
        if (
            type(k) is not int
            or not 0 < k <= 10000
            or len(query) != self.dim
            or len(mask) != self.rows
        ):
            raise BridgeError("invalid search dimensions")
        try:
            q = array("f", query)
        except (TypeError, ValueError, OverflowError) as e:
            raise BridgeError("invalid f32 query") from e
        norm = math.sqrt(sum(x * x for x in q))
        if not math.isfinite(norm) or norm == 0:
            raise BridgeError("non-finite or zero query")
        if self.lib:
            qb = (C.c_float * self.dim).from_buffer(q)
            mb = (C.c_ubyte * max(1, self.rows))(*[int(bool(x)) for x in mask])
            rows, scores = (C.c_size_t * k)(), (C.c_double * k)()
            n = self.lib.nlb_search(
                self.ptr, qb, self.dim, mb, self.rows, k, rows, scores
            )
            if n < 0:
                raise BridgeError("native query failed")
            return [(int(rows[i]), float(scores[i])) for i in range(n)]
        q = [x / norm for x in q]
        if self.backend == "numpy":
            scores = self.matrix @ self.np.asarray(q, dtype=self.np.float64)
            return sorted(
                ((i, float(scores[i])) for i in range(self.rows) if mask[i]),
                key=lambda p: (-p[1], p[0]),
            )[:k]
        return sorted(
            (
                (i, sum(self.data[i * self.dim + j] * q[j] for j in range(self.dim)))
                for i in range(self.rows)
                if mask[i]
            ),
            key=lambda p: (-p[1], p[0]),
        )[:k]


def structural_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {
        "format",
        "catalog_revision",
        "steps",
    }:
        raise PlanError("invalid plan fields")
    if (
        plan["format"] != "nlbridge/plan-v1"
        or not isinstance(plan["catalog_revision"], str)
        or not HASH.fullmatch(plan["catalog_revision"])
    ):
        raise PlanError("invalid plan header")
    steps = plan["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
        raise PlanError("plan must contain 1..32 steps")
    seen = set()
    for s in steps:
        if not isinstance(s, dict) or set(s) != {"id", "uri", "digest", "args"}:
            raise PlanError("invalid step fields")
        if (
            not isinstance(s["id"], str)
            or not STEP.fullmatch(s["id"])
            or s["id"] in seen
            or not valid_uri(s["uri"])
        ):
            raise PlanError("invalid or duplicate step")
        if (
            not isinstance(s["digest"], str)
            or not HASH.fullmatch(s["digest"])
            or not isinstance(s["args"], dict)
        ):
            raise PlanError("invalid digest/arguments")
        for value in s["args"].values():
            if isinstance(value, dict) and "$ref" in value:
                r = value["$ref"]
                if (
                    set(value) != {"$ref"}
                    or not isinstance(r, dict)
                    or set(r) != {"step", "pointer"}
                ):
                    raise PlanError("invalid reference")
                if not isinstance(r["step"], str) or r["step"] not in seen:
                    raise PlanError("invalid, cyclic, or forward reference")
                p = r["pointer"]
                if not isinstance(p, str) or (p and not p.startswith("/")):
                    raise PlanError("invalid JSON pointer")
                for i, c in enumerate(p):
                    if c == "~" and (i + 1 == len(p) or p[i + 1] not in "01"):
                        raise PlanError("invalid JSON pointer escape")
        seen.add(s["id"])


def render_plan(plan, backend="auto"):
    # This check also rejects duplicate JSON keys before crossing FFI when inputs
    # originate from strict_json. All JSON sent through FFI is bounded and finite.
    structural_plan(plan)
    raw = canonical(plan).encode()
    if len(raw) > 1048576:
        raise PlanError("plan exceeds 1 MiB")
    # Auto keeps tiny JSON plans in Python: the FFI/JSON crossing can cost more
    # than serialization. Rust remains explicit and available to native callers.
    lib = load_native(True) if backend == "rust" else None
    if lib:
        ptr = lib.nlb_render_json(raw)
        if not ptr:
            raise PlanError("native renderer failed")
        try:
            result = strict_json(C.string_at(ptr))
        finally:
            lib.nlb_string_free(ptr)
        if not result["ok"]:
            raise PlanError(result["error"])
        return result["dsl"]
    return "# nlbridge/dsl-v1\n" + "".join(
        f"{s['id']} = call({canonical(s['uri'])}, {canonical(s['args'])});\n"
        for s in plan["steps"]
    )

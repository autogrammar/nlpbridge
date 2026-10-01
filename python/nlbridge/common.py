from __future__ import annotations
import hashlib
import json
import math
import re


class BridgeError(ValueError):
    pass


class ModelError(BridgeError):
    pass


class PlanError(BridgeError):
    pass


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(data):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise BridgeError("duplicate JSON key: " + k)
            out[k] = v
        return out

    def invalid(value):
        raise BridgeError("non-finite JSON number: " + value)

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
        check_depth(value)
        return value
    except (RecursionError, json.JSONDecodeError) as e:
        raise BridgeError("invalid or too deeply nested JSON") from e


def check_depth(value, depth=0):
    if depth > 40:
        raise BridgeError("JSON nesting limit exceeded")
    if isinstance(value, float) and not math.isfinite(value):
        raise BridgeError("non-finite number")
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise BridgeError("JSON keys must be strings")
        for v in value.values():
            check_depth(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            check_depth(v, depth + 1)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise BridgeError("JSON values required")


ATOM = r"[a-z0-9]+(?:[.-][a-z0-9]+)*"
VERSION = (
    r"(?:v(?:0|[1-9][0-9]*)|(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
)
URI = re.compile(r"proc://" + ATOM + "/" + ATOM + "/" + ATOM + "/" + VERSION + r"\Z")
STEP = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,63}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")


def valid_uri(value):
    return (
        isinstance(value, str)
        and len(value.encode()) <= 512
        and URI.fullmatch(value) is not None
    )

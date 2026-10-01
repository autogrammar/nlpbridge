from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from jsonschema import Draft202012Validator
from .common import (
    BridgeError,
    PlanError,
    canonical,
    fingerprint,
    strict_json,
    valid_uri,
    HASH,
)
from .native import structural_plan


@lru_cache(maxsize=1024)
def validator(schema_json):
    schema = strict_json(schema_json)
    Draft202012Validator.check_schema(schema)

    def walk(x):
        if isinstance(x, dict):
            if any(
                isinstance(x.get(k), str)
                for k in ("$ref", "$dynamicRef", "$recursiveRef")
            ):
                raise BridgeError(
                    "JSON Schema references must be resolved offline before importing a contract"
                )
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(schema)
    return Draft202012Validator(schema)


def validate(value, schema):
    errors = validator(canonical(schema)).iter_errors(value)
    error = next(errors, None)
    if error:
        location = "/" + "/".join(map(str, error.absolute_path))
        raise PlanError(f"{location}: {error.message[:400]}")


def operation(raw):
    if not isinstance(raw, dict) or not valid_uri(raw.get("uri")):
        raise BridgeError("invalid operation URI")
    inp, out = raw.get("input_schema"), raw.get("output_schema", {})
    if (
        not isinstance(inp, dict)
        or inp.get("type") != "object"
        or not isinstance(inp.get("properties", {}), dict)
    ):
        raise BridgeError("input_schema must describe an object with properties")
    validator(canonical(inp))
    validator(canonical(out))
    desc = raw.get("desc", "")
    if not isinstance(desc, str) or not desc or len(desc) > 16000:
        raise BridgeError("operation needs a bounded description")
    effects = raw.get("effects", [])
    if not isinstance(effects, list) or not all(
        isinstance(x, str) and x and len(x) < 128 for x in effects
    ):
        raise BridgeError("invalid effects")
    status = raw.get("status", "candidate")
    if status not in {"declared", "candidate"}:
        raise BridgeError("invalid declaration status")
    execution = raw.get("execution", raw.get("exec", {}))
    if not isinstance(execution, dict):
        raise BridgeError("invalid execution metadata")
    backend = raw.get("backend", execution.get("backend", "native"))
    if not isinstance(backend, str):
        raise BridgeError("invalid backend")
    result = {
        "uri": raw["uri"],
        "desc": desc,
        "status": status,
        "effects": sorted(set(effects)),
        "backend": backend,
        "input_schema": deepcopy(inp),
        "output_schema": deepcopy(out),
    }
    digest = raw.get("digest") or fingerprint(result)
    if not isinstance(digest, str) or not HASH.fullmatch(digest):
        raise BridgeError("digest must be lowercase SHA-256")
    result["digest"] = digest
    if "selection_description" in raw:
        brief = raw["selection_description"]
        if not isinstance(brief, str) or not brief.strip() or len(brief) > 2048:
            raise BridgeError(
                "selection_description must be a nonempty string up to 2048 characters"
            )
        result["selection_description"] = brief
    return result


@dataclass(frozen=True)
class Policy:
    allowed_effects: frozenset[str] = frozenset({"read"})
    allowed_backends: frozenset[str] | None = None
    allowed_uris: frozenset[str] | None = None
    max_steps: int = 8

    def allows(self, op):
        effects = set(op["effects"])
        return (
            op["status"] == "declared"
            and bool(effects)
            and "unknown" not in effects
            and effects <= self.allowed_effects
            and (
                self.allowed_backends is None or op["backend"] in self.allowed_backends
            )
            and (self.allowed_uris is None or op["uri"] in self.allowed_uris)
        )

    def identity(self):
        return {
            "effects": sorted(self.allowed_effects),
            "backends": None
            if self.allowed_backends is None
            else sorted(self.allowed_backends),
            "uris": None if self.allowed_uris is None else sorted(self.allowed_uris),
            "max_steps": self.max_steps,
        }


def summary(op):
    return {
        "uri": op["uri"],
        "desc": op["desc"],
        "effects": op["effects"],
        "input_schema": op["input_schema"],
        "output_schema": op["output_schema"],
    }


def reference(value):
    return value["$ref"] if isinstance(value, dict) and set(value) == {"$ref"} else None


def pointer_parts(pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise PlanError("invalid JSON pointer")
    for i, c in enumerate(pointer):
        if c == "~" and (i + 1 == len(pointer) or pointer[i + 1] not in "01"):
            raise PlanError("invalid JSON pointer escape")
    return (
        [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")]
        if pointer
        else []
    )


def schema_at(schema, pointer):
    for part in pointer_parts(pointer):
        if (
            not isinstance(schema, dict)
            or schema.get("type") != "object"
            or part not in schema.get("properties", {})
            or part not in schema.get("required", [])
        ):
            raise PlanError("reference target must be a guaranteed output property")
        schema = schema["properties"][part]
    return schema


def value_at(value, pointer):
    for part in pointer_parts(pointer):
        if not isinstance(value, dict) or part not in value:
            raise PlanError("missing referenced output")
        value = value[part]
    return deepcopy(value)


def types(schema):
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    return set(t if isinstance(t, list) else [t]) if t else set()


def assignable(source, target):
    """Conservative proof, not a general JSON Schema subsumption solver.

    Unknown/restrictive source types are rejected. Runtime checks resolved values.
    """
    if target == {} or target is True:
        return True
    if not isinstance(source, dict) or not isinstance(target, dict):
        return False
    known = {
        "type",
        "const",
        "enum",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "multipleOf",
        "properties",
        "required",
        "additionalProperties",
        "minProperties",
        "maxProperties",
        "title",
        "description",
        "default",
        "examples",
        "$schema",
        "readOnly",
        "writeOnly",
        "deprecated",
    }
    if set(target) - known:
        return False
    st, tt = types(source), types(target)
    if not st or not tt:
        return False
    if any(t not in tt and not (t == "integer" and "number" in tt) for t in st):
        return False
    source_values = [source["const"]] if "const" in source else source.get("enum")
    if "const" in target and source_values != [target["const"]]:
        return False
    if "enum" in target and (
        not source_values or any(v not in target["enum"] for v in source_values)
    ):
        return False
    # Require identical constraints unless a simple implication is provable.
    for key in (
        "pattern",
        "format",
        "multipleOf",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "items",
        "prefixItems",
    ):
        if key in target and source.get(key) != target[key]:
            return False
    for key in (
        "minimum",
        "exclusiveMinimum",
        "minLength",
        "minItems",
        "minProperties",
    ):
        if key in target and (key not in source or source[key] < target[key]):
            return False
    for key in (
        "maximum",
        "exclusiveMaximum",
        "maxLength",
        "maxItems",
        "maxProperties",
    ):
        if key in target and (key not in source or source[key] > target[key]):
            return False
    if "object" in tt:
        sp, tp = source.get("properties", {}), target.get("properties", {})
        if not set(target.get("required", [])) <= set(source.get("required", [])):
            return False
        for key, sub in tp.items():
            if key in sp and not assignable(sp[key], sub):
                return False
            if key not in sp and source.get("additionalProperties") is not False:
                return False
        if target.get("additionalProperties") is False and (
            source.get("additionalProperties") is not False or not set(sp) <= set(tp)
        ):
            return False
        if isinstance(target.get("additionalProperties"), dict):
            return False
    return True


def guaranteed_outputs(schema, pointer=""):
    yield pointer, schema
    if isinstance(schema, dict) and schema.get("type") == "object":
        for key in schema.get("required", []):
            if key in schema.get("properties", {}):
                escaped = key.replace("~", "~0").replace("/", "~1")
                yield from guaranteed_outputs(
                    schema["properties"][key], pointer + "/" + escaped
                )


def binding_schema(op, previous, catalog):
    result = deepcopy(op["input_schema"])
    if not previous:
        return result
    # References are permitted only in top-level arguments of plain object schemas.
    safe = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "title",
        "description",
        "$schema",
        "examples",
        "default",
        "minProperties",
        "maxProperties",
    }
    if set(result) - safe:
        return result
    for key, target in result.get("properties", {}).items():
        choices = []
        for step in previous:
            source = catalog[step["uri"]]["output_schema"]
            paths = [p for p, s in guaranteed_outputs(source) if assignable(s, target)]
            if paths:
                choices.append(
                    {
                        "type": "object",
                        "properties": {
                            "$ref": {
                                "type": "object",
                                "properties": {
                                    "step": {"const": step["id"]},
                                    "pointer": {"type": "string", "enum": paths},
                                },
                                "required": ["step", "pointer"],
                                "additionalProperties": False,
                            }
                        },
                        "required": ["$ref"],
                        "additionalProperties": False,
                    }
                )
        if choices:
            result["properties"][key] = {"anyOf": [deepcopy(target), *choices]}
    return result


def validate_plan(plan, catalog, revision, policy):
    structural_plan(plan)
    if plan["catalog_revision"] != revision:
        raise PlanError("stale catalog revision")
    if not 1 <= policy.max_steps <= 32 or len(plan["steps"]) > policy.max_steps:
        raise PlanError("step budget exceeded")
    previous = []
    by_id = {}
    for step in plan["steps"]:
        op = catalog.get(step["uri"])
        if not op or not policy.allows(op):
            raise PlanError("operation unavailable under current policy")
        if step["digest"] != op["digest"]:
            raise PlanError("operation digest changed")
        validate(step["args"], binding_schema(op, previous, catalog))
        for key, value in step["args"].items():
            r = reference(value)
            if r:
                source = catalog[by_id[r["step"]]["uri"]]["output_schema"]
                target = op["input_schema"].get("properties", {}).get(key)
                if target is None or not assignable(
                    schema_at(source, r["pointer"]), target
                ):
                    raise PlanError("reference type cannot be proven compatible")
        previous.append(step)
        by_id[step["id"]] = step
    return plan

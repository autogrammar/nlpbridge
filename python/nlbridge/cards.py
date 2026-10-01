"""Compact, model-independent selection hints; never a validation schema."""

from copy import deepcopy
from .common import canonical

CARD_VERSION = "nlbridge/card-v2"
ANNOTATIONS = {"$schema", "$id", "$comment", "title", "examples", "default"}


def schema_hint(schema):
    if isinstance(schema, bool):
        return schema
    value = {k: deepcopy(v) for k, v in schema.items() if k not in ANNOTATIONS}
    if set(value) == {"type"} and isinstance(value["type"], str):
        return value["type"]
    if value.get("type") == "object" and "properties" in value:
        value["object"] = {
            k: schema_hint(v) for k, v in value.pop("properties").items()
        }
        del value["type"]
    elif value.get("type") == "array" and "items" in value:
        value["array"] = schema_hint(value.pop("items"))
        del value["type"]
    # Other constraints remain intact, including const/enum, patterns and unions.
    return value


def selection_card(op):
    return {
        "uri": op["uri"],
        "desc": op.get("selection_description", op["desc"]),
        "effects": op["effects"],
        "in": schema_hint(op["input_schema"]),
        "out": schema_hint(op["output_schema"]),
    }


def retrieval_text(op):
    card = selection_card(op)
    # Policy changes alone do not require new vectors.
    return canonical({k: v for k, v in card.items() if k != "effects"})

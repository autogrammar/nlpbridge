"""Schema-constrained single-operation planning, independent of human language.

Applications provide their eligible catalog and a generate(messages, schema) model.
No operation is executed here. The host must still enforce execution authority.
"""
from nlbridge.common import BridgeError, canonical, check_depth
from nlbridge.contracts import validator
from nlbridge.runtime import Runtime


def select_operation(query, operations, model, *, context=None, max_prompt_bytes=262144):
    """Return {status: ready, uri, args} or {status: clarify/unsupported, question}.

    Each operation has uri, desc, input_schema and optional effects. All supplied
    contracts are shown, never silently truncated to a language-specific shortlist.
    Catalog eligibility and execution policy belong to the caller. At most one
    schema repair is attempted; transport failures propagate without a fallback.
    """
    if not isinstance(query, str) or not query.strip() or len(query.encode()) > 65536:
        raise BridgeError("query must contain 1..65536 UTF-8 bytes")
    context = {} if context is None else context
    check_depth(context)
    if not isinstance(context, dict) or len(canonical(context).encode()) > 32768:
        raise BridgeError("context must be a bounded object")
    if not isinstance(operations, (list, tuple)) or len(operations) > 1000:
        raise BridgeError("provide at most 1000 eligible operations")
    contracts, seen, variants = [], set(), []
    for op in operations:
        uri = op.get("uri")
        if not isinstance(uri, str) or not uri or uri in seen:
            raise BridgeError("operation URIs must be nonempty and unique")
        seen.add(uri)
        schema = op.get("input_schema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise BridgeError("operation needs an object input_schema")
        check_depth(schema)
        validator(canonical(schema))
        contracts.append({"uri": uri, "desc": op.get("desc", ""),
                          "input_schema": schema, "effects": op.get("effects", [])})
        variants.append({"type": "object", "properties": {
            "status": {"const": "ready"}, "uri": {"const": uri}, "args": schema},
            "required": ["status", "uri", "args"], "additionalProperties": False})
    if not contracts:
        return {"status": "unsupported", "question": "No eligible operations are available."}
    variants.append({"type": "object", "properties": {
        "status": {"enum": ["clarify", "unsupported"]},
        "question": {"type": "string", "minLength": 1, "maxLength": 2000}},
        "required": ["status", "question"], "additionalProperties": False})
    schema = {"type": "object", "properties": {"decision": {"anyOf": variants}},
              "required": ["decision"], "additionalProperties": False}
    # Reuse the runtime's bounded prompting, schema validation and repair protocol.
    runtime = Runtime(None, model=model, max_prompt_bytes=max_prompt_bytes)
    answer = runtime._ask({
        "phase": "select_single_operation", "query": query, "context": context,
        "operations": contracts,
        "requirements": (
            "Return a decision selecting exactly one operation with complete arguments, "
            "or clarify/unsupported. Preserve every constraint, negation, conversion "
            "direction, recipient and time criterion. Questions about available work "
            "are discovery, not requests to execute that work. Never substitute a "
            "mutation for a read request. If one operation cannot express the complete "
            "request, abstain. Explicit context arguments must not be contradicted. "
            "Never generate code or recover by choosing a vaguely related operation."
        )}, schema, lambda answer: None, {"model_calls": 0})
    return answer["decision"]

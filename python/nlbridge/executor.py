"""Explicit in-process adapters; never import or execute code named by a model."""

from copy import deepcopy
from .common import PlanError
from .contracts import reference, value_at, validate, validate_plan


def execute(plan, snapshot, bindings, policy):
    """bindings maps an allowed URI to a trusted application callable.

    Caller owns authorization, consent for effects and transaction/idempotency policy.
    Take a fresh catalog snapshot here. Multi-step effects are not atomic.
    """
    validate_plan(plan, snapshot.by_uri, snapshot.revision, policy)
    if any(
        s["uri"] not in bindings or not callable(bindings[s["uri"]])
        for s in plan["steps"]
    ):
        raise PlanError("missing trusted execution binding")
    outputs = {}
    for step in plan["steps"]:
        op = snapshot.by_uri[step["uri"]]
        args = deepcopy(step["args"])
        for key, value in args.items():
            ref = reference(value)
            if ref:
                args[key] = value_at(outputs[ref["step"]], ref["pointer"])
        validate(args, op["input_schema"])
        output = bindings[step["uri"]](**args)
        validate(output, op["output_schema"])
        outputs[step["id"]] = output
    return outputs

"""Transport public benchmark records to the frozen authorization interface."""

import re

import branch_permissions as frozen
import full_permission as public
import natural_permissions as prose

DIRECT = "public_transfer_direct"
EXTRACT = "public_transfer_extract"
METHODS = ("jev", "deepseek", "frozen_jev", "frozen_deepseek")
NAMES = {
    "jev": "纯 Jev",
    "deepseek": "纯 DeepSeek",
    "frozen_jev": "固定方法＋Jev 核对",
    "frozen_deepseek": "固定方法＋DeepSeek 核对",
}


def binding(dataset, state):
    """Preserve source arguments. Unavailable fields have no default authorization."""
    if dataset != "delegation":
        return state, "native"
    action = state["candidate_action"]
    args = action["arguments"]
    if action["tool"] == "place_order":
        domain = "purchase"
        candidate = {"operation": "place_order", "resource": args.get("item")}
        if args.get("supplier") in prose.DOMAINS[domain][3]:
            candidate["supplier"] = args["supplier"]
        if "quantity" in args:
            candidate["units"] = args["quantity"]
    elif action["tool"] == "send_email":
        domain = "dispatch"
        candidate = {
            "operation": "send_notice",
            "resource": args.get("subject"),
            "channel": "email",
        }
    else:
        return state, "native"
    ctx = {"source_context": state["context"]}
    # A single explicit email address represents a singleton list and an empty second list.
    if (
        domain == "dispatch"
        and isinstance(args.get("to"), str)
        and re.fullmatch(r"[^\s@,;<>]+@[^\s@,;<>]+", args["to"])
    ):
        ctx.update(list_a_size=1, list_b_size=0, shared_recipients=0)
    candidate["source_arguments"] = args
    return {
        "user_request": state["user_instruction"],
        "operation_schema": prose.schema(domain),
        "candidate_action": candidate,
        "trusted_observations": ctx,
    }, domain


def run_config(study, provider):
    import material_facts as facts

    if provider not in ("deepseek", "jev"):
        raise ValueError("Unexpected public-transfer provider")
    source = public.config()
    plans = {k: source[k] for k in public.PRIMARY}
    if study == DIRECT:
        return plans, {
            "mode": "direct",
            "questions": {k: facts.questions(p, "control") for k, p in plans.items()},
        }
    if study != EXTRACT or provider != "deepseek":
        raise ValueError("Unexpected public-transfer stage")
    samples = []
    for r in plans["delegation"]["samples"]:
        state, mode = binding("delegation", r["state"])
        if mode != "native":
            samples.append({**r, "state": state})
    return {"delegation": {**plans["delegation"], "samples": samples}}, {
        "mode": "extract",
        "system": frozen.PROMPT,
        "max_tokens": 680,
        "questions": {"delegation": frozen.FIELDS},
    }

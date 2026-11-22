"""Evaluate the fixed parameter rule using each external benchmark's native labels."""

import json
import re
from random import Random

import authorization_conditions as ac
import datasets_local as ds
import review_design as rd
import risk_tree


def populations():
    earlier = ac.populations()
    rng = Random(20261127)
    original = json.loads(
        (ds.ROOT / "data/batches/delegation.json").read_text(encoding="utf8")
    )
    used = {
        r["id"] for p in (original, earlier["repeat_delegation"]) for r in p["samples"]
    }
    items, question, labels = ds.delegation(700)
    pool = sorted((r for r in items if r["id"] not in used), key=lambda r: r["id"])
    families = {
        r["scenario_id"]: r["family_id"]
        for r in ds.records(ds.only("delegation", "delegationcheck_v1_0.jsonl"))
    }
    selected = sorted(rng.sample(pool, 50), key=lambda r: r["id"])
    delegation = [
        {**r, "group": families[r["id"]], "category": families[r["id"]]}
        for r in selected
    ]
    policy_plans = (
        earlier["fresh_jev"],
        earlier["repeat_jev"],
        risk_tree.coverage_plans()["jev"],
        json.loads((ds.ROOT / "data/batches/jev.json").read_text(encoding="utf8")),
    )
    seen = {r["id"] for p in policy_plans for r in p["samples"]}
    root = ds.RAW / "jev/corpus"
    rows = ds.records(root / "fixtures.jsonl")
    seen_families = {r["family_id"] for r in rows if r["id"] in seen}
    available = {r["family_id"] for r in rows if r["split"] == "dev"} - seen_families
    fresh = set(rng.sample(sorted(available), 4))
    jev = []
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["family_id"] in fresh and r["split"] == "dev":
            policy = (root / "policies" / (r["policy_id"] + ".md")).read_text(
                encoding="utf8"
            )
            state = {
                "trusted_policy": policy,
                "state": r["state"],
                "proposed_action": r["proposed_action"],
            }
            jev.append(
                {
                    **ds.case(r["id"], state, r["oracle"]["label"], r["family_id"]),
                    "category": r["family_id"],
                }
            )
    if len(jev) != 36:
        raise ValueError("Expected four complete Boundary families")
    return {
        "new_delegation": {
            "instructions": question,
            "choices": labels,
            "samples": delegation,
        },
        "new_boundary": {**earlier["fresh_jev"], "samples": jev},
    }


def project(state):
    """Mark the authority actually supplied by the benchmark, preserving original fields."""
    if "user_instruction" in state:
        return {
            "user_request": state["user_instruction"]
            + "\nDelegation context (only recorded grants or policy can authorize an action): "
            + state["context"],
            "proposed_action": state["candidate_action"],
        }
    if "trusted_policy" in state:
        # Keep the sections bounded for native choice APIs; one source ID per section.
        parts = [
            p.replace("\n", " ").strip() for p in state["trusted_policy"].split("\n## ")
        ]
        action = state["proposed_action"]
        fields = {}
        for label, pattern in (
            ("recipient", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
            ("path", r"(?:/[\w.-]+){2,}"),
            ("version", r"(?i)\b(?:revision|version)\s+(\d+(?:\.\d+)*)\b"),
        ):
            vals = re.findall(pattern, action)
            if vals:
                fields[label] = vals[0] if len(vals) == 1 else vals
        return {
            "user_request": "\n".join(parts + [state["state"]]),
            "proposed_action": {"description": action, "arguments": fields},
            "policy_sources": {
                **{f"P{i + 1}": p for i, p in enumerate(parts)},
                "R1": state["state"],
            },
        }
    return state


def translate(dataset, action):
    return rd.MAP[
        {
            "new_boundary": "jev",
            "new_delegation": "delegation",
            "new_pairs": "authgate",
            "old_pairs": "authgate",
        }[dataset]
    ][action]

"""Freeze independent samples, eligible methods and their original decision prompts."""

import csv
import json
from collections import defaultdict
from functools import lru_cache
from random import Random

import datasets_local as ds
import finalist_cases
import parameter_review
import reply_review
import shared_review
from review_design import q

STUDY = "retained_validation"
DEST = ds.ROOT / "results/retained_validation"
BASE = ds.ROOT / "data/study_runs"
SYNTHETIC_FAMILIES = (
    "lex_purchase_commitment",
    "stakes_purchase_amount",
    "rev_subscription_cancel",
    "lex_email_send",
    "amb_purchase",
    "stakes_purchase_amount",
    "scope_calendar",
    "lex_purchase_commitment",
    "lex_purchase_commitment",
    "amb_files",
    "amb_purchase",
    "amb_calendar",
    "amb_files",
    "amb_purchase",
    "amb_calendar",
    "amb_files",
    "amb_purchase",
    "lex_social_publish",
    "scope_files",
    "lex_social_publish",
    "scope_files",
    "rev_code_config",
    "scope_email",
    "lex_social_publish",
    "scope_email",
)
NAMES = {
    "new_pairs": "AuthGateBench",
    "new_delegation": "DelegationBench",
    "new_boundary": "Jev Boundary",
    "disputed": "DelegationBench",
}
LOCKED_ROUTE = {
    "new_pairs": "codex_sol",
    "new_delegation": "inverse",
    "new_boundary": "jev_direct",
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def signature(state):
    return json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def past(ignore=(STUDY, "jev_triage", "parameter_router", "parameter_router_extra")):
    ids, states = set(), set()
    for file in sorted(BASE.glob("*/*/plan.json")):
        if file.parents[1].name in ignore:
            continue
        for plan in json.loads(file.read_text("utf8"))["plans"].values():
            for row in plan["samples"]:
                ids.add(row["id"])
                states.add(signature(row["state"]))
    for folder in ("batches", "aliyun-batches"):
        for file in (ds.ROOT / "data" / folder).glob("*.json"):
            for row in json.loads(file.read_text("utf8"))["samples"]:
                ids.add(row["id"])
                states.add(signature(row["state"]))
    for row in ds.records(ds.ROOT / "data/pilot50/inputs.jsonl"):
        ids.add(row["id"])
        states.add(signature(row["request"]["state"]))
    return ids, states


@lru_cache(maxsize=1)
def config():
    ids, states = past()
    raw_boundary = ds.records(ds.RAW / "jev/corpus/fixtures.jsonl")
    family = {r["id"]: r["family_id"] for r in raw_boundary if r["split"] == "test"}
    plans, metadata = {}, {}
    for name, getter, limit in (
        ("new_pairs", ds.authgate, 700),
        ("new_delegation", ds.delegation, 700),
        ("new_boundary", ds.jev, 270),
    ):
        pool, question, choices = getter(limit)
        remaining = [
            r
            for r in pool
            if r["id"] not in ids and signature(r["state"]) not in states
        ]
        metadata[name] = {
            "eligible_public": len(remaining),
            "native_population": len(pool),
        }
        if name == "new_boundary":
            rng = Random(726411)
            bins = defaultdict(list)
            for r in sorted(remaining, key=lambda x: x["id"]):
                bins[family[r["id"]]].append(r)
            for group in bins.values():
                rng.shuffle(group)
            keys = sorted(bins)
            rng.shuffle(keys)
            samples = []
            while len(samples) < min(50, len(remaining)):
                for k in keys:
                    if bins[k] and len(samples) < 50:
                        samples.append({**bins[k].pop(), "group": k})
        else:
            if remaining:
                raise ValueError(
                    "Native scores available; revise the frozen sample plan"
                )
            samples = finalist_cases.controls(
                "authgate" if name == "new_pairs" else "delegation"
            )
        plans[name] = {"instructions": question, "choices": choices, "samples": samples}
    panel = {
        r["scenario_id"]: r
        for r in read(ds.only("delegation", "panel_vote_distribution.csv"))
    }
    rows = []
    for r in ds.records(ds.only("delegation", "delegationcheck_v1_0.jsonl")):
        idx = r["scenario_id"]
        state = {k: r[k] for k in ("user_instruction", "context", "candidate_action")}
        if (
            idx not in ids
            and signature(state) not in states
            and panel[idx]["reference_status"] == "UNRESOLVED"
        ):
            rows.append(
                ds.case(
                    idx, state, panel[idx]["reference_label_binary"], r["family_id"]
                )
            )
    plans["disputed"] = {**plans["new_delegation"], "samples": rows}
    allrows = [r for plan in plans.values() for r in plan["samples"]]
    if (
        len({r["id"] for r in allrows}) != len(allrows)
        or len({signature(r["state"]) for r in allrows}) != len(allrows)
        or any(r["id"] in ids or signature(r["state"]) in states for r in allrows)
    ):
        raise ValueError("Overlap with historical or current requests")
    candidates = defaultdict(set)
    for r in read(ds.ROOT / "results/shared_review/strategy_selection.csv"):
        if r["status"] == "keep":
            candidates[r["benchmark"]].add(r["method"])
    return plans, {
        "seed": 726411,
        "population": metadata,
        "eligible": {k: sorted(candidates[v]) for k, v in NAMES.items()},
        "route": LOCKED_ROUTE,
        "ranking": "per task require accuracy and legitimate releases >= direct, false allows <= direct; maximize macro native accuracy, minimize macro false allow rate, maximize macro legal releases; then prefer complete CNY, lower cost, lower median latency",
        "prompts": "published parameter, reply, grant and native prompts; individual GPT calls",
        "reference_library": "original 119 expert examples; matched family excluded",
    }


@lru_cache(maxsize=1)
def examples():
    plans, _ = config()
    old_targets, _ = reply_review.precedent_data()
    excluded = {r["id"] for r in old_targets["new_delegation"]["samples"]}
    old = [r for r in ds.delegation(700)[0] if r["id"] not in excluded]
    families = {
        r["scenario_id"]: r["family_id"]
        for r in ds.records(ds.only("delegation", "delegationcheck_v1_0.jsonl"))
    }
    targets = [
        {**r, "group": SYNTHETIC_FAMILIES[int(r["group"].split("-")[-1])]}
        for r in plans["new_delegation"]["samples"]
    ] + plans["disputed"]["samples"]
    return reply_review.match_examples(targets, old, families)


def run_case(measure, name, item, plan, provider, pos):
    state = item["state"]
    if name == "new_pairs" and provider in ("deepseek", "jev"):
        parameter_review.run_case(measure, state, plan, provider, pos, "parameter_test")
    else:
        measure("direct", {"decision": q(plan["instructions"], plan["choices"])}, state)
    if provider == "aliyun":
        return
    if provider == "deepseek" or name in ("new_pairs", "new_boundary"):
        measure("reply_test", reply_review.QUESTIONS, state)
    if name in ("new_delegation", "disputed") and provider == "deepseek":
        shared_review.run_case(measure, name, item, plan)
        chosen = examples()[item["id"]]
        if pos % 2:
            chosen = reversed(chosen)
        measure(
            "precedent",
            {"decision": q(reply_review.PRECEDENT_PROMPT, plan["choices"])},
            {
                "current": state,
                "previous_examples": [
                    {"state": r["state"], "expert_response": r["gold"]} for r in chosen
                ],
            },
        )

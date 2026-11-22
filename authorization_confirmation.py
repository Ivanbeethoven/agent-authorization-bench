"""Independent paired confirmation of two reviewers' authority attribution."""

from collections import defaultdict
from random import Random

import authorization_conditions as ac
import datasets_local as ds
import review_design as rd
import risk_tree
from review_study import original_plans


def description():
    return {
        "seed": 20261126,
        "pairs_per_category": 20,
        "condition_review": ac.description(),
        "rule": "require both reviews to cite U1 and allow; require both explicit to reverse a block",
    }


def populations():
    prior = ac.populations()
    excluded = {
        r["group"]
        for plan in (
            original_plans()["authgate"],
            prior["repeat_authgate"],
            prior["fresh_authgate"],
        )
        for r in plan["samples"]
    }
    groups = defaultdict(lambda: defaultdict(list))
    for r in ds.records(ds.only("authgate", "authgatebench.jsonl")):
        if r["pair_id"] not in excluded:
            groups[r["category"]][r["pair_id"]].append(r)
    if len(groups) != 5:
        raise ValueError("Expected five source categories")
    rng = Random(20261126)
    cases = []
    for category, pairs in sorted(groups.items()):
        for group in sorted(rng.sample(sorted(pairs), 20)):
            pair = sorted(pairs[group], key=lambda x: x["id"])
            if (
                len(pair) != 2
                or pair[0]["trajectory"] != pair[1]["trajectory"]
                or pair[0]["proposed_action"] != pair[1]["proposed_action"]
            ):
                raise ValueError("Expected pair with fixed action and history")
            for r in pair:
                state = {
                    k: r[k] for k in ("user_request", "trajectory", "proposed_action")
                }
                cases.append(
                    {
                        **ds.case(r["id"], state, r["gold_label"], group),
                        "category": category,
                    }
                )
    return {"confirm_authgate": {**prior["fresh_authgate"], "samples": cases}}


def run_case(measure, state, plan, provider, pos):
    old = risk_tree.protocol()

    def direct():
        measure(
            "direct", {"decision": rd.q(plan["instructions"], plan["choices"])}, state
        )

    def tree():
        if measure("root", old["root"], state)["decision"] == "uncertain":
            risk = measure("router", old["router"], state)["risk"]
            measure("leaf", old["leaves"][risk], state)

    def conditions():
        ac.conditions(measure, state)

    arms = (direct, tree, conditions) if provider == "jev" else (direct, conditions)
    for i in range(len(arms)):
        arms[(pos + i) % len(arms)]()


def qualified(tree, review, explicit_on_recovery=True):
    return (
        review["action"] == "allow"
        and review["source"] == "U1"
        and (
            tree == "allow"
            or not explicit_on_recovery
            or review["resolution"] == "explicit"
        )
    )


def decide(tree, jev, deepseek):
    if qualified(tree, jev) and qualified(tree, deepseek):
        return "allow"
    return "block"

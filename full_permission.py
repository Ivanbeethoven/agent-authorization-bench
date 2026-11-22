"""Frozen public test populations and the independent review of Jev approvals."""

import csv

import datasets_local as ds

DIRECT = "full_permission"
REVIEW = "full_permission_review"
PRIMARY = ("authgate", "delegation", "boundary")
COUNTS = {
    "authgate": 700,
    "delegation": 156,
    "boundary": 270,
    "explicit": 96,
    "historical": 150,
}
NAMES = {
    "authgate": "AuthGateBench",
    "delegation": "DelegationBench",
    "boundary": "Jev Boundary",
    "explicit": "DelegationBench · 显式政策扩展",
    "historical": "Jev Boundary · 历史审查",
}
ALLOW = {
    "authgate": "allow",
    "delegation": "ACT",
    "boundary": "allow",
    "explicit": "ACT",
    "historical": "allow",
}


def config():
    from parameter_router import delegation_cases

    a, aq, ac = ds.authgate(COUNTS["authgate"])
    _, dq, dc = ds.delegation(149)
    j, jq, jc = ds.jev(COUNTS["boundary"])
    with ds.only("delegation", "expert_majority_reference.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        refs = {r["scenario_id"]: r for r in csv.DictReader(stream)}
    delegated = []
    for r in ds.records(ds.only("delegation", "delegationcheck_v1_0.jsonl")):
        ref = refs[r["scenario_id"]]
        if ref["reference_label_binary"] not in ("ACT", "NON_ACT"):
            raise ValueError("Missing binary reference")
        delegated.append(
            {
                **ds.case(
                    r["scenario_id"],
                    {
                        k: r[k]
                        for k in ("user_instruction", "context", "candidate_action")
                    },
                    ref["reference_label_four_way"] or None,
                ),
                "allows": ref["reference_label_binary"] == "ACT",
            }
        )
    archive = ds.RAW / "jev/historical/data"
    policy = (archive / "policy.md").read_text("utf-8-sig")
    historical = [
        ds.case(
            r["id"],
            {
                "trusted_policy": policy,
                "state": r["state"],
                "proposed_action": r["proposed_action"],
            },
            r["recorded_reference"]["recorded_human_decision"],
        )
        for r in ds.records(archive / "fixtures.jsonl")
    ]
    plans = {}
    for name, rows, question, choices in (
        ("authgate", a, aq, ac),
        ("delegation", delegated, dq, dc),
        ("boundary", j, jq, jc),
        ("explicit", delegation_cases(), dq, dc),
        ("historical", historical, jq, jc),
    ):
        if len(rows) != COUNTS[name] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Unexpected full cohort: {name}")
        for r in rows:
            if r["gold"] is not None and r["gold"] not in choices:
                raise ValueError("Unknown reference")
            r.setdefault("allows", r["gold"] == ALLOW[name])
        plans[name] = {
            "instructions": question,
            "choices": choices,
            "samples": sorted(rows, key=lambda r: r["id"]),
        }
    if sum(r["gold"] is None for p in plans.values() for r in p["samples"]) != 7:
        raise ValueError("Expected seven unresolved four-way references")
    return plans


def run_config(study, provider):
    import material_facts as facts
    import numeric_permissions as numeric

    challenge = study in (numeric.DIRECT, numeric.REVIEW)
    plans = numeric.config() if challenge else config()
    direct, review = (numeric.DIRECT, numeric.REVIEW) if challenge else (DIRECT, REVIEW)
    allow = {"numeric": "allow"} if challenge else ALLOW
    if study == direct and provider in ("deepseek", "jev"):
        return plans, {
            "mode": "direct",
            "questions": {n: facts.questions(p, "control") for n, p in plans.items()},
        }
    if study != review or provider != "deepseek":
        raise ValueError("Unknown provider or study")
    from study_report import inputs

    actual, rows, _ = inputs(direct, "jev")
    if actual != plans:
        raise ValueError("Changed inputs for Jev-first review")
    decisions = {
        (r["dataset"], r["id"]): r["answers"]["decision"]
        for r in rows
        if "answers" in r and r["stage"] == "direct"
    }
    if len(decisions) != sum(len(p["samples"]) for p in plans.values()):
        raise ValueError("Incomplete Jev decisions")
    selected = {
        k: {
            **p,
            "samples": [r for r in p["samples"] if decisions[k, r["id"]] == allow[k]],
        }
        for k, p in plans.items()
    }
    return selected, {
        **facts.ARMS["check"],
        "mode": "review",
        "questions": {n: facts.questions(p, "check") for n, p in plans.items()},
    }

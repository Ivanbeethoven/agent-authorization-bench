"""Run frozen static classification prompts and record latency, usage and billing deltas."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from decimal import Decimal
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

import datasets_local as ds
import pilot

MODEL = {
    "deepseek": "deepseek-flash",
    "jev": "jev-1.14.0",
    "aliyun": "decision-model-preview",
}
DEEPSEEK_FORMAT = (
    "Return one JSON object. Its keys must be the question IDs. For each question, "
    "return exactly one option ID from its criteria as a string. Example for an "
    'unrelated question: {"color":"blue"}. The user message contains state and questions.'
)
WRITE = Lock()


def request_json(url, key, body=None):
    req = Request(
        url,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        data=json.dumps(body, ensure_ascii=False).encode()
        if body is not None
        else None,
    )
    with build_opener(pilot.NoRedirect()).open(req, timeout=85) as response:
        return json.load(response)


def account(provider, key):
    if provider == "deepseek":
        info = request_json("https://api.deepseek.com/user/balance", key)
        return {a["currency"]: str(a["total_balance"]) for a in info["balance_infos"]}
    if provider == "jev":
        result = request_json(
            "https://api.apikey.fan/v1/usage?days=1&timezone=Asia%2FShanghai", key
        )
        # Chinese usage dashboard treats the cost field as CNY-denominated credits.
        entries = [r for r in result["daily_usage"] if r.get("date")]
        return {
            "CNY": str(sum(Decimal(str(r["actual_cost"])) for r in entries)),
            "requests": sum(r["requests"] for r in entries),
        }
    return {}


def endpoint(provider):
    if provider == "deepseek":
        return "https://api.deepseek.com/chat/completions"
    if provider == "jev":
        return "https://api.apikey.fan/v1/systemone"
    pilot.configure_aliyun(
        os.environ["DASHSCOPE_WORKSPACE_ID"], os.environ["DASHSCOPE_REGION"]
    )
    return pilot.ENDPOINT


def atomic(path, item):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=1), encoding="utf8")
    tmp.replace(path)


def question(text, options):
    return {"type": "choice", "instructions": text, "criteria": options}


def call(
    provider,
    url,
    key,
    state,
    questions,
    *,
    system_prompt=DEEPSEEK_FORMAT,
    max_tokens=160,
):
    deep = provider == "deepseek"
    body = {"model": MODEL[provider], "state": state, "questions": questions}
    if deep:
        body = {
            "model": MODEL[provider],
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"state": state, "questions": questions}, ensure_ascii=False
                    ),
                },
            ],
        }
    start = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    observations = []
    for attempt in range(1, 4):
        try:
            res = request_json(url, key, body)
            observations.append(res.get("usage", {}))
            if res.get("model") != MODEL[provider]:
                raise ValueError("unexpected_model")
            if deep:
                output = res["choices"][0]
                if output["finish_reason"] != "stop":
                    raise ValueError("incomplete_json")
                answers = json.loads(output["message"]["content"])
                confidence = {}
                probabilities = {}
            else:
                answers = {q: a["choice"] for q, a in res["answers"].items()}
                confidence = {q: a.get("confidence") for q, a in res["answers"].items()}
                probabilities = {
                    q: a.get("probabilities") for q, a in res["answers"].items()
                }
            if set(questions) != set(answers) or not all(
                (isinstance(answers[q], list) and len(answers[q]) <= spec["max_items"])
                if deep and spec.get("type") == "array"
                else isinstance(answers[q], str)
                and (
                    0 < len(answers[q].strip()) <= spec["max_length"]
                    if deep and spec.get("type") == "short_text"
                    else answers[q] in spec["criteria"]
                )
                for q, spec in questions.items()
            ):
                raise ValueError("invalid_choice")
            return {
                "answers": answers,
                "confidence": confidence,
                "probabilities": probabilities,
                "returned_model": res.get("model"),
                "request_id": res.get("id", res.get("request_id")),
                "started": started,
                "seconds": round(time.perf_counter() - start, 6),
                "attempts": attempt,
                "usage": observations,
            }
        except HTTPError as exc:
            error = "HTTP " + str(exc.code)
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (
            TimeoutError,
            URLError,
            ConnectionError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            error = type(exc).__name__
        if attempt < 3:
            time.sleep(2**attempt)
    return {
        "error": error,
        "started": started,
        "seconds": round(time.perf_counter() - start, 6),
        "attempts": attempt,
        "usage": observations,
    }


def original_plans():
    return {
        n: json.loads(
            (ds.ROOT / "data/batches" / (n + ".json")).read_text(encoding="utf8")
        )
        for n in pilot.METHODS
    }


def run(study, provider, workers):
    if study.startswith(("branch_", "scope_confirm_")):
        import branch_permissions as design

        plans, protocol = design.run_config(study, provider)
    elif study.startswith(("natural_", "replica_")):
        import natural_permissions as design

        plans, protocol = design.run_config(study, provider)
    elif study in (
        "full_permission",
        "full_permission_review",
        "numeric_permission",
        "numeric_permission_review",
    ):
        import full_permission as design

        plans, protocol = design.run_config(study, provider)
    elif study in ("material_facts", "material_fact_replies"):
        import material_facts as design

        if provider != "deepseek" and (study != design.STUDY or provider != "jev"):
            raise ValueError(
                "Only the primary experiment supports a direct Jev comparator."
            )
        plans, protocol = (
            design.replay_config() if study == design.REPLAY else design.config()
        )
        if provider == "jev":
            protocol = {
                "questions": {
                    name: design.questions(plan, "control")
                    for name, plan in plans.items()
                }
            }
    elif study in ("parameter_router", "parameter_router_extra"):
        import parameter_router as design

        plans, protocol = (
            design.extra_config() if study == design.EXTRA else design.config()
        )
        if provider == "jev":
            from study_report import inputs

            direct_plans, decisions, _ = inputs(study, "deepseek")
            if direct_plans != plans:
                raise ValueError("Different DeepSeek inputs")
            provisional = {
                (r["dataset"], r["id"]): r["answers"]["decision"]
                for r in decisions
                if not r.get("error")
            }
        elif provider != "deepseek":
            raise ValueError("Parameter routing requires Jev and DeepSeek")
    elif study == "jev_triage":
        import jev_triage as design

        plans, protocol = design.config()
        if provider == "deepseek":
            plans = {design.FRESH: plans[design.FRESH]}
        elif provider != "jev":
            raise ValueError("Triage uses Jev and DeepSeek")
    elif study == "retained_validation":
        import retained_validation as design

        plans, protocol = design.config()
        if provider == "aliyun":
            plans = {
                name: plan
                for name, plan in plans.items()
                if "aliyun_direct" in protocol["eligible"][name]
            }
    elif study in ("shared_review", "commitment_review"):
        import shared_review as design

        if provider != "deepseek":
            raise ValueError("Grant-first review uses DeepSeek")
        plans, protocol = (
            design.config() if study == "shared_review" else design.commitment_config()
        )
    elif study in ("reply_validation", "reply_precedent"):
        import reply_review as design

        if provider not in ("jev", "deepseek") or (
            study == "reply_precedent" and provider != "deepseek"
        ):
            raise ValueError("Reply validation uses Jev and DeepSeek")
        plans, protocol = (
            design.precedent_config() if study == "reply_precedent" else design.config()
        )
    elif study in (
        "parameter_calibration",
        "parameter_test",
        "parameter_versions",
        "parameter_final",
    ):
        import parameter_review as design

        if provider == "aliyun":
            raise ValueError("The parameter experiment specifies Jev and DeepSeek")
        plans, protocol = design.config(study)
    elif study in ("conditions", "confirmation"):
        if study == "confirmation":
            import authorization_confirmation as design

            if provider == "aliyun":
                raise ValueError(
                    "The two-review confirmation specifies Jev and DeepSeek"
                )
        else:
            import authorization_conditions as design

        plans = design.populations()
        protocol = design.description()
    elif study in ("review", "risk", "coverage"):
        if study in ("risk", "coverage"):
            import risk_tree as design
        else:
            import review_design as design
        plans = design.coverage_plans() if study == "coverage" else design.plans()
        protocol = design.protocol()
    else:
        plans = original_plans()
        protocol = {}
    root = ds.ROOT / "data" / "study_runs" / study / provider
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "plan.json"
    ledger = root / "billing.json"
    log = root / "calls.jsonl"
    key = os.environ[
        {
            "deepseek": "DEEPSEEK_API_KEY",
            "jev": "TYPESAFE_API_KEY",
            "aliyun": "DASHSCOPE_API_KEY",
        }[provider]
    ]
    url = endpoint(provider)
    with pilot.exclusive(root / "run.lock"):
        config = {
            "model": MODEL[provider],
            "wrapper": DEEPSEEK_FORMAT if provider == "deepseek" else None,
            "plans": plans,
            "protocol": protocol,
        }
        if (
            snapshot.exists()
            and json.loads(snapshot.read_text(encoding="utf8")) != config
        ):
            raise RuntimeError("Different inputs or protocol in saved run")
        if not snapshot.exists():
            atomic(snapshot, config)
        cache = (
            {(r["dataset"], r["id"], r["stage"]): r for r in ds.records(log)}
            if log.exists()
            else {}
        )
        budget = (
            json.loads(ledger.read_text(encoding="utf8")) if ledger.exists() else {}
        )
        if "before" not in budget:
            budget = {
                "before": account(provider, key),
                "start": datetime.now(UTC).isoformat(),
            }
            atomic(ledger, budget)

        def measure(name, item, stage, questions, state=None, **options):
            k = (name, item["id"], stage)
            if k in cache and "answers" in cache[k]:
                return cache[k]["answers"]
            result = call(
                provider,
                url,
                key,
                item["state"] if state is None else state,
                questions,
                **options,
            )
            row = {"dataset": name, "id": item["id"], "stage": stage, **result}
            with WRITE:
                pilot.append_json(log, row)
                cache[k] = row
            if "error" in result:
                raise RuntimeError(str(k) + ": " + result["error"])
            return result["answers"]

        def evaluate(name, item):
            if study in ("branch_attest", "scope_confirm_attest"):
                packet, questions = protocol["packets"][item["id"]]
                measure(name, item, "attest", questions, packet)
                return
            if study in (
                "branch_extract",
                "branch_original",
                "scope_confirm_extract",
                "scope_confirm_original",
            ):
                measure(
                    name,
                    item,
                    "extract",
                    protocol["questions"][name],
                    design.prose.request_view(item["state"]),
                    system_prompt=protocol["system"],
                    max_tokens=protocol["max_tokens"],
                )
                return
            if study in ("natural_extract", "replica_extract"):
                measure(
                    name,
                    item,
                    "extract",
                    protocol["questions"][name],
                    design.request_view(item["state"]),
                    system_prompt=protocol["system"],
                    max_tokens=protocol["max_tokens"],
                )
                return
            if study in (
                "natural_direct",
                "natural_review",
                "natural_matched",
                "replica_direct",
                "replica_review",
                "replica_matched",
                "branch_direct",
                "branch_review",
                "scope_confirm_direct",
                "scope_confirm_review",
                "full_permission",
                "full_permission_review",
                "numeric_permission",
                "numeric_permission_review",
            ):
                options = (
                    {
                        "system_prompt": protocol["system"],
                        "max_tokens": protocol["max_tokens"],
                    }
                    if protocol["mode"] == "review"
                    else {}
                )
                measure(
                    name, item, protocol["mode"], protocol["questions"][name], **options
                )
                return
            if study == "material_fact_replies":
                branches = (
                    ("yes", "no")
                    if positions[name, item["id"]] % 2 == 0
                    else ("no", "yes")
                )
                for answer in branches:
                    state = {
                        "original_record": item["state"],
                        "verification": {
                            "instructions": protocol["instruction"],
                            "question": item["clarification"],
                            "verified_answer": answer,
                        },
                    }
                    measure(
                        name,
                        item,
                        answer,
                        design.questions(plans[name], "rule"),
                        state,
                        system_prompt=protocol["system"],
                    )
                return
            if study == "material_facts":
                if provider == "jev":
                    measure(name, item, "direct", protocol["questions"][name])
                    return
                pos = positions[name, item["id"]]
                for stage in design.order(pos):
                    arm = protocol["arms"][stage]
                    measure(
                        name,
                        item,
                        stage,
                        design.questions(plans[name], stage),
                        system_prompt=arm["system"],
                        max_tokens=arm["max_tokens"],
                    )
                return
            if study in ("parameter_router", "parameter_router_extra"):
                plan = plans[name]
                if provider == "deepseek":
                    measure(
                        name,
                        item,
                        "direct",
                        {"decision": question(plan["instructions"], plan["choices"])},
                    )
                else:

                    def invoke(stage, questions, state):
                        return measure(name, item, stage, questions, state)

                    design.run_jev(invoke, item, plan, provisional[name, item["id"]])
                return
            if study == "jev_triage":
                plan = plans[name]
                if provider == "jev":
                    measure(
                        name,
                        item,
                        "triage",
                        design.QUESTIONS,
                        {
                            "decision_task": plan["instructions"],
                            "record": item["state"],
                        },
                    )
                else:
                    measure(
                        name,
                        item,
                        "direct",
                        {"decision": question(plan["instructions"], plan["choices"])},
                    )
                return
            if study == "retained_validation":

                def invoke(stage, questions, state):
                    return measure(name, item, stage, questions, state)

                design.run_case(
                    invoke,
                    name,
                    item,
                    plans[name],
                    provider,
                    positions[name, item["id"]],
                )
                return
            if study in ("shared_review", "commitment_review"):

                def invoke(stage, questions, state):
                    return measure(name, item, stage, questions, state)

                design.run_case(invoke, name, item, plans[name])
                return
            if study in ("reply_validation", "reply_precedent"):

                def invoke(stage, questions, state):
                    return measure(name, item, stage, questions, state)

                pos = positions[name, item["id"]]
                if study == "reply_precedent":
                    design.precedent_case(invoke, item, plans[name], pos)
                else:
                    design.run_case(
                        invoke, name, item["state"], plans[name], provider, pos
                    )
                return
            if study in (
                "conditions",
                "confirmation",
                "parameter_calibration",
                "parameter_test",
                "parameter_versions",
                "parameter_final",
            ):

                def invoke(stage, questions, state):
                    return measure(name, item, stage, questions, state)

                if study.startswith("parameter_"):
                    design.run_case(
                        invoke,
                        item["state"],
                        plans[name],
                        provider,
                        positions[name, item["id"]],
                        study,
                    )
                    return
                if study == "confirmation":
                    design.run_case(
                        invoke,
                        item["state"],
                        plans[name],
                        provider,
                        positions[name, item["id"]],
                    )
                    return
                design.run_case(
                    invoke,
                    item["state"],
                    plans[name],
                    item,
                    positions[name, item["id"]],
                )
                return
            if study == "coverage":
                measure(name, item, "root", protocol["root"])
                route = measure(name, item, "shadow_router", protocol["router"])["risk"]
                measure(name, item, "shadow_leaf", protocol["leaves"][route])
                return
            if study == "risk":
                screen = measure(name, item, "root", protocol["root"])["decision"]
                if screen == "uncertain":
                    risk = measure(name, item, "router", protocol["router"])["risk"]
                    # Alternate leaf order independently of the reference label.
                    leaves = [
                        ("generic", protocol["generic"]),
                        ("specialized", protocol["leaves"][risk]),
                    ]
                    for stage, questions in (
                        leaves if leaf_order[(name, item["id"])] else reversed(leaves)
                    ):
                        measure(name, item, stage, questions)
                return
            plan = plans[name]
            native = {"decision": question(plan["instructions"], plan["choices"])}
            measure(name, item, "A", native)
            if study == "pilot":
                return
            route = measure(name, item, "B", protocol["B"])["decision"]
            if route in design.HOLDS:
                measure(name, item, "C", protocol["C"][route])
                measure(name, item, "D", protocol["D"][route])

        cases = [
            (name, item) for name, plan in plans.items() for item in plan["samples"]
        ]
        # A full first case catches credential or schema errors before concurrent dispatch.
        positions = {(name, item["id"]): i for i, (name, item) in enumerate(cases)}
        leaf_order = {key: pos % 2 == 0 for key, pos in positions.items()}
        if not cases:
            if "after" not in budget:
                budget.update(
                    after=account(provider, key), end=datetime.now(UTC).isoformat()
                )
                atomic(ledger, budget)
            print(study, provider, "no eligible cases", flush=True)
            return
        evaluate(*cases[0])
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = [executor.submit(evaluate, *pair) for pair in cases[1:]]
            for task in as_completed(tasks):
                try:
                    task.result()
                except RuntimeError as exc:
                    failed.append(str(exc))
        if failed:
            raise RuntimeError(f"{len(failed)} incomplete cases; first: {failed[0]}")
        if "after" not in budget:
            budget.update(
                after=account(provider, key), end=datetime.now(UTC).isoformat()
            )
            atomic(ledger, budget)
        print(study, provider, len(cases), "cases completed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "study",
        choices=[
            "natural_direct",
            "natural_review",
            "natural_extract",
            "natural_matched",
            "replica_direct",
            "replica_review",
            "replica_extract",
            "replica_matched",
            "branch_direct",
            "branch_review",
            "branch_original",
            "branch_extract",
            "branch_attest",
            "scope_confirm_direct",
            "scope_confirm_review",
            "scope_confirm_original",
            "scope_confirm_extract",
            "scope_confirm_attest",
            "full_permission",
            "full_permission_review",
            "numeric_permission",
            "numeric_permission_review",
            "material_facts",
            "material_fact_replies",
            "parameter_router",
            "parameter_router_extra",
            "jev_triage",
            "retained_validation",
            "shared_review",
            "commitment_review",
            "reply_validation",
            "reply_precedent",
            "pilot",
            "review",
            "risk",
            "coverage",
            "conditions",
            "confirmation",
            "parameter_calibration",
            "parameter_test",
            "parameter_versions",
            "parameter_final",
        ],
    )
    p.add_argument("--provider", choices=MODEL, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    args = p.parse_args()
    run(args.study, args.provider, args.workers)

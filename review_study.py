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


def call(provider, url, key, state, questions):
    deep = provider == "deepseek"
    body = {"model": MODEL[provider], "state": state, "questions": questions}
    if deep:
        body = {
            "model": MODEL[provider],
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 160,
            "messages": [
                {"role": "system", "content": DEEPSEEK_FORMAT},
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
                isinstance(answers[q], str) and answers[q] in spec["criteria"]
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
    if study in ("review", "risk", "coverage"):
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

        def measure(name, item, stage, questions):
            k = (name, item["id"], stage)
            if k in cache and "answers" in cache[k]:
                return cache[k]["answers"]
            result = call(provider, url, key, item["state"], questions)
            row = {"dataset": name, "id": item["id"], "stage": stage, **result}
            with WRITE:
                pilot.append_json(log, row)
                cache[k] = row
            if "error" in result:
                raise RuntimeError(str(k) + ": " + result["error"])
            return result["answers"]

        def evaluate(name, item):
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
        leaf_order = {
            (name, item["id"]): i % 2 == 0 for i, (name, item) in enumerate(cases)
        }
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
    p.add_argument("study", choices=["pilot", "review", "risk", "coverage"])
    p.add_argument("--provider", choices=MODEL, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    args = p.parse_args()
    run(args.study, args.provider, args.workers)

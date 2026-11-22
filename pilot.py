"""Run static decisions on public benchmark inputs. Outputs are checkpointed per case."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import datasets_local as ds

ENDPOINT = "https://api.apikey.fan/v1/systemone"
MODEL = "jev-1.14.0"
OUT = ds.ROOT / "results"
DATA = ds.ROOT / "data" / "batches"
METHODS = {
    "authgate": "原生二分类；25 对，无合成信号",
    "delegation": "原生四类决策；只抽有多数意见的场景",
    "jev": "官方三态标签；测试集，直接三分类",
    "authbench": "派生文件权限题；每个样本一个任务和一个候选权限",
    "authorization": "作者列明越权行为的拦截率；仅负例",
    "yolofs": "静态描述预测损害；25 对，不运行工具",
    "agentdojo": "参考方案首个具名工具的相关性；仅正例",
    "tsbench": "官方逐步风险标签；严格二分类",
    "agentdam": "识别人工列出的敏感项；仅正例",
    "privacylens": "种子情境共享适当性；仅不当共享",
    "privacylive": "与作者对既有邮件的泄露判断一致率；静态",
}


def configure_aliyun(workspace, region):
    global ENDPOINT, MODEL, OUT, DATA
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", workspace or "") or region not in (
        "cn-beijing",
        "ap-southeast-1",
    ):
        raise ValueError(
            "Provide DASHSCOPE_WORKSPACE_ID and a supported DASHSCOPE_REGION."
        )
    ENDPOINT = (
        f"https://{workspace}.{region}.maas.aliyuncs.com/compatible-mode/v1/systemone"
    )
    MODEL = "decision-model-preview"
    OUT = ds.ROOT / "results" / "aliyun"
    DATA = ds.ROOT / "data" / "aliyun-batches"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def append_json(path, obj):
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


@contextmanager
def exclusive(path):
    try:
        with path.open("x", encoding="ascii") as f:
            f.write(str(os.getpid()))
    except FileExistsError as e:
        raise RuntimeError(f"Existing run lock: {path}") from e
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def predict(item, instructions, choices, key):
    body = {
        "model": MODEL,
        "state": item["state"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": instructions,
                "criteria": choices,
            }
        },
    }
    req = Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST",
    )
    start = time.monotonic()
    for attempt in range(3):
        try:
            with build_opener(NoRedirect()).open(req, timeout=85) as response:
                payload = json.load(response)
            ans = payload["answers"]["decision"]
            if ans["choice"] not in choices or payload.get("model") != MODEL:
                raise ValueError("unexpected model or label")
            return {
                "prediction": ans["choice"],
                "confidence": ans.get("confidence"),
                "model": payload["model"],
                "usage": payload.get("usage", {}),
                "seconds": round(time.monotonic() - start, 3),
                "attempts": attempt + 1,
            }
        except HTTPError as exc:
            reason = "HTTP " + str(exc.code)
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, ConnectionError) as exc:
            reason = type(exc).__name__
        except (KeyError, ValueError, TypeError) as exc:
            reason = "invalid_response_" + type(exc).__name__
            break
        if attempt < 2:
            time.sleep(3 * (attempt + 1))
    return {"error": reason, "attempts": attempt + 1}


def run(name, limit, workers, prepare_only=False):
    OUT.mkdir(exist_ok=True, parents=True)
    DATA.mkdir(exist_ok=True, parents=True)
    items, question, labels = getattr(ds, name)(limit)
    ids = [r["id"] for r in items]
    if len(ids) != len(set(ids)) or any(
        r["gold"] is not None and r["gold"] not in labels for r in items
    ):
        raise ValueError("Duplicate case ID or unrecognized reference label")
    manifest = {
        "benchmark": name,
        "seed": ds.SEED,
        "limit": limit,
        "model": MODEL,
        "instructions": question,
        "choices": labels,
        "samples": items,
    }
    plan_file = DATA / (name + ".json")
    log = OUT / (name + ".jsonl")
    with exclusive(OUT / (name + ".lock")):
        if (
            plan_file.exists()
            and json.loads(plan_file.read_text(encoding="utf8")) != manifest
        ):
            raise RuntimeError(
                f"Saved study differs for {name}; use a fresh results and batches directory."
            )
        if log.exists() and not plan_file.exists():
            raise RuntimeError("Saved results have no matching sample manifest.")
        if not plan_file.exists():
            tmp = plan_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf8")
            tmp.replace(plan_file)
        if prepare_only:
            print(name, "prepared", len(items), flush=True)
            return
        old = {r["id"]: r for r in ds.records(log)} if log.exists() else {}
        todo = [r for r in items if not old.get(r["id"], {}).get("prediction")]
        if todo:
            key = os.environ[
                "DASHSCOPE_API_KEY"
                if MODEL == "decision-model-preview"
                else "TYPESAFE_API_KEY"
            ]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {
                    pool.submit(predict, r, question, labels, key): r for r in todo
                }
                for future in as_completed(pending):
                    r = pending[future]
                    res = {
                        "id": r["id"],
                        "reference": r["gold"],
                        "group": r["group"],
                        "utc": datetime.now(timezone.utc).isoformat(),
                        **future.result(),
                    }
                    append_json(log, res)
                    old[r["id"]] = res
        errors = Counter(r.get("error") for r in old.values() if r.get("error"))
        print(
            name,
            "planned",
            len(items),
            "valid",
            sum("prediction" in r for r in old.values()),
            "errors",
            dict(errors),
            flush=True,
        )


def summarize():
    OUT.mkdir(exist_ok=True, parents=True)
    fields = [
        "benchmark",
        "measurement",
        "sampled",
        "responses",
        "scored",
        "correct",
        "errors",
        "input_tokens",
        "output_tokens",
        "distribution",
    ]
    with (
        (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as out,
        (OUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as detail,
    ):
        summary = csv.DictWriter(out, fieldnames=fields)
        summary.writeheader()
        results = csv.writer(detail)
        results.writerow(
            [
                "benchmark",
                "sample",
                "group",
                "reference",
                "prediction",
                "confidence",
                "error",
            ]
        )
        for name, method in METHODS.items():
            f = DATA / (name + ".json")
            if not f.exists():
                continue
            items = json.loads(f.read_text(encoding="utf8"))["samples"]
            log = OUT / (name + ".jsonl")
            old = {r["id"]: r for r in ds.records(log)} if log.exists() else {}
            good = [r for r in old.values() if "prediction" in r]
            scored = [r for r in good if r["reference"] is not None]
            dist = Counter(r["prediction"] for r in good)
            row = [
                name,
                method,
                len(items),
                len(good),
                len(scored),
                sum(r["reference"] == r["prediction"] for r in scored),
                len(items) - len(good),
                sum(r["usage"].get("input_tokens", 0) for r in good),
                sum(r["usage"].get("output_tokens", 0) for r in good),
                json.dumps(dist, ensure_ascii=False),
            ]
            summary.writerow(dict(zip(fields, row)))
            for case in items:
                r = old.get(case["id"], {})
                results.writerow(
                    [
                        name,
                        case["id"],
                        case["group"],
                        case["gold"],
                        r.get("prediction"),
                        r.get("confidence"),
                        r.get("error"),
                    ]
                )
            print(*row[:7], flush=True)


def export_samples():
    folder = DATA.parent / ("pilot50" if MODEL == "jev-1.14.0" else "aliyun-samples")
    folder.mkdir(parents=True, exist_ok=True)
    with (
        (folder / "inputs.jsonl").open("w", encoding="utf8") as inputs,
        (folder / "references.jsonl").open("w", encoding="utf8") as refs,
    ):
        for name in METHODS:
            f = DATA / (name + ".json")
            if not f.exists():
                continue
            plan = json.loads(f.read_text(encoding="utf8"))
            for row in plan["samples"]:
                identifier = {"benchmark": name, "id": row["id"]}
                body = {
                    "state": row["state"],
                    "questions": {
                        "decision": {
                            "type": "choice",
                            "instructions": plan["instructions"],
                            "criteria": plan["choices"],
                        }
                    },
                }
                inputs.write(
                    json.dumps({**identifier, "request": body}, ensure_ascii=False)
                    + "\n"
                )
                refs.write(
                    json.dumps(
                        {**identifier, "reference": row["gold"], "group": row["group"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "run", "summarize", "export"])
    p.add_argument("datasets", nargs="*")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--provider", choices=("jev", "aliyun"), default="jev")
    args = p.parse_args()
    if args.provider == "aliyun":
        configure_aliyun(
            os.getenv("DASHSCOPE_WORKSPACE_ID"),
            os.getenv("DASHSCOPE_REGION", "cn-beijing"),
        )
    if args.command in ("run", "prepare"):
        if (
            not args.datasets
            or any(n not in METHODS for n in args.datasets)
            or not 1 <= args.workers <= 16
            or args.limit < 1
        ):
            p.error("Specify a known dataset and positive limits.")
        for name in args.datasets:
            run(name, args.limit, args.workers, args.command == "prepare")
    elif args.command == "export":
        export_samples()
    else:
        summarize()

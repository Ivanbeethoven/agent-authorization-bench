"""Download public task data. Upstream programs are never imported or executed."""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def download(url):
    for attempt in range(4):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "agent-authorization-bench/1"}),
                timeout=40,
            ) as res:
                return res.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            if getattr(exc, "code", 0) in (400, 401, 403, 404) or attempt == 3:
                raise
            time.sleep(2**attempt)


def save_jsonl(name, rows):
    folder = RAW / name
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "cases.tmp"
    with tmp.open("w", encoding="utf8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(folder / "cases.jsonl")


def yolofs():
    repo = "wudaoz/Caper"
    tree = json.loads(
        download(f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1")
    )
    if tree.get("truncated"):
        raise ValueError("Incomplete upstream file listing")
    paths = sorted(
        n["path"]
        for n in tree["tree"]
        if re.fullmatch(
            r"yolofs_ext/data/tasks/(dangerous|safe_twin)/[^/]+\.json", n["path"]
        )
    )
    if not paths:
        raise ValueError("No YoloFS-Ext task files")

    def read_task(path):
        case = json.loads(
            download(f"https://raw.githubusercontent.com/{repo}/main/{quote(path)}")
        )
        return {k: case[k] for k in ("id", "type", "twin_id", "request", "command")}

    with ThreadPoolExecutor(max_workers=10) as pool:
        cases = list(pool.map(read_task, paths))
    by_id = {r["id"]: r for r in cases}
    assert len(by_id) == len(cases)
    assert all(by_id[r["twin_id"]]["twin_id"] == r["id"] for r in cases)
    save_jsonl("yolofs", cases)
    print("yolofs", len(cases), flush=True)


def literal_text(node, constants):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, ast.JoinedStr):
        return "".join(literal_text(p, constants) for p in node.values)
    if isinstance(node, ast.FormattedValue):
        return str(literal_text(node.value, constants))
    raise ValueError("Dynamic expression")


def privacy():
    base = "https://raw.githubusercontent.com/microsoft/ACV/main/misc/PrivacyInAction/PrivacyLens-Live"
    lens = RAW / "privacylens"
    lens.mkdir(parents=True, exist_ok=True)
    for rel in (
        "main_data.json",
        "extensibility/confide_subset.json",
        "extensibility/culturebank_subset.json",
    ):
        dest = lens / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            download(
                "https://raw.githubusercontent.com/SALT-NLP/PrivacyLens/main/data/"
                + rel
            )
        )
    folder = RAW / "privacylive"
    folder.mkdir(parents=True, exist_ok=True)
    for rel in (
        "MCP-2Tools/baseline/filtered_data.json",
        "MCP-2Tools/baseline/main_data.json",
        "MCPA2A-2Tools/baseline/A2A_data.json",
        "MCPA2A-3Tools/baseline/new_A2A_data.json",
        "MCPA2A-3Tools/baseline/A2A_data.json",
        "MCPA2A-3Tools/baseline/main_data.json",
    ):
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(download(base + "/" + rel))

    def read_item(number):
        return json.loads(
            download(
                base
                + "/MCP-2Tools/baseline/check_results/check_result_item"
                + str(number)
                + ".json"
            )
        )

    source = json.loads(
        (folder / "MCP-2Tools/baseline/filtered_data.json").read_text(encoding="utf8")
    )
    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(read_item, range(1, len(source) + 1)))
    save_jsonl("privacylive", rows)
    print("privacylive", len(rows), flush=True)


def dojo():
    cases = []
    for suite in ("banking", "slack", "travel", "workspace"):
        url = f"https://raw.githubusercontent.com/ethz-spylab/agentdojo/main/src/agentdojo/default_suites/v1/{suite}/user_tasks.py"
        root = ast.parse(download(url).decode())
        for cls in root.body:
            if not isinstance(cls, ast.ClassDef) or not re.fullmatch(
                r"UserTask\d+", cls.name
            ):
                continue
            constants = {}
            for item in cls.body:
                if isinstance(item, ast.Assign) and isinstance(
                    item.targets[0], ast.Name
                ):
                    try:
                        constants[item.targets[0].id] = literal_text(
                            item.value, constants
                        )
                    except (KeyError, ValueError):
                        pass
            expected = []
            gt = next(
                (
                    x
                    for x in cls.body
                    if isinstance(x, ast.FunctionDef) and x.name == "ground_truth"
                ),
                None,
            )
            if gt:
                calls = sorted(
                    (
                        c
                        for c in ast.walk(gt)
                        if isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Name)
                        and c.func.id == "FunctionCall"
                    ),
                    key=lambda c: (c.lineno, c.col_offset),
                )
                for c in calls:
                    for kw in c.keywords:
                        if kw.arg == "function" and isinstance(kw.value, ast.Constant):
                            expected.append(kw.value.value)
            cases.append(
                {
                    "id": suite + "/" + cls.name,
                    "suite": suite,
                    "request": constants.get("PROMPT"),
                    "reference_tools": expected,
                }
            )
    save_jsonl("agentdojo", cases)
    print(
        "agentdojo",
        len(cases),
        "eligible",
        sum(bool(x["request"] and x["reference_tools"]) for x in cases),
        flush=True,
    )


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        {"yolofs": yolofs, "agentdojo": dojo, "privacy": privacy}[arg]()

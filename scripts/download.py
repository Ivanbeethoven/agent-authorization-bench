"""Retrieve the public input files required by the pilot from their publishers."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from zipfile import ZipFile
from fetch_tasks import RAW, download, dojo, privacy, yolofs

REPOS = {
    "authgate": "Digital-Trust-Lab/AuthGateBench",
    "delegation": "PieLord757/delegation-check",
    "authbench": "evolvent-ai/Authbench",
    "authorization": "PEACH-Research-Lab/AuthorizationBench",
    "agentdam": "facebookresearch/ai-agent-privacy",
}


def fetch_files(dataset, base, paths):
    def fetch(rel):
        dest = RAW / dataset / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(download(base + "/" + quote(rel, safe="/")))
        tmp.replace(dest)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, paths))


def github(name):
    repo = REPOS[name]
    base = f"https://raw.githubusercontent.com/{repo}/main"
    if name == "authgate":
        paths = ["data/authgatebench.jsonl", "README.md"]
    elif name == "delegation":
        paths = [
            "delegationcheck_v1_0.jsonl",
            "data/reference/expert_majority_reference.csv",
            "LICENSE-DATA.md",
            "README.md",
        ]
    else:
        listing = json.loads(
            download(f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1")
        )
        if listing.get("truncated"):
            raise ValueError("Incomplete upstream directory")
        files = [o["path"] for o in listing["tree"] if o["type"] == "blob"]
        if name == "authbench":
            paths = [
                f
                for f in files
                if f.startswith("tasks/")
                and (
                    f.endswith("/instruction.md")
                    or f.endswith("/tests/permission_eval_spec.json")
                )
            ]
        elif name == "authorization":
            paths = ["benchmark/manifest.json"] + [
                f
                for f in files
                if f.startswith("authorization_eva_sandbox/tasks/")
                and f.endswith("/vignette.json")
            ]
        else:
            paths = [
                f
                for f in files
                if f.startswith("agentdam/data/wa_format/")
                and f.endswith("_privacy.json")
            ]
        paths += ["README.md"]
    fetch_files(name, base, paths)


def boundary():
    base = "https://zenodo.org/api/records/22952538/files/agent-authorization-evidence-v1.2.0.zip/content"
    with ZipFile(io.BytesIO(download(base))) as z:
        for rel in (
            "corpus/fixtures.jsonl",
            "corpus/policies/balanced.md",
            "corpus/policies/strict.md",
            "corpus/policies/delegated.md",
            "corpus/README.md",
            "historical/data/fixtures.jsonl",
            "historical/data/policy.md",
            "historical/data/disputed_ids.json",
            "LICENSES.md",
        ):
            dst = RAW / "jev" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read("agent-authorization-evidence-v1.2.0/" + rel))


def fetch(name):
    if name in REPOS:
        github(name)
    elif name == "jev":
        boundary()
    elif name == "agentdojo":
        dojo()
    elif name == "yolofs":
        yolofs()
    elif name == "privacy":
        privacy()
    elif name == "xaab":
        base = "https://raw.githubusercontent.com/xybern-ai/agent-authz-benchmark/0249d9ccee4f48a9d49da3363d478715ff99f335"
        files = (
            "destructive_ops", "privilege_escalation", "credential_access", "data_exfiltration",
            "pii_disclosure", "prompt_injection", "unsafe_code", "velocity_abuse", "financial",
            "social_engineering", "evasion_variants", "hard_adversarial", "hard_benign",
            "agent_to_agent", "information_flow", "authority_attacks", "intent_drift", "pre_execution",
        )
        fetch_files(name, base, [f"scenarios/{i:02d}_{f}.yaml" for i, f in enumerate(files, 1)]
                    + ["policy-pack.json", "README.md", "METHODOLOGY.md", "LICENSE"])
    elif name == "tsbench":
        base = "https://huggingface.co/datasets/MurrayTom/TS-Bench/resolve/main"
        fetch_files(
            name,
            base,
            [
                "data/eval/" + src + "-Traj.jsonl"
                for src in ("AgentDojo", "AgentHarm", "ASB")
            ]
            + ["README.md"],
        )
    else:
        raise ValueError("Unknown benchmark: " + name)
    print("Downloaded:", name, flush=True)


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:] or [
        *REPOS,
        "jev",
        "tsbench",
        "agentdojo",
        "yolofs",
        "privacy",
    ]:
        fetch(name)

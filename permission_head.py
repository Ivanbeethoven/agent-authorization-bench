"""Fit and serve small binary authorization heads on frozen AgentDecision states."""

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from permission_features import DEFAULT_TEMPLATE, OUT, connect, extract, frozen
from permission_training_data import records

HEADS = {"single": (1, False), "dual": (2, False), "paired": (2, True)}
RIDGE = (0.0001, 0.001, 0.01)
POLICIES = ("nominal", "risk02", "zero")
DEFAULT_GGUF = Path("D:/AgentPermissionModels/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf")


def bank(split):
    data = records(split)
    db = connect()
    try:
        cached = {
            r[0]: r[1:]
            for r in db.execute(
                "SELECT id,coverage,tokens,pair,seconds FROM vectors WHERE split=?",
                (split,),
            )
        }
    finally:
        db.close()
    ids = [r[0] for r in data]
    if len(set(ids)) != len(ids) or cached.keys() != set(ids):
        raise ValueError(f"{split}: requested {len(ids)}, cached {len(cached)}")
    features = np.zeros((len(ids), 2, 2560), np.float32)
    info = []
    for k, key in enumerate(ids):
        cover, tokens, blob, seconds = cached[key]
        if blob is not None:
            features[k] = np.frombuffer(blob, dtype="<f2").reshape(2, 2560)
        elif cover != "unrepresentable":
            raise ValueError("Missing representation")
        info.append((cover, tokens, seconds))
    return ids, np.array([r[1] for r in data]), [r[3] for r in data], features, info


def threshold(scores, labels, risk):
    negatives = np.sort(scores[labels == 0])[::-1]
    cutoff = negatives[math.floor(risk * len(negatives))]
    return float(np.nextafter(np.float64(cutoff), np.inf))


def scores(x, w, bias):
    return np.einsum("nvd,vd->n", x, w, dtype=np.float64, optimize=False) + float(bias)


def fit(x, labels, groups, nviews, pairwise, penalty):
    import torch
    import torch.nn.functional as fn

    features = x[:, :nviews]
    center = features.mean(axis=0, dtype=np.float64).astype("f")
    scale = np.sqrt(((features - center) ** 2).sum(axis=2).mean(axis=0)) * math.sqrt(
        nviews
    )
    design = torch.tensor(((features - center) / scale[:, None]).reshape(len(x), -1))
    target = torch.tensor(labels, dtype=torch.float32)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    case_weights = torch.tensor(
        [1.0 / (len(members[g]) * len(members)) for g in groups]
    )
    left, right, pair_weights = [], [], []
    if pairwise:
        for indices in members.values():
            positive = [i for i in indices if labels[i] == 1]
            negative = [i for i in indices if labels[i] == 0]
            if not positive or not negative:
                raise ValueError(
                    "The ranking loss requires both outcomes in each training family"
                )
            for i in positive:
                for j in negative:
                    left.append(i)
                    right.append(j)
                    pair_weights.append(
                        1 / (len(members) * len(positive) * len(negative))
                    )
        left, right = torch.tensor(left), torch.tensor(right)
        pair_weights = torch.tensor(pair_weights)
    w = torch.zeros(design.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [w, b],
        max_iter=70,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def loss():
        optimizer.zero_grad()
        z = design @ w + b
        obj = (
            fn.binary_cross_entropy_with_logits(z, target, reduction="none")
            * case_weights
        ).sum()
        if pairwise:
            obj = obj + 0.5 * (fn.softplus(z[right] - z[left]) * pair_weights).sum()
        obj = obj + 0.5 * penalty * w.square().sum()
        obj.backward()
        return obj

    optimizer.step(loss)
    weight = np.zeros((2, 2560), dtype="f")
    weight[:nviews] = w.detach().numpy().reshape(nviews, -1) / scale[:, None]
    intercept = b.item() - np.sum(weight[:nviews] * center, dtype=np.float64)
    return weight, float(intercept)


def partition(groups):
    strata = defaultdict(set)
    for g in groups:
        strata[g.rsplit("/", 1)[0]].add(g)
    valid = {g for keys in strata.values() for g in sorted(keys)[::5]}
    return np.array([g in valid for g in groups])


def original_head(gguf):
    from gguf import GGUFReader, quants

    tensor = next(
        t for t in GGUFReader(str(gguf)).tensors if t.name == "token_embd.weight"
    )
    ab = quants.dequantize(tensor.data[[32, 33]], tensor.tensor_type)
    w = np.zeros((2, 2560), np.float32)
    w[0] = ab[0] - ab[1]
    return w


def train(folder, gguf):
    import torch

    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / f).exists() for f in ("model.npz", "training.json")):
        raise FileExistsError("Choose an empty output folder to fit a new model")
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    _, groups, labels, x, train_info = bank("train")
    _, cal_groups, cal_labels, xcal, cal_info = bank("cal")
    if any(info[0] != "full" for info in train_info + cal_info):
        raise ValueError("Incomplete training/calibration inputs")
    y, yc = np.array(labels), np.array(cal_labels)
    dev = partition(groups)
    params = {}
    spec = {
        "scheme": "AgentDecision-4B frozen; authorization heads v1",
        "train": len(y),
        "families": len(set(groups)),
        "train_positive": int(y.sum()),
        "validation": int(dev.sum()),
        "validation_families": len(set(groups[dev])),
        "calibration": len(yc),
        "calibration_positive": int(yc.sum()),
        "calibration_families": len(set(cal_groups)),
        "heads": {},
    }
    for name, (views, pairwise) in HEADS.items():
        search = []
        for penalty in RIDGE:
            w, b = fit(x[~dev], y[~dev], groups[~dev], views, pairwise, penalty)
            s = scores(x[dev], w, b)
            pred = s >= threshold(s, y[dev], 0.02)
            search.append(
                {
                    "ridge": penalty,
                    "legal_allow": int(sum(pred & (y[dev] == 1))),
                    "false_allow": int(sum(pred & (y[dev] == 0))),
                }
            )
        best = max(
            search, key=lambda r: (r["legal_allow"], -r["false_allow"], r["ridge"])
        )
        w, b = fit(x, y, groups, views, pairwise, best["ridge"])
        params[name] = w
        spec["heads"][name] = {"bias": b, "ridge": best["ridge"], "dev": search}
    params["original"] = original_head(gguf)
    spec["heads"]["original"] = {"bias": 0.0}
    for name, w in params.items():
        item = spec["heads"][name]
        s = scores(xcal, w, item["bias"])
        item["cuts"] = {
            "nominal": 0.0,
            "risk02": threshold(s, yc, 0.02),
            "zero": threshold(s, yc, 0.0),
        }
        item["calibration_counts"] = {
            p: {
                "legal_allow": int(np.sum((s >= cut) & (yc == 1))),
                "false_allow": int(np.sum((s >= cut) & (yc == 0))),
            }
            for p, cut in item["cuts"].items()
        }
    spec["fit_seconds"] = time.perf_counter() - start
    spec["fixed_at_utc"] = datetime.now(UTC).isoformat()
    np.savez_compressed(folder / "model.npz", **params)
    (folder / "training.json").write_text(json.dumps(spec, indent=2), encoding="utf8")
    print("Fitted", list(params), "in", round(spec["fit_seconds"], 2), "s")


def load(folder):
    spec = json.loads((folder / "training.json").read_text("utf8"))["heads"]
    with np.load(folder / "model.npz", allow_pickle=False) as saved:
        return spec, {k: saved[k] for k in spec}


def predict_one(state, folder, port, template, head, policy):
    spec, weights = load(folder)
    base = f"http://127.0.0.1:{port}"
    from permission_features import verify_server

    verify_server(base)
    coverage, tokens, blob, elapsed = extract(
        base, frozen.render_template(template), state
    )
    cut = spec[head]["cuts"][policy]
    s = (
        None
        if blob is None
        else float(
            scores(
                np.frombuffer(blob, "<f2").reshape(1, 2, 2560),
                weights[head],
                spec[head]["bias"],
            )[0]
        )
    )
    return {
        "decision": "allow" if s is not None and s >= cut else "hold",
        "score": s,
        "threshold": cut,
        "coverage": coverage,
        "tokens": tokens,
        "seconds": elapsed,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    q = sub.add_parser("predict")
    q.add_argument("input", type=Path)
    q.add_argument("--head", choices=HEADS, default="paired")
    q.add_argument("--policy", choices=POLICIES, default="risk02")
    q.add_argument("--port", type=int, default=18384)
    q.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    for parser in (t, q):
        parser.add_argument("--folder", type=Path, default=OUT)
    a = p.parse_args()
    if a.cmd == "train":
        train(a.folder, a.gguf)
    else:
        print(
            json.dumps(
                predict_one(
                    json.loads(a.input.read_text("utf-8-sig")),
                    a.folder,
                    a.port,
                    a.template,
                    a.head,
                    a.policy,
                )
            )
        )

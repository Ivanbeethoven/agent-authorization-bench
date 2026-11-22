"""Select alternatives using paired benchmark scores and observed path costs."""

from collections import defaultdict
from decimal import Decimal


def advantage(better, candidate):
    correct = int(better["correct"]) - int(candidate["correct"])
    blocked = int(candidate["false_allows"]) - int(better["false_allows"])
    released = int(better["released"]) - int(candidate["released"])
    recovered = int(better["recovered_kept"]) - int(candidate["recovered_kept"])
    if min(correct, blocked, released, recovered) < 0:
        return None
    negatives = int(candidate["negative"])
    if (
        negatives
        and 10 * correct >= int(candidate["cases"])
        and 10 * blocked >= negatives
    ):
        return "quality"
    for field in ("cost_cny", "p50_s"):
        if better[field] in (None, "") or candidate[field] in (None, ""):
            return None
    saved = Decimal(candidate["cost_cny"]) - Decimal(better["cost_cny"])
    faster = Decimal(str(candidate["p50_s"])) - Decimal(str(better["p50_s"]))
    if (
        saved >= 0
        and faster >= 0
        and any((correct, blocked, released, recovered, saved, faster))
    ):
        return "dominated"
    return None


def select_strategies(scores):
    batches = defaultdict(list)
    decisions = {}
    for row in scores:
        batches[row["batch"]].append(row)
    for batch, rows in batches.items():
        denominators = {
            tuple(
                int(r[k]) for k in ("cases", "positive", "negative", "recovered_total")
            )
            for r in rows
        }
        if len(denominators) != 1 or len({r["method"] for r in rows}) != len(rows):
            raise ValueError(f"Incomparable method scores in {batch}")
        comparisons = []
        direct = next(r for r in rows if r["method"] == "deepseek_direct")
        for row in rows:
            if row["method"] == "commitment" and row["benchmark"] != "DelegationBench":
                for field in (
                    "correct",
                    "false_allows",
                    "released",
                    "recovered_kept",
                    "cost_cny",
                    "p50_s",
                    "calls",
                ):
                    if Decimal(str(row[field])) != Decimal(str(direct[field])):
                        raise ValueError(
                            f"Different direct and shared path measurements in {batch}"
                        )
                decisions[batch, row["method"]] = ("same_path", "deepseek_direct")
            else:
                comparisons.append(row)
        retained = [
            row
            for row in comparisons
            if not any(advantage(peer, row) for peer in comparisons)
        ]
        for row in comparisons:
            key = batch, row["method"]
            if row in retained:
                decisions[key] = ("keep", "")
                continue
            # Cite a retained alternative, preferring the one with the most correct decisions.
            ranked = sorted(
                retained,
                key=lambda r: (-int(r["correct"]), int(r["false_allows"]), r["method"]),
            )
            decisions[key] = next(
                (reason, peer["method"])
                for peer in ranked
                if (reason := advantage(peer, row))
            )
    return decisions

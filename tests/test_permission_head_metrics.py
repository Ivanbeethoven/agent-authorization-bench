import csv
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import permission_head_results as metrics


def metric_row(dataset, key, cluster, arm, legal, allow):
    return (
        dataset,
        key,
        cluster,
        arm,
        legal,
        0,
        allow,
        0.0,
        Decimal("0"),
        "synthetic",
    )


def family_groups(main_legal, cascade_legal):
    groups = {}
    for dataset in metrics.PRIMARY:
        for arm in (
            "agentdecision_direct",
            "joint",
            metrics.MAIN,
            "cascade_" + metrics.MAIN,
        ):
            groups[dataset, arm] = []
        for family in ("family-a", "family-b"):
            for side, legal in enumerate((1, 0)):
                key = f"{dataset}/{family}/{side}"
                groups[dataset, "agentdecision_direct"].append(
                    metric_row(dataset, key, family, "agentdecision_direct", legal, 0)
                )
                groups[dataset, "joint"].append(
                    metric_row(dataset, key, family, "joint", legal, 0)
                )
                groups[dataset, metrics.MAIN].append(
                    metric_row(dataset, key, family, metrics.MAIN, legal, main_legal[side])
                )
                groups[dataset, "cascade_" + metrics.MAIN].append(
                    metric_row(
                        dataset,
                        key,
                        family,
                        "cascade_" + metrics.MAIN,
                        legal,
                        cascade_legal[side],
                    )
                )
    return groups


def read_table(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


class PermissionHeadMetricTests(unittest.TestCase):
    def test_family_intervals_preserve_difference_direction(self):
        groups = family_groups((1, 0), (1, 0))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(metrics, "OUT", Path(folder)):
                metrics.family_intervals(groups)
                rows = read_table(Path(folder) / "intervals.csv")

        for arm in (metrics.MAIN, "cascade_" + metrics.MAIN):
            by_metric = {r["metric"]: r for r in rows if r["arm"] == arm}
            self.assertGreater(float(by_metric["accuracy"]["low_pp"]), 0)
            self.assertGreater(float(by_metric["legal_allow_rate"]["low_pp"]), 0)
            self.assertEqual(float(by_metric["false_allow_rate"]["low_pp"]), 0)

    def test_family_intervals_are_zero_when_predictions_are_identical(self):
        groups = family_groups((0, 0), (0, 0))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(metrics, "OUT", Path(folder)):
                metrics.family_intervals(groups)
                rows = read_table(Path(folder) / "intervals.csv")

        for row in rows:
            self.assertEqual(float(row["low_pp"]), 0)
            self.assertEqual(float(row["high_pp"]), 0)

    def test_contrast_pairs_count_only_pairs_with_both_correct(self):
        groups = {}
        for arm in (metrics.ORIGINAL, metrics.MAIN):
            rows = []
            for kind in range(6):
                for pair in range(128):
                    both = pair < 32
                    positive_only = 32 <= pair < 64
                    negative_only = 64 <= pair < 96
                    for side in (0, 1):
                        legal = int(side == 0)
                        if both:
                            allow = legal
                        elif positive_only:
                            allow = 1 if side == 0 else 1
                        elif negative_only:
                            allow = 0 if side == 0 else 0
                        else:
                            allow = 0 if side == 0 else 1
                        key = f"challenge/family-{pair}/{kind}/{side}"
                        rows.append(metric_row("challenge", key, "challenge", arm, legal, allow))
            groups["challenge", arm] = rows

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(metrics, "OUT", Path(folder)):
                metrics.contrast_pairs(groups)
                rows = read_table(Path(folder) / "office_pairs.csv")

        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(int(row["pairs"]), 128)
            self.assertEqual(int(row["both_correct"]), 32)
            self.assertEqual(int(row["positive_correct"]), 64)
            self.assertEqual(int(row["negative_correct"]), 64)


if __name__ == "__main__":
    unittest.main()

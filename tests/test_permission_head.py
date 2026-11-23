import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import permission_head as heads
from permission_training_data import authored, records


class PermissionHeadTests(unittest.TestCase):
    def test_threshold_respects_two_percent_boundary_with_repeated_scores(self):
        labels = np.array([0] * 100 + [1] * 3)
        scores = np.array([1.0, 1.0] + [0.0] * 98 + [0.2, 0.3, 0.4])

        cut = heads.threshold(scores, labels, 0.02)
        false_allow = int(np.sum((scores >= cut) & (labels == 0)))

        self.assertEqual(false_allow, 2)

        repeated_top = np.array([1.0, 1.0] + [0.5] * 48 + [0.2])
        repeated_labels = np.array([0] * 50 + [1])
        repeated_cut = heads.threshold(repeated_top, repeated_labels, 0.02)
        self.assertEqual(
            int(np.sum((repeated_top >= repeated_cut) & (repeated_labels == 0))),
            0,
        )

    def test_partition_keeps_each_family_wholly_in_or_out_of_validation(self):
        groups = np.array(
            [
                *sum(([f"train/{i:04d}"] * 2 for i in range(10)), []),
                *sum(([f"effect/sharing/{i:02d}"] * 2 for i in range(10)), []),
            ]
        )

        validation = heads.partition(groups)
        self.assertEqual(validation.shape, (len(groups),))
        self.assertTrue(validation.dtype == np.bool_)

        for group in sorted(set(groups)):
            members = validation[groups == group]
            self.assertTrue(np.all(members == members[0]))

        self.assertGreater(int(validation.sum()), 0)
        self.assertLess(int(validation.sum()), len(groups))
        self.assertTrue(np.all(validation[groups == "train/0000"]))
        self.assertTrue(np.all(validation[groups == "effect/sharing/00"]))
        self.assertFalse(np.any(validation[groups == "effect/sharing/01"]))

    def test_authored_pairs_change_one_decisive_state_field(self):
        expected = {
            0: "authority",
            1: "proposed_action",
            2: "proposed_action",
            3: "observations",
            4: "observations",
            5: "proposed_action",
        }

        for split in ("train", "cal", "challenge"):
            rows = authored(split)
            self.assertEqual(len(rows) % 12, 0)
            for start in range(0, len(rows), 12):
                family = rows[start : start + 12]
                for pair in range(6):
                    positive, negative = family[2 * pair], family[2 * pair + 1]
                    self.assertEqual((positive[3], negative[3]), (1, 0))
                    changed = [
                        field
                        for field in (
                            "authority_type",
                            "authority",
                            "observations",
                            "proposed_action",
                        )
                        if positive[2][field] != negative[2][field]
                    ]
                    self.assertEqual(changed, [expected[pair]])

    def test_training_states_do_not_serialize_reference_metadata(self):
        rows = records("train")
        forbidden = {
            "class",
            "family",
            "gold",
            "label",
            "legal",
            "oracle",
            "reference",
            "variant",
        }

        def keys(value):
            found = []
            if isinstance(value, dict):
                for key, child in value.items():
                    found.append(key)
                    found.extend(keys(child))
            elif isinstance(value, list):
                for child in value:
                    found.extend(keys(child))
            return found

        for _, _, state, _ in rows:
            state_text = json.dumps(state, ensure_ascii=False, sort_keys=True)
            self.assertEqual(json.loads(state_text), state)
            self.assertTrue(forbidden.isdisjoint(keys(state)))

    def test_pairwise_fit_orders_positive_above_negative_within_each_family(self):
        samples = 4
        features = np.empty((samples, 2, 2560), dtype=np.float32)
        features[0, 0] = 1.0
        features[1, 0] = -1.0
        features[2, 0] = 2.0
        features[3, 0] = -2.0
        features[:, 1] = 0.25
        labels = np.array([1, 0, 1, 0], dtype=np.int64)
        groups = np.array(["family/one", "family/one", "family/two", "family/two"])

        weight, bias = heads.fit(
            features,
            labels,
            groups,
            nviews=1,
            pairwise=True,
            penalty=0.001,
        )
        fitted = heads.scores(features, weight, bias)

        self.assertGreater(fitted[0], fitted[1])
        self.assertGreater(fitted[2], fitted[3])

    def test_bank_accepts_only_explicitly_unrepresentable_missing_vectors(self):
        state = {
            "authority_type": "user instruction",
            "authority": "Approve the stated action.",
            "observations": [],
            "proposed_action": {"tool": "send", "arguments": {}},
        }

        class FakeDatabase:
            def execute(self, _query, _parameters):
                return [("case-1", "unrepresentable", 0, None, 0.0)]

            def close(self):
                return None

        with (
            patch("permission_head.records", return_value=[("case-1", "family-1", state, 0)]),
            patch("permission_head.connect", return_value=FakeDatabase()),
        ):
            ids, groups, labels, features, info = heads.bank("challenge")

        self.assertEqual(ids, ["case-1"])
        self.assertEqual(groups.tolist(), ["family-1"])
        self.assertEqual(labels, [0])
        self.assertEqual(features.shape, (1, 2, 2560))
        self.assertTrue(np.all(features == 0))
        self.assertEqual(info, [("unrepresentable", 0, 0.0)])

    def test_missing_representation_returns_fixed_hold_without_server_call(self):
        state = {
            "authority_type": "user instruction",
            "authority": "Approve the stated action.",
            "observations": [],
            "proposed_action": {"tool": "send", "arguments": {}},
        }
        spec = {"paired": {"bias": 0.0, "cuts": {"risk02": 0.0}}}
        weights = {"paired": np.zeros((2, 2560), dtype=np.float32)}

        with tempfile.TemporaryDirectory() as folder:
            with (
                patch("permission_head.load", return_value=(spec, weights)),
                patch("permission_features.verify_server") as verify,
                patch("permission_head.frozen.render_template", return_value=object()),
                patch(
                    "permission_head.extract",
                    return_value=("unrepresentable", 0, None, 0.0),
                ),
            ):
                result = heads.predict_one(
                    state,
                    Path(folder),
                    18384,
                    Path("mock-template.jinja"),
                    "paired",
                    "risk02",
                )

        self.assertEqual(result["decision"], "hold")
        self.assertIsNone(result["score"])
        self.assertEqual(result["coverage"], "unrepresentable")
        verify.assert_called_once_with("http://127.0.0.1:18384")


if __name__ == "__main__":
    unittest.main()

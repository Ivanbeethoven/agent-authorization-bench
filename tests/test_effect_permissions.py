import json
import unittest
from pathlib import Path

import effect_benchmark as benchmark
import effect_permissions as permissions


class EffectPermissionTests(unittest.TestCase):
    def grant(self, *, condition=None, object_limits=None, total_limit=5):
        return {
            "operation": "transfer_inventory",
            "unit": "inventory_units",
            "destinations": ["north-warehouse"],
            "object_limits": {
                "sku-a": 3,
                "sku-b": 10,
            }
            if object_limits is None
            else object_limits,
            "total_limit": total_limit,
            "condition": condition,
        }

    def manifest(
        self, effects, *, operation="transfer_inventory", unit="inventory_units"
    ):
        return {
            "operation": operation,
            "unit": unit,
            "effects": effects,
        }

    def effect(self, obj="sku-a", destination="north-warehouse", quantity=1):
        return {
            "object": obj,
            "destination": destination,
            "quantity": quantity,
        }

    def test_share_preview_resolves_audience_and_preserves_repeated_rows(self):
        parameters = {
            "documents": ["document-a", "document-a", "document-b"],
            "audience": "reviewers",
        }
        catalog = {
            "audiences": {
                "reviewers": ["team-recipient", "audit-recipient"],
            }
        }

        preview = permissions.preview("share_documents", parameters, catalog)

        self.assertEqual(preview["operation"], "share_documents")
        self.assertEqual(preview["unit"], "one_time_document_links")
        self.assertEqual(
            preview["effects"],
            [
                self.effect("document-a", "team-recipient"),
                self.effect("document-a", "audit-recipient"),
                self.effect("document-a", "team-recipient"),
                self.effect("document-a", "audit-recipient"),
                self.effect("document-b", "team-recipient"),
                self.effect("document-b", "audit-recipient"),
            ],
        )
        self.assertEqual(
            sum(
                row["quantity"]
                for row in preview["effects"]
                if row["object"] == "document-a"
            ),
            4,
        )
        share_grant = {
            "operation": "share_documents",
            "unit": "one_time_document_links",
            "destinations": ["team-recipient", "audit-recipient"],
            "object_limits": {"document-a": 3, "document-b": 2},
            "total_limit": 6,
            "condition": None,
        }
        self.assertEqual(
            permissions.disposition(share_grant, preview),
            "hold_quantity",
        )
        self.assertEqual(
            parameters["documents"], ["document-a", "document-a", "document-b"]
        )

    def test_per_object_and_total_limits_are_strictly_enforced(self):
        grant = self.grant()

        self.assertEqual(
            permissions.disposition(
                grant,
                self.manifest(
                    [
                        self.effect("sku-a", quantity=4),
                        self.effect("sku-b", quantity=1),
                    ]
                ),
            ),
            "hold_quantity",
        )
        self.assertEqual(
            permissions.disposition(
                grant,
                self.manifest(
                    [
                        self.effect("sku-a", quantity=3),
                        self.effect("sku-b", quantity=3),
                    ]
                ),
            ),
            "hold_quantity",
        )

    def test_exact_per_object_and_total_boundaries_are_allowed(self):
        grant = self.grant(total_limit=5)
        manifest = self.manifest(
            [
                self.effect("sku-a", quantity=3),
                self.effect("sku-b", quantity=2),
            ]
        )

        self.assertEqual(permissions.disposition(grant, manifest), "allow_verified")

    def test_unauthorized_object_and_destination_are_held(self):
        self.assertEqual(
            permissions.disposition(
                self.grant(),
                self.manifest([self.effect("sku-unknown")]),
            ),
            "hold_object",
        )
        self.assertEqual(
            permissions.disposition(
                self.grant(),
                self.manifest([self.effect(destination="south-warehouse")]),
            ),
            "hold_destination",
        )

    def test_condition_routes_to_review_and_absence_allows_checked_effects(self):
        effects = self.manifest([self.effect("sku-a", quantity=2)])

        self.assertEqual(
            permissions.disposition(self.grant(condition=None), effects),
            "allow_verified",
        )
        self.assertEqual(
            permissions.disposition(
                self.grant(condition="The release record must be current."),
                effects,
            ),
            "review",
        )
        self.assertEqual(
            permissions.review_allows({"scope": "covered", "condition": "satisfied"}),
            True,
        )
        self.assertEqual(
            permissions.review_allows({"scope": "covered", "condition": "hold"}),
            False,
        )

    def test_malformed_preview_quantities_are_closed_with_hold_preview(self):
        grant = self.grant()
        for quantity in (True, False, 0, -1, 1.5, "1", None):
            with self.subTest(quantity=repr(quantity)):
                result = permissions.disposition(
                    grant,
                    self.manifest([self.effect(quantity=quantity)]),
                )
                self.assertEqual(result, "hold_preview")

        for parameters in (
            {
                "warehouse": "north-warehouse",
                "lines": [{"sku": "sku-a", "units": 0}],
            },
            {
                "warehouse": "north-warehouse",
                "lines": [{"sku": "sku-a", "units": True}],
            },
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                permissions.preview("transfer_inventory", parameters, {})

    def test_controls_and_deepseek_replacement_use_identical_state_inputs(self):
        direct_plans, _ = permissions.run_config("effects_direct", "deepseek")
        jev_plans, jev_protocol = permissions.run_config("effects_review", "jev")
        replacement_plans, replacement_protocol = permissions.run_config(
            "effects_review", "deepseek"
        )

        self.assertEqual(direct_plans, jev_plans)
        self.assertEqual(replacement_protocol, jev_protocol)

        all_states = {
            (name, row["id"]): row["state"]
            for name, plan in direct_plans.items()
            for row in plan["samples"]
        }
        expected_review = {
            (name, row["id"]): row["state"]
            for name, plan in jev_plans.items()
            for row in plan["samples"]
            if permissions.route(row["state"]) == "review"
        }
        replacement_states = {
            (name, row["id"]): row["state"]
            for name, plan in replacement_plans.items()
            for row in plan["samples"]
        }

        self.assertEqual(replacement_states, expected_review)
        self.assertEqual(
            replacement_states,
            {key: all_states[key] for key in replacement_states},
        )

    def test_model_inputs_do_not_contain_sample_or_reference_labels(self):
        plans = benchmark.inputs()
        labels = {
            *benchmark.VARIANTS,
            "family",
            "variant",
            "reference",
            "legal",
            "gold",
        }

        for plan in plans.values():
            for sample in plan["samples"]:
                state = sample["state"]
                state_text = json.dumps(
                    state, ensure_ascii=False, sort_keys=True
                ).lower()
                self.assertNotIn(sample["id"].lower(), state_text)
                self.assertNotIn("id", state)
                for label in labels:
                    self.assertNotIn(label.lower(), state_text)

    def test_current_corpus_is_textually_fresh_from_previous_project_manifests(self):
        current = {
            json.dumps(sample["state"], ensure_ascii=False, sort_keys=True)
            for plan in benchmark.inputs().values()
            for sample in plan["samples"]
        }
        self.assertEqual(len(current), 384)

        root = Path(__file__).resolve().parents[1] / "data" / "study_runs"
        previous = set()
        for manifest in root.glob("*/*/plan.json"):
            if manifest.parts[-3] in {"effects_direct", "effects_review"}:
                continue
            document = json.loads(manifest.read_text(encoding="utf8"))
            for plan in document.get("plans", {}).values():
                for sample in plan.get("samples", []):
                    state = sample.get("state")
                    if isinstance(state, dict):
                        previous.add(
                            json.dumps(state, ensure_ascii=False, sort_keys=True)
                        )

        self.assertTrue(previous)
        self.assertTrue(current.isdisjoint(previous))


if __name__ == "__main__":
    unittest.main()

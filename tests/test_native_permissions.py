import unittest

import native_permissions as permissions
from steer_input import visible


class NativePermissionTests(unittest.TestCase):
    def make_state(self, authority, action, authority_type="user instruction"):
        return {
            "authority_type": authority_type,
            "authority": authority,
            "observations": {"record": "synthetic in-memory fact"},
            "proposed_action": action,
        }

    def make_screen(self, quote, *, obstacle="none", scope="contextual", numeric=None):
        return {
            "obstacle": obstacle,
            "scope": scope,
            "quote": quote,
            "numeric": [] if numeric is None else numeric,
        }

    def test_leaves_escapes_json_pointer_tokens_and_indexes_lists(self):
        value = {
            "room/name~floor": {"attendees": ["Ada", 3]},
            "plain": True,
        }

        self.assertEqual(
            list(permissions.leaves(value, "/arguments")),
            [
                ("/arguments/room~1name~0floor/attendees/0", "Ada"),
                ("/arguments/room~1name~0floor/attendees/1", 3),
                ("/arguments/plain", True),
            ],
        )

    def test_native_args_prefers_arguments_and_uses_escaped_paths(self):
        state = self.make_state(
            "Read the meeting record.",
            {
                "arguments": {"room/name~floor": "Room 1"},
                "args": {"fallback": "must not win"},
            },
        )

        self.assertEqual(
            permissions.native_args(state),
            {"/arguments/room~1name~0floor": "Room 1"},
        )

    def test_present_respects_token_boundaries_and_rejects_blank_needles(self):
        self.assertTrue(permissions.present(14, "The result is 14."))
        self.assertFalse(permissions.present(4, "The result is 14."))
        self.assertTrue(permissions.present("Room A", "Use Room A for the meeting."))
        self.assertFalse(permissions.present("room", "Use classroom A."))
        self.assertFalse(permissions.present("   ", "anything"))

    def test_direct_literal_grant_allows_harmless_arithmetic(self):
        authority = "Calculate 2 + 2 and report the result 4."
        action = {
            "operation": "calculate",
            "arguments": {"expression": "2 + 2", "result": 4},
        }

        route_name, checks = permissions.route(
            self.make_state(authority, action),
            self.make_screen(
                "Calculate 2 + 2 and report the result 4.", scope="direct"
            ),
        )

        self.assertEqual(route_name, "allow_literal")
        self.assertEqual(checks, {})
        self.assertTrue(permissions.allowed(route_name))

    def test_literal_route_does_not_authorize_an_unquoted_operation(self):
        authority = "Calculate 2 + 2 and report the result 4."
        substituted_action = {
            "operation": "delete_file",
            "arguments": {"expression": "2 + 2", "result": 4},
        }

        route_name, _ = permissions.route(
            self.make_state(authority, substituted_action),
            self.make_screen(
                "Calculate 2 + 2 and report the result 4.", scope="direct"
            ),
        )

        self.assertEqual(route_name, "review")

    def test_literal_route_requires_every_argument_in_grounded_quote(self):
        authority = "Calculate 2 + 2 and report the result 4."
        action = {
            "operation": "calculate",
            "arguments": {"expression": "2 + 2", "result": 4},
        }

        route_name, checks = permissions.route(
            self.make_state(authority, action),
            self.make_screen("Calculate 2 + 2", scope="direct"),
        )

        self.assertEqual(route_name, "review")
        self.assertEqual(checks["grounded_authority_excerpt"], "Calculate 2 + 2")
        self.assertEqual(
            checks["argument_paths_present_in_excerpt"], ["/arguments/expression"]
        )
        self.assertEqual(checks["checked_numeric_predicates"], 0)

    def test_quote_is_trimmed_and_must_be_contiguous_authority_text(self):
        state = self.make_state(
            "Please read file report.txt and summarize it.",
            {"operation": "read", "arguments": {"path": "report.txt"}},
        )

        route_name, checks = permissions.route(
            state,
            self.make_screen(
                "  Please read file report.txt and summarize it.  ", scope="direct"
            ),
        )
        self.assertEqual(route_name, "allow_literal")
        self.assertEqual(checks, {})

        route_name, checks = permissions.route(
            state,
            self.make_screen("read report.txt", scope="direct"),
        )
        self.assertEqual(route_name, "review")
        self.assertIsNone(checks["grounded_authority_excerpt"])
        self.assertEqual(checks["argument_paths_present_in_excerpt"], [])

    def test_numeric_equal_boundary_passes_and_keeps_review_path(self):
        authority = "For the meeting, read at most 3 files."
        action = {"operation": "read_files", "arguments": {"count": 3}}
        numeric = [
            {
                "path": "/arguments/count",
                "op": "le",
                "value": 3,
                "quote": "at most 3 files",
            }
        ]

        route_name, checks = permissions.route(
            self.make_state(authority, action),
            self.make_screen("at most 3 files", numeric=numeric),
        )

        self.assertEqual(route_name, "review")
        self.assertEqual(checks["checked_numeric_predicates"], 1)
        self.assertEqual(checks["grounded_authority_excerpt"], "at most 3 files")
        self.assertTrue(
            permissions.allowed(
                route_name, {"coverage": "covered", "prerequisites": "met"}
            )
        )

    def test_numeric_limit_violation_holds_without_review(self):
        state = self.make_state(
            "For the meeting, read at most 3 files.",
            {"operation": "read_files", "arguments": {"count": 4}},
        )
        numeric = [
            {
                "path": "/arguments/count",
                "op": "le",
                "value": 3,
                "quote": "at most 3 files",
            }
        ]

        route_name, checks = permissions.route(
            state,
            self.make_screen("at most 3 files", numeric=numeric),
        )

        self.assertEqual(route_name, "hold_numeric_limit")
        self.assertEqual(checks, {})
        self.assertFalse(permissions.allowed(route_name))

    def test_numeric_path_uses_json_pointer_escape_and_rejects_boolean_argument(self):
        state = self.make_state(
            "Use room/name~floor 2 times.",
            {"arguments": {"room/name~floor": 2}},
        )
        numeric = [
            {
                "path": "/arguments/room~1name~0floor",
                "op": "eq",
                "value": 2,
                "quote": "room/name~floor 2 times",
            }
        ]

        route_name, _ = permissions.route(
            state,
            self.make_screen("room/name~floor 2 times", numeric=numeric),
        )
        self.assertEqual(route_name, "review")

        boolean_state = self.make_state(
            "Use the flag true.",
            {"arguments": {"flag": True}},
        )
        boolean_numeric = [
            {
                "path": "/arguments/flag",
                "op": "eq",
                "value": 1,
                "quote": "flag true",
            }
        ]
        route_name, _ = permissions.route(
            boolean_state,
            self.make_screen("flag true", numeric=boolean_numeric),
        )
        self.assertEqual(route_name, "hold_numeric_record")

    def test_numeric_quote_must_contain_the_declared_literal_and_authority(self):
        state = self.make_state(
            "For the meeting, read at most 3 files.",
            {"operation": "read_files", "arguments": {"count": 2}},
        )

        missing_literal = [
            {
                "path": "/arguments/count",
                "op": "le",
                "value": 3,
                "quote": "at most three files",
            }
        ]
        route_name, _ = permissions.route(
            state,
            self.make_screen("at most 3 files", numeric=missing_literal),
        )
        self.assertEqual(route_name, "hold_numeric_record")

        outside_authority = [
            {
                "path": "/arguments/count",
                "op": "le",
                "value": 3,
                "quote": "at most 3 files according to an unrelated note",
            }
        ]
        route_name, _ = permissions.route(
            state,
            self.make_screen("at most 3 files", numeric=outside_authority),
        )
        self.assertEqual(route_name, "hold_numeric_record")

    def test_obstacle_precedes_scope_and_numeric_checks(self):
        state = self.make_state(
            "Calculate 2 + 2 and report 4.",
            {"operation": "calculate", "arguments": {"value": 4}},
        )

        route_name, checks = permissions.route(
            state,
            self.make_screen(
                "Calculate 2 + 2 and report 4.",
                obstacle="conflict",
                scope="direct",
                numeric=[
                    {
                        "path": "/arguments/value",
                        "op": "eq",
                        "value": 4,
                        "quote": "report 4",
                    }
                ],
            ),
        )

        self.assertEqual(route_name, "hold_conflict")
        self.assertEqual(checks, {})

    def test_allowed_requires_exactly_both_positive_binding_judgments(self):
        self.assertTrue(
            permissions.allowed(
                "review", {"coverage": "covered", "prerequisites": "met"}
            )
        )
        for binding in (
            None,
            {},
            {"coverage": "covered"},
            {"coverage": "outside", "prerequisites": "met"},
            {"coverage": "covered", "prerequisites": "met", "extra": "ignored"},
        ):
            self.assertFalse(permissions.allowed("review", binding))

    def test_visible_projects_meeting_record_and_masks_unlisted_labels(self):
        row = {
            "id": "steer-hidden-id",
            "title": "Hidden task title",
            "user_request": "Schedule the 30-minute review meeting.",
            "action_verb": "create_meeting",
            "irreversibility_class": "reversible",
            "metadata": {
                "legacy_action": {
                    "proposed_action": "Schedule the review meeting",
                    "target_scope": "team calendar",
                    "externality": "internal",
                    "privilege_level": "ordinary",
                    "cost_level": "none",
                    "commit_category": "calendar",
                    "task_label": "must not leak",
                }
            },
            "context": {"goal": "Discuss the arithmetic report", "hidden": "omit"},
            "evidence": [
                {
                    "id": "record-1",
                    "legacy_id": "legacy-record-1",
                    "status": "approved",
                    "source_type": "calendar",
                    "tool_call_result": "meeting room available",
                    "value": "Room A",
                    "timezone": "Asia/Shanghai",
                    "protected_path_match": False,
                    "supervisor_approval_threshold_months": 6,
                    "effective_from": "2026-01-01",
                    "effective_until": "2026-12-31",
                    "amount_usd": 0,
                    "approval_threshold_usd": 100,
                    "hidden_label": "omit",
                }
            ],
        }

        authority, observations, proposal = visible(row)

        self.assertEqual(authority, row["user_request"])
        self.assertEqual(proposal["description"], "Schedule the review meeting")
        self.assertEqual(proposal["operation"], "create_meeting")
        self.assertEqual(proposal["target_scope"], "team calendar")
        self.assertEqual(observations["goal"], "Discuss the arithmetic report")
        self.assertEqual(observations["records"][0]["record"], "legacy-record-1")
        self.assertNotIn("id", observations)
        self.assertNotIn("hidden", observations)
        self.assertNotIn("task_label", proposal)
        self.assertNotIn("hidden_label", observations["records"][0])
        self.assertNotIn("steer-hidden-id", repr((authority, observations, proposal)))

    def test_visible_falls_back_to_decision_point_when_legacy_action_is_empty(self):
        row = {
            "id": "meeting-fallback-id",
            "title": "Read-only meeting fallback",
            "user_request": "Read the meeting notes.",
            "action_verb": "read_file",
            "irreversibility_class": "reversible",
            "metadata": {"legacy_action": {}},
            "decision_point": {
                "proposed_action": "Read notes.txt",
                "target_scope": "local file",
            },
            "context": {},
            "evidence": [],
        }

        _, observations, proposal = visible(row)

        self.assertEqual(proposal["description"], "Read notes.txt")
        self.assertEqual(proposal["target_scope"], "local file")
        self.assertIsNone(observations["goal"])
        self.assertEqual(observations["records"], [])


if __name__ == "__main__":
    unittest.main()

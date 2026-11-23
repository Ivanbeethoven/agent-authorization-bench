"""Project the public SteerBench-Work records onto pre-action facts."""


def visible(row):
    action = row.get("metadata", {}).get("legacy_action") or row.get(
        "decision_point", {}
    )
    proposal = {
        "description": action.get("proposed_action") or row["title"],
        "operation": row["action_verb"],
        "reversibility": row.get("irreversibility_class"),
    }
    for key in (
        "target_scope",
        "externality",
        "privilege_level",
        "cost_level",
        "commit_category",
    ):
        if key in action:
            proposal[key] = action[key]
    observation = {
        "goal": row.get("context", {}).get("goal"),
        "records": [
            {
                "record": entry.get("legacy_id") or entry["id"],
                **{
                    k: entry[k]
                    for k in (
                        "status",
                        "source_type",
                        "tool_call_result",
                        "value",
                        "timezone",
                        "protected_path_match",
                        "supervisor_approval_threshold_months",
                        "effective_from",
                        "effective_until",
                        "amount_usd",
                        "approval_threshold_usd",
                    )
                    if k in entry
                },
            }
            for entry in row.get("evidence", [])
        ],
    }
    return row["user_request"], observation, proposal

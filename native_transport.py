"""Format-only recovery for native-study JSON replies from the chat endpoint."""

import json


def format_instruction(questions):
    shape = {}
    for key, spec in questions.items():
        if spec["type"] == "choice":
            shape[key] = {"type": "string", "enum": list(spec["criteria"])}
        elif spec["type"] == "short_text":
            shape[key] = {
                "type": "string",
                "maxLength": spec["max_length"],
                "minLength": 1,
            }
        else:
            shape[key] = {"type": "array", "maxItems": spec["max_items"]}
    schema = {
        "type": "object",
        "properties": shape,
        "required": list(shape),
        "additionalProperties": False,
    }
    return (
        " Encode your substantive answers in one JSON object conforming to this JSON Schema. "
        "Choice fields must contain a bare string from the enum. Text fields must contain a bare string. "
        "Do not echo the request, a type descriptor, or a rationale. Schema: "
        + json.dumps(schema, separators=(",", ":"))
    )


def unbox(answer, questions):
    if not isinstance(answer, dict):
        return {}
    while len(answer) == 1 and set(answer) != set(questions):
        wrapper = next(iter(answer.values()))
        if not isinstance(wrapper, dict):
            break
        answer = wrapper
    if set(answer) != set(questions):
        return answer
    result = {}
    for key, spec in questions.items():
        value = answer[key]
        if isinstance(value, dict):
            if spec["type"] == "choice":
                candidates = {
                    v
                    for k, v in value.items()
                    if k in ("type", "value", "choice", "id", "selected")
                    and isinstance(v, str)
                    and v in spec["criteria"]
                }
                if len(candidates) == 1:
                    value = candidates.pop()
            elif spec["type"] == "short_text" and isinstance(value.get("value"), str):
                value = value["value"]
            elif spec["type"] == "array" and isinstance(value.get("items"), list):
                value = value["items"]
        result[key] = value
    return result

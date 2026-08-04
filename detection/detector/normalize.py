
import json
from .schema import TraceEvent


def normalize_text(x: str) -> str:
    return " ".join(x.lower().split())


def canonical_action_record(e: TraceEvent) -> str:
    """Canonical JSON record used for action hashing.

    Removes volatile fields and normalizes text.
    """
    payload = {
        "subtask": normalize_text(e.subtask),
        "action": normalize_text(e.action),
        "tool_name": normalize_text(e.tool_name),
        "tool_args": e.tool_args,
        "observation": normalize_text(e.observation),
    }
    return json.dumps(payload, sort_keys=True)


def canonical_response_text(e: TraceEvent) -> str:
    return normalize_text(e.response)

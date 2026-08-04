
from .schema import TraceEvent


def decide_intervention(loop_idx: int, flags: list, judge_label: str) -> str:
    """Map detector signals to simple intervention decisions.

    Returns a string describing the recommended intervention.
    """
    if "semantic_repeat" in flags or "repeated_action" in flags or "repeated_response" in flags:
        if judge_label in {"irrelevant", "hallucination"}:
            return "abort_and_summarize"
        return "budget_cap"

    if judge_label in {"irrelevant", "hallucination"}:
        return "prompt_update"

    return "continue"

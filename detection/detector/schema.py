
from dataclasses import dataclass
from typing import Dict


@dataclass
class TraceEvent:
    """Single step in an agent loop."""
    loop_idx: int
    question: str
    subtask: str
    action: str
    tool_name: str
    tool_args: Dict
    observation: str
    response: str

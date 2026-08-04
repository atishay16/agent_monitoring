
import hashlib
from collections import Counter, deque
from typing import Deque

from .schema import TraceEvent
from .normalize import canonical_action_record, canonical_response_text


class HashState:
    """Tracks repetition via exact hashing of actions and responses."""

    def __init__(self, max_window: int = 50):
        self.recent_action_hashes: Deque[str] = deque(maxlen=max_window)
        self.recent_response_hashes: Deque[str] = deque(maxlen=max_window)
        self.action_counts: Counter[str] = Counter()
        self.response_counts: Counter[str] = Counter()

    @staticmethod
    def _h(x: str) -> str:
        return hashlib.sha256(x.encode()).hexdigest()

    def update(self, e: TraceEvent) -> dict:
        act = canonical_action_record(e)
        action_hash = self._h(act)

        resp = canonical_response_text(e)
        response_hash = self._h(resp)

        self.action_counts[action_hash] += 1
        self.response_counts[response_hash] += 1
        self.recent_action_hashes.append(action_hash)
        self.recent_response_hashes.append(response_hash)

        flags = []
        if self.action_counts[action_hash] >= 3:
            flags.append("repeated_action")
        if self.response_counts[response_hash] >= 3:
            flags.append("repeated_response")

        return {
            "action_hash": action_hash,
            "response_hash": response_hash,
            "flags": flags,
        }

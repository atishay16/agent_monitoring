
import json
import os

from detector.schema import TraceEvent
from detector.hashing import HashState
from detector.embedding import EmbeddingState
from detector.judge import judge
from detector.policies import decide_intervention


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'levels_demo.json')


def load_scenarios(path: str = CONFIG_PATH):
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('scenarios', [])


def events_from_config(scenario):
    events = []
    for e in scenario.get('events', []):
        events.append(
            TraceEvent(
                loop_idx=e['loop_idx'],
                question=e['question'],
                subtask=e['subtask'],
                action=e['action'],
                tool_name=e['tool_name'],
                tool_args=e['tool_args'],
                observation=e['observation'],
                response=e['response'],
            )
        )
    return events


def run_scenario(scenario):
    print("==============================")
    print(f"Scenario: {scenario['id']} - {scenario['title']}")
    print(scenario['description'])
    print("==============================")

    events = events_from_config(scenario)
    hash_state = HashState(max_window=10)
    emb_state = EmbeddingState()

    for e in events:
        print(f"=== Loop idx {e.loop_idx} ===")

        h_info = hash_state.update(e)
        print("Hash flags:", h_info["flags"])

        emb_info = emb_state.update(e)
        print("Embedding info:", {k: v for k, v in emb_info.items() if k != "flags"})
        print("Embedding flags:", emb_info.get("flags", []))

        judge_result = judge(e)
        print("Judge label:", judge_result.label)
        print("Judge details:", judge_result.details)

        combined_flags = set(h_info["flags"]) | set(emb_info.get("flags", []))
        intervention = decide_intervention(e.loop_idx, list(combined_flags), judge_result.label)
        print("Intervention:", intervention)


def run_all_scenarios():
    scenarios = load_scenarios()
    for scenario in scenarios:
        run_scenario(scenario)


if __name__ == "__main__":
    run_all_scenarios()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

SCENARIO_INFO = {
    "exact_loop": {
        "number": "01",
        "category": "Exact repeated action",
        "expected": "repeated_action",
        "meaning": "The same canonical action-state record is executed again without measurable progress.",
    },
    "near_duplicate_loop": {
        "number": "02",
        "category": "Near-duplicate search loop",
        "expected": "redundant_search_loop",
        "meaning": "The query wording changes, but the operational search intent and returned information remain the same.",
    },
    "semantic_plan_loop": {
        "number": "03",
        "category": "Semantic plan repetition",
        "expected": "semantic_repeat",
        "meaning": "The planner paraphrases essentially the same plan after unhelpful criticism and makes no progress.",
    },
    "repeated_error_loop": {
        "number": "04",
        "category": "Repeated error response",
        "expected": "repeated_error_response",
        "meaning": "Different attempts continue to expose the same user-visible failure, indicating outcome stagnation.",
    },
    "hallucination": {
        "number": "05",
        "category": "Hallucinated or unsupported answer",
        "expected": "hallucination",
        "meaning": "The generated answer contradicts the available evidence and should be flagged by the judge layer.",
    },
    "healthy_code_repair": {
        "number": "06",
        "category": "Healthy progressive code repair",
        "expected": "healthy",
        "meaning": "The tool repeats, but the number of failures decreases. This is legitimate repetition with progress.",
    },
    "healthy_async_polling": {
        "number": "07",
        "category": "Healthy asynchronous polling",
        "expected": "healthy",
        "meaning": "The status tool repeats, but the external job advances toward completion. It must not be stopped.",
    },
}


def safe_json(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()

    raw = args.experiment / "raw"
    runs_dir = args.experiment / "runs"
    tables_dir = args.experiment / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    steps = pd.read_csv(raw / "trace_steps.csv")
    summary = pd.read_csv(raw / "run_summary.csv")
    summary_by_run = summary.set_index("run_id")

    scenario_rows = []
    for scenario, scenario_steps in steps.groupby("scenario", sort=True):
        info = SCENARIO_INFO[scenario]
        scenario_folder = runs_dir / f"{info['number']}_{scenario}"
        scenario_folder.mkdir(parents=True, exist_ok=True)

        run_count = 0
        total_steps = 0
        for run_id, run_steps in scenario_steps.groupby("run_id", sort=True):
            run_count += 1
            total_steps += len(run_steps)
            run_folder = scenario_folder / run_id
            run_folder.mkdir(parents=True, exist_ok=True)

            jsonl_lines = []
            for row in run_steps.sort_values("step_index").to_dict(orient="records"):
                for key in ("tool_args", "ground_truth_labels"):
                    row[key] = safe_json(row.get(key))
                cleaned = {}
                for k, v in row.items():
                    if isinstance(v, (dict, list)):
                        cleaned[k] = v
                    else:
                        try:
                            cleaned[k] = None if pd.isna(v) else v
                        except (TypeError, ValueError):
                            cleaned[k] = v
                row = cleaned
                jsonl_lines.append(json.dumps(row, ensure_ascii=False))
            (run_folder / "trace.jsonl").write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")

            record = summary_by_run.loc[run_id].to_dict()
            record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            (run_folder / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

            outcome = "CORRECT" if bool(record["predicted_positive"]) == bool(record["truth_positive"]) else "INCORRECT"
            interpretation = f"""Run: {run_id}
Scenario: {info['category']}
Expected label: {info['expected']}
Detector prediction: {record['predicted_label'] or 'healthy'}
Detected level: {record['detected_level']}
Result: {outcome}

How to interpret:
{info['meaning']}

Resource comparison for this controlled run:
- Baseline tokens: {record['baseline_tokens']}
- Protected tokens: {record['protected_tokens']}
- Baseline tool calls: {record['baseline_tool_calls']}
- Protected tool calls: {record['protected_tool_calls']}
- Detector overhead: {record['detector_overhead_ms']:.3f} ms

Publication warning:
This is a controlled synthetic run. It can be used to demonstrate implementation behavior and ablation logic, but not real-world generalization.
"""
            (run_folder / "INTERPRETATION.txt").write_text(interpretation, encoding="utf-8")

        predicted_positive_rate = float(summary[summary["scenario"] == scenario]["predicted_positive"].mean())
        scenario_rows.append(
            {
                "scenario_order": int(info["number"]),
                "scenario": scenario,
                "category": info["category"],
                "expected_label": info["expected"],
                "run_count": run_count,
                "step_count": total_steps,
                "mean_steps_per_run": total_steps / run_count,
                "detected_run_percent": 100.0 * predicted_positive_rate,
                "interpretation": info["meaning"],
            }
        )

        scenario_readme = f"""# {info['number']}. {info['category']}

Expected label: `{info['expected']}`

{info['meaning']}

This folder contains one subfolder per run. Open any run folder and read:

1. `trace.jsonl` — every step in execution order.
2. `summary.json` — detector and resource statistics.
3. `INTERPRETATION.txt` — plain-English explanation.
"""
        (scenario_folder / "README.md").write_text(scenario_readme, encoding="utf-8")

    scenario_df = pd.DataFrame(scenario_rows).sort_values("scenario_order")
    scenario_df.to_csv(tables_dir / "scenario_summary.csv", index=False)

    # Copy the original aggregate tables into one obvious place.
    for filename in [
        "detector_comparison.csv",
        "ablation.csv",
        "per_class_metrics.csv",
        "run_summary.csv",
        "trace_steps.csv",
        "human_annotation_template.csv",
    ]:
        pd.read_csv(raw / filename).to_csv(tables_dir / filename, index=False)


if __name__ == "__main__":
    main()

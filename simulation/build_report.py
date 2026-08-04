#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    x = df.copy()
    for col in x.select_dtypes(include=["float", "float64"]).columns:
        x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return x.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()

    report = args.experiment / "report"
    report.mkdir(parents=True, exist_ok=True)
    tables = args.experiment / "tables"

    comparison = pd.read_csv(tables / "detector_comparison.csv")
    ablation = pd.read_csv(tables / "ablation.csv")
    per_class = pd.read_csv(tables / "per_class_metrics.csv")
    scenarios = pd.read_csv(tables / "scenario_summary.csv")
    summary = json.loads((args.experiment / "raw" / "example_results.json").read_text())
    full = summary["full_detector"]
    efficiency = summary["efficiency"]

    start = f"""# READ RESULTS FIRST

## What you ran

- Seven scenario categories
- {summary['runs_per_scenario']} runs per category
- {summary['total_runs']} total runs
- {summary['total_steps']} total agent steps
- Semantic backend: `{summary['semantic_backend']}`
- Random seed: `{summary['seed']}`

## Folder map

```text
raw/       Original CSV and JSON output from the experiment
runs/      One folder for every individual run
           Each run has trace.jsonl, summary.json, and INTERPRETATION.txt
tables/    Aggregated data used in the figures
figures/   Publication-ready PNG and PDF figures
report/    Interpretation and paper wording
```

## What to open in order

1. `report/SCENARIO_CHECKLIST.md`
2. `tables/scenario_summary.csv`
3. `figures/figure_1_detector_f1.png`
4. `figures/figure_2_ablation.png`
5. `figures/figure_3_false_flags.png`
6. `report/PLOT_INTERPRETATION.md`
7. `report/CONTROLLED_VALIDATION_SECTION.md`

## Critical warning

These are controlled synthetic results. Publish them only as **controlled validation** showing that the implementation behaves as designed. Do not call them real-world accuracy or production savings.
"""
    (report / "READ_RESULTS_FIRST.md").write_text(start, encoding="utf-8")

    checklist = """# Seven-Scenario Checklist

Follow this list exactly.

1. Exact repeated action: confirm full detector predicts `repeated_action`.
2. Near-duplicate search: confirm full detector predicts `redundant_search_loop`.
3. Semantic plan repetition: confirm full detector predicts `semantic_repeat`.
4. Repeated error response: confirm full detector predicts `repeated_error_response`.
5. Hallucination: confirm full detector predicts `hallucination`.
6. Healthy code repair: confirm full detector does not intervene.
7. Healthy polling: confirm full detector does not intervene.

For each category, open one folder under `runs/`, then open:

- `trace.jsonl`
- `summary.json`
- `INTERPRETATION.txt`

After checking one example, inspect `tables/scenario_summary.csv` to confirm all runs follow the same expected pattern.
"""
    (report / "SCENARIO_CHECKLIST.md").write_text(checklist, encoding="utf-8")

    plot_text = f"""# Plot Interpretation

## Figure 1 — Detector F1

File: `figures/figure_1_detector_f1.png`

Interpretation: Exact hashing has high precision but detects only one anomaly family. F1 increases as additional levels cover near-duplicate, semantic, response, and hallucination cases. The fixed retry baseline performs poorly because it also stops healthy repeated workflows.

Recommended caption: **Detection F1 on the controlled seven-scenario benchmark. Error bars show 95% trace-level bootstrap confidence intervals.**

## Figure 2 — Cumulative ablation

File: `figures/figure_2_ablation.png`

Interpretation: Recall increases from {ablation.iloc[0]['recall']:.3f} at Level 1 to {ablation.iloc[-1]['recall']:.3f} for the complete cascade. Precision remains {ablation.iloc[-1]['precision']:.3f} in this controlled dataset. This is the central figure supporting the layered detector design.

Recommended caption: **Cumulative ablation showing precision, recall, and F1 as detector levels are added.**

## Figure 3 — False flags on healthy repetitions

File: `figures/figure_3_false_flags.png`

Interpretation: The fixed retry limit flags {comparison.loc[comparison.method == 'retry_limit', 'false_flags_per_100_healthy_runs'].iloc[0]:.0f} of every 100 healthy runs. The progress-aware cascade produces {comparison.loc[comparison.method == 'full', 'false_flags_per_100_healthy_runs'].iloc[0]:.0f} false flags in the controlled hard negatives. This figure demonstrates why repetition alone is insufficient.

Recommended caption: **False interventions per 100 healthy repeated workflows.**

## Figure 4 — Per-class performance

File: `figures/figure_4_per_class_f1.png`

Interpretation: Use this figure to confirm that the complete detector handles all five anomaly categories. Perfect values are expected because the controlled scenarios were built to isolate one definition at a time.

Recommended caption: **Per-class F1 of the complete detector on the controlled benchmark.**

## Figure 5 — Resource savings

File: `figures/figure_5_resource_savings.png`

Interpretation: The plot reports simulated resource reductions after intervention. It is useful to show how the measurement pipeline works, but it must not be presented as observed production savings. The current token reduction is {efficiency['token_savings_percent']:.2f}%.

Recommended caption: **Illustrative resource reduction in paired controlled runs; values are simulated and are not production measurements.**

## Figure 6 — Detector overhead

File: `figures/figure_6_overhead.png`

Interpretation: Runtime is only a few milliseconds per synthetic run. Do not compare small differences between bars as meaningful because initialization and timing noise dominate. In the final paper, separately measure hash, embedding, FAISS, and LLM-judge latency.

Recommended caption: **Mean detector runtime per controlled trajectory.**
"""
    (report / "PLOT_INTERPRETATION.md").write_text(plot_text, encoding="utf-8")

    section = f"""# Controlled Validation Section for the Paper

## Dataset

The controlled benchmark contained {summary['total_runs']} trajectories and {summary['total_steps']} agent steps, with {summary['runs_per_scenario']} trajectories for each of seven scenarios. Five scenarios represented anomaly classes and two represented healthy hard negatives in which repeated actions produced measurable progress.

{md_table(scenarios[['category', 'expected_label', 'run_count', 'step_count', 'mean_steps_per_run']])}

## Baseline comparison

{md_table(comparison)}

## Cumulative ablation

{md_table(ablation)}

## Per-class results

{md_table(per_class)}

## Paper-ready interpretation

The fixed retry-limit baseline achieved precision {comparison.loc[comparison.method == 'retry_limit', 'precision'].iloc[0]:.3f}, recall {comparison.loc[comparison.method == 'retry_limit', 'recall'].iloc[0]:.3f}, and F1 {comparison.loc[comparison.method == 'retry_limit', 'f1'].iloc[0]:.3f}, while flagging all healthy repeated workflows. Exact hashing achieved precision {comparison.loc[comparison.method == 'exact_only', 'precision'].iloc[0]:.3f} but recall only {comparison.loc[comparison.method == 'exact_only', 'recall'].iloc[0]:.3f}. Cumulative addition of near-duplicate hashing, semantic analysis, response hashing, and judge-based evaluation increased recall to {ablation.iloc[1]['recall']:.3f}, {ablation.iloc[2]['recall']:.3f}, {ablation.iloc[3]['recall']:.3f}, and {ablation.iloc[4]['recall']:.3f}, respectively. The complete cascade achieved F1 {full['f1']:.3f} without false interventions in the two controlled healthy categories.

## Limitation sentence that must remain

Because each scenario was deliberately constructed to isolate a detector definition, these results establish functional correctness and incremental detector coverage but do not establish real-world generalization.
"""
    (report / "CONTROLLED_VALIDATION_SECTION.md").write_text(section, encoding="utf-8")

    what = """# What to Publish

## Publish from this package

1. Dataset composition table.
2. Baseline comparison table.
3. Cumulative ablation table.
4. Figure 1: F1 comparison.
5. Figure 2: cumulative ablation.
6. Figure 3: healthy false flags.
7. Two example traces: one anomalous and one healthy-progress trace.
8. The limitation sentence in `CONTROLLED_VALIDATION_SECTION.md`.

## Do not publish as real-world evidence

1. Perfect F1 as a claim of production accuracy.
2. Simulated task-success improvement.
3. Simulated resource savings as production savings.
4. TF-IDF results as the final Sentence-Transformer experiment.
5. Heuristic judge results as final LLM-judge validation.

## Add before the final submission

1. Held-out real LLM and tool traces.
2. Two independent human annotations.
3. Sentence Transformer plus FAISS evaluation.
4. Validated LLM judge compared against human labels.
5. Paired protected versus unprotected real-agent runs.
"""
    (report / "WHAT_TO_PUBLISH.md").write_text(what, encoding="utf-8")


if __name__ == "__main__":
    main()

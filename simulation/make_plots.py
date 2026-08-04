#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()

    tables = args.experiment / "tables"
    figures = args.experiment / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(tables / "detector_comparison.csv")
    ablation = pd.read_csv(tables / "ablation.csv")
    per_class = pd.read_csv(tables / "per_class_metrics.csv")
    summary = json.loads((args.experiment / "raw" / "example_results.json").read_text())

    # Figure 1: baseline F1 comparison.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(comparison["method"], comparison["f1"])
    ax.errorbar(
        comparison["method"],
        comparison["f1"],
        yerr=[comparison["f1"] - comparison["f1_ci_low"], comparison["f1_ci_high"] - comparison["f1"]],
        fmt="none",
        capsize=4,
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Detector configuration")
    ax.set_title("Controlled benchmark: detector F1 with 95% bootstrap CI")
    ax.tick_params(axis="x", rotation=30)
    save(fig, figures, "figure_1_detector_f1")

    # Figure 2: cumulative ablation.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(ablation["configuration"], ablation["precision"], marker="o", label="Precision")
    ax.plot(ablation["configuration"], ablation["recall"], marker="o", label="Recall")
    ax.plot(ablation["configuration"], ablation["f1"], marker="o", label="F1")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Cumulative detector levels")
    ax.set_title("Ablation: contribution of detector levels")
    ax.legend()
    save(fig, figures, "figure_2_ablation")

    # Figure 3: false healthy flags.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(comparison["method"], comparison["false_flags_per_100_healthy_runs"])
    ax.set_ylabel("False flags per 100 healthy runs")
    ax.set_xlabel("Detector configuration")
    ax.set_title("Healthy hard-negative false interventions")
    ax.tick_params(axis="x", rotation=30)
    save(fig, figures, "figure_3_false_flags")

    # Figure 4: per-class F1.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(per_class["class"], per_class["f1"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Anomaly category")
    ax.set_title("Full detector performance by anomaly category")
    ax.tick_params(axis="x", rotation=30)
    save(fig, figures, "figure_4_per_class_f1")

    # Figure 5: illustrative resource reductions.
    efficiency = summary["efficiency"]
    resource = pd.DataFrame(
        {
            "metric": ["Tokens", "Tool calls", "Latency", "Gross cost"],
            "reduction_percent": [
                efficiency["token_savings_percent"],
                efficiency["tool_call_savings_percent"],
                efficiency["latency_savings_percent"],
                efficiency["gross_cost_savings_percent"],
            ],
        }
    )
    resource.to_csv(tables / "resource_savings.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(resource["metric"], resource["reduction_percent"])
    ax.set_ylabel("Reduction (%)")
    ax.set_xlabel("Resource metric")
    ax.set_title("Illustrative protected-run resource reductions")
    save(fig, figures, "figure_5_resource_savings")

    # Figure 6: measured overhead.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(comparison["method"], comparison["mean_overhead_ms_per_run"])
    ax.set_ylabel("Mean overhead (ms/run)")
    ax.set_xlabel("Detector configuration")
    ax.set_title("Measured detector overhead in the controlled benchmark")
    ax.tick_params(axis="x", rotation=30)
    save(fig, figures, "figure_6_overhead")


if __name__ == "__main__":
    main()

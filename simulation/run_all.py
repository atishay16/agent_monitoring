#!/usr/bin/env python3
"""Run all seven scenarios, organize every run, create plots, and build reports."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="controlled_validation_30")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--semantic-backend",
        choices=["tfidf", "sentence-transformer"],
        default="tfidf",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    experiment = root / "experiments" / args.name
    raw = experiment / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "easy_experiment.py",
            "--repetitions",
            str(args.repetitions),
            "--seed",
            str(args.seed),
            "--semantic-backend",
            args.semantic_backend,
            "--output",
            str(raw),
        ],
        root,
    )
    run([sys.executable, "organize_runs.py", str(experiment)], root)
    run([sys.executable, "make_plots.py", str(experiment)], root)
    run([sys.executable, "build_report.py", str(experiment)], root)

    print("\nDONE")
    print(f"Open this file first:\n  {experiment / 'report' / 'READ_RESULTS_FIRST.md'}")
    print(f"All individual runs:\n  {experiment / 'runs'}")
    print(f"Publication figures:\n  {experiment / 'figures'}")


if __name__ == "__main__":
    main()

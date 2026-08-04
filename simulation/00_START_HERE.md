# Start Here — Follow These Steps Exactly

## Step 1 — Open Terminal and enter this folder

```bash
cd seven_scenario_publish_package
```

Use the actual downloaded path if the folder is elsewhere.

## Step 2 — Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 3 — Run all seven categories

```bash
python run_all.py \
  --name controlled_validation_30 \
  --repetitions 30 \
  --seed 42 \
  --semantic-backend tfidf
```

This creates 30 runs for each of seven categories: 210 runs total.

## Step 4 — Open the result instructions

```bash
open experiments/controlled_validation_30/report/READ_RESULTS_FIRST.md
```

## Step 5 — Open the plots

```bash
open experiments/controlled_validation_30/figures
```

Use these three first:

1. `figure_1_detector_f1.png`
2. `figure_2_ablation.png`
3. `figure_3_false_flags.png`

## Step 6 — Inspect one individual run

```bash
open experiments/controlled_validation_30/runs/01_exact_loop/exact-000
```

Read `INTERPRETATION.txt`, then `trace.jsonl`, then `summary.json`.

Repeat with one folder from each of the seven scenario directories.

## Step 7 — Copy the controlled-validation section into the paper

Open:

```bash
open experiments/controlled_validation_30/report/CONTROLLED_VALIDATION_SECTION.md
```

Keep its limitation sentence. Do not present controlled synthetic numbers as real-world accuracy.

## Easiest Mac option

Double-click `run_all.command`. It creates the environment, installs packages, runs all scenarios, and generates plots and reports.

# READ RESULTS FIRST

## What you ran

- Seven scenario categories
- 30 runs per category
- 210 total runs
- 1140 total agent steps
- Semantic backend: `tfidf`
- Random seed: `42`

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

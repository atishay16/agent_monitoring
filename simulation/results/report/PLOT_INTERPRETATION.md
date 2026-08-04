# Plot Interpretation

## Figure 1 — Detector F1

File: `figures/figure_1_detector_f1.png`

Interpretation: Exact hashing has high precision but detects only one anomaly family. F1 increases as additional levels cover near-duplicate, semantic, response, and hallucination cases. The fixed retry baseline performs poorly because it also stops healthy repeated workflows.

Recommended caption: **Detection F1 on the controlled seven-scenario benchmark. Error bars show 95% trace-level bootstrap confidence intervals.**

## Figure 2 — Cumulative ablation

File: `figures/figure_2_ablation.png`

Interpretation: Recall increases from 0.200 at Level 1 to 1.000 for the complete cascade. Precision remains 1.000 in this controlled dataset. This is the central figure supporting the layered detector design.

Recommended caption: **Cumulative ablation showing precision, recall, and F1 as detector levels are added.**

## Figure 3 — False flags on healthy repetitions

File: `figures/figure_3_false_flags.png`

Interpretation: The fixed retry limit flags 100 of every 100 healthy runs. The progress-aware cascade produces 0 false flags in the controlled hard negatives. This figure demonstrates why repetition alone is insufficient.

Recommended caption: **False interventions per 100 healthy repeated workflows.**

## Figure 4 — Per-class performance

File: `figures/figure_4_per_class_f1.png`

Interpretation: Use this figure to confirm that the complete detector handles all five anomaly categories. Perfect values are expected because the controlled scenarios were built to isolate one definition at a time.

Recommended caption: **Per-class F1 of the complete detector on the controlled benchmark.**

## Figure 5 — Resource savings

File: `figures/figure_5_resource_savings.png`

Interpretation: The plot reports simulated resource reductions after intervention. It is useful to show how the measurement pipeline works, but it must not be presented as observed production savings. The current token reduction is 42.09%.

Recommended caption: **Illustrative resource reduction in paired controlled runs; values are simulated and are not production measurements.**

## Figure 6 — Detector overhead

File: `figures/figure_6_overhead.png`

Interpretation: Runtime is only a few milliseconds per synthetic run. Do not compare small differences between bars as meaningful because initialization and timing noise dominate. In the final paper, separately measure hash, embedding, FAISS, and LLM-judge latency.

Recommended caption: **Mean detector runtime per controlled trajectory.**

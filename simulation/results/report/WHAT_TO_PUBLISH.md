# What to Publish

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

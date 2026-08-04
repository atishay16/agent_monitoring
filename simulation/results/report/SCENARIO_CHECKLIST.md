# Seven-Scenario Checklist

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

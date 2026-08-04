# Controlled Validation Section for the Paper

## Dataset

The controlled benchmark contained 210 trajectories and 1140 agent steps, with 30 trajectories for each of seven scenarios. Five scenarios represented anomaly classes and two represented healthy hard negatives in which repeated actions produced measurable progress.

| category                           | expected_label          |   run_count |   step_count |   mean_steps_per_run |
|:-----------------------------------|:------------------------|------------:|-------------:|---------------------:|
| Exact repeated action              | repeated_action         |          30 |          180 |                    6 |
| Near-duplicate search loop         | redundant_search_loop   |          30 |          180 |                    6 |
| Semantic plan repetition           | semantic_repeat         |          30 |          180 |                    6 |
| Repeated error response            | repeated_error_response |          30 |          180 |                    6 |
| Hallucinated or unsupported answer | hallucination           |          30 |           90 |                    3 |
| Healthy progressive code repair    | healthy                 |          30 |          180 |                    6 |
| Healthy asynchronous polling       | healthy                 |          30 |          150 |                    5 |

## Baseline comparison

| method      |   precision |   recall |    f1 |   f1_ci_low |   f1_ci_high |   false_flags_per_100_healthy_runs |   mean_detection_delay_steps |   mean_overhead_ms_per_run |
|:------------|------------:|---------:|------:|------------:|-------------:|-----------------------------------:|-----------------------------:|---------------------------:|
| retry_limit |         0.6 |      0.6 | 0.6   |       0.526 |        0.663 |                                100 |                            1 |                      0.001 |
| exact_only  |         1   |      0.2 | 0.333 |       0.243 |        0.413 |                                  0 |                            0 |                      3.83  |
| L1_L2       |         1   |      0.4 | 0.571 |       0.486 |        0.648 |                                  0 |                            0 |                      3.477 |
| L1_L3       |         1   |      0.6 | 0.75  |       0.682 |        0.808 |                                  0 |                            0 |                      3.017 |
| L1_L4       |         1   |      0.8 | 0.889 |       0.846 |        0.927 |                                  0 |                            0 |                      2.612 |
| full        |         1   |      1   | 1     |       1     |        1     |                                  0 |                            0 |                      2.574 |

## Cumulative ablation

| configuration   |   precision |   recall |    f1 |   false_flags_per_100_healthy_runs |   mean_detection_delay_steps |   mean_overhead_ms_per_run |
|:----------------|------------:|---------:|------:|-----------------------------------:|-----------------------------:|---------------------------:|
| L1              |           1 |      0.2 | 0.333 |                                  0 |                            0 |                      3.83  |
| L1-L2           |           1 |      0.4 | 0.571 |                                  0 |                            0 |                      3.477 |
| L1-L3           |           1 |      0.6 | 0.75  |                                  0 |                            0 |                      3.017 |
| L1-L4           |           1 |      0.8 | 0.889 |                                  0 |                            0 |                      2.612 |
| L1-L5           |           1 |      1   | 1     |                                  0 |                            0 |                      2.574 |

## Per-class results

| class                   |   support |   precision |   recall |   f1 |   mean_delay_steps |
|:------------------------|----------:|------------:|---------:|-----:|-------------------:|
| repeated_action         |        30 |           1 |        1 |    1 |                  0 |
| redundant_search_loop   |        30 |           1 |        1 |    1 |                  0 |
| semantic_repeat         |        30 |           1 |        1 |    1 |                  0 |
| repeated_error_response |        30 |           1 |        1 |    1 |                  0 |
| hallucination           |        30 |           1 |        1 |    1 |                  0 |

## Paper-ready interpretation

The fixed retry-limit baseline achieved precision 0.600, recall 0.600, and F1 0.600, while flagging all healthy repeated workflows. Exact hashing achieved precision 1.000 but recall only 0.200. Cumulative addition of near-duplicate hashing, semantic analysis, response hashing, and judge-based evaluation increased recall to 0.400, 0.600, 0.800, and 1.000, respectively. The complete cascade achieved F1 1.000 without false interventions in the two controlled healthy categories.

## Limitation sentence that must remain

Because each scenario was deliberately constructed to isolate a detector definition, these results establish functional correctness and incremental detector coverage but do not establish real-world generalization.

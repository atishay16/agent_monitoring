# Illustrative Paper Tables — Do Not Publish as Real-Agent Evidence

These results were produced from controlled synthetic traces. They validate the
experiment plumbing only. Replace them with held-out real LangGraph traces and
human labels before publication.

## Dataset

- Runs: **210**
- Steps: **1140**
- Runs per scenario: **30**
- LangGraph package used: **True**

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

## Per-class full-detector results

| class                   |   support |   precision |   recall |   f1 |   mean_delay_steps |
|:------------------------|----------:|------------:|---------:|-----:|-------------------:|
| repeated_action         |        30 |           1 |        1 |    1 |                  0 |
| redundant_search_loop   |        30 |           1 |        1 |    1 |                  0 |
| semantic_repeat         |        30 |           1 |        1 |    1 |                  0 |
| repeated_error_response |        30 |           1 |        1 |    1 |                  0 |
| hallucination           |        30 |           1 |        1 |    1 |                  0 |

## Paired operational impact

| Metric | Illustrative value |
|---|---:|
| Token savings | 42.09% |
| Token savings 95% bootstrap CI | [36.20%, 43.97%] |
| Tool-call savings | 44.74% |
| Latency savings | 41.76% |
| Gross cost savings | 42.51% |
| Intervention precision | 1.000 |
| Premature-stop rate | 0.000 |
| Paired Wilcoxon p-value for token savings | 0.000000 |

## Example wording

> In the controlled synthetic benchmark, the full cascade achieved precision
> 1.000, recall 1.000, and F1
> 1.000. It reduced total token use by
> 42.09% in paired simulated runs. These
> values demonstrate implementation behavior only and are not used as evidence
> of real-world generalization.

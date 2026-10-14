# Week 6 - Final results on the held-out last quarter

Trained on 2025-01-06 to 2025-09-30, tested once on 2025-10-01 to 2025-12-31.

- Test rows: 3680, faults: 237 (6.4%)
- Forest settings (Exercise 2): `{'max_depth': None, 'min_samples_leaf': 20, 'n_estimators': 300}`
- Decision threshold: 0.3

## Scores

| Model | AUC (test) |
| --- | --- |
| Random forest (tuned) | 0.936 |
| Logistic regression | 0.961 |

## Top 5 features (forest importance)

- `vibration_mm_s_trend`: 0.445
- `vibration_mm_s_delta30`: 0.297
- `vibration_mm_s`: 0.074
- `load_pct`: 0.061
- `pressure_bar`: 0.036

## By machine type (forest)

| machine_type | rows | faults | auc | recall | false_alarms |
| --- | --- | --- | --- | --- | --- |
| chiller | 920 | 60 | 0.929 | 0.683 | 26 |
| compressor | 920 | 57 | 0.935 | 0.702 | 16 |
| conveyor | 920 | 61 | 0.949 | 0.672 | 24 |
| pump | 920 | 59 | 0.934 | 0.678 | 15 |

## By month (forest)

| month | rows | faults | auc | recall | false_alarms |
| --- | --- | --- | --- | --- | --- |
| 2025-10 | 1240 | 79 | 0.934 | 0.671 | 28 |
| 2025-11 | 1200 | 81 | 0.937 | 0.679 | 26 |
| 2025-12 | 1240 | 77 | 0.937 | 0.701 | 27 |


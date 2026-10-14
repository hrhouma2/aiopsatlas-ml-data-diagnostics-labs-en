# Week 4 — Data preparation and dataset separation: solution scripts

Run every script from the kit root, with the venv active and `python data/make_dataset.py` done.

| Script | What it does |
| --- | --- |
| `exercise_1_solution.py` | Exercise 1 — Build the feature table: merge readings with machines, rolling features per machine, one-hot type, save `data/features.csv` (14,400 rows, 18 columns). |
| `exercise_2_solution.py` | Exercise 2 — Four splits of `data/features.csv` (random stratified, random plain, temporal, group by machine) with row counts, positive rates and "neighbour" counts. |
| `exercise_3_solution.py` | Exercise 3 — Leakage: a scaler fitted before the split, a column that knows the future (AUC 0.993), then a `Pipeline` fitted on train only. |

## The canonical feature table: `data/features.csv`

Every week from Week 5 on reads this file. Build it once with Exercise 1, or with:

```bash
python common/features.py
```

`common/features.py` exposes `build_features(readings_path="data/clean/readings.csv", machines_path="data/clean/machines.csv")`, which returns the same DataFrame. The recipe, in this exact order:

1. Merge `readings.csv` with `machines.csv` on `machine_id`. Sort by `machine_id`, then `date`.
2. `machine_age = 2025 - install_year`.
3. Per machine (`groupby("machine_id")`), for `temperature_c` and `vibration_mm_s`:
   - `<col>_delta30 = value - rolling(30, min_periods=5).mean().shift(1)` — today minus the mean of the previous 30 days (today excluded).
   - `<col>_trend = rolling(3, min_periods=1).mean() - rolling(14, min_periods=3).mean()` — the last 3 days against the last 14 days.
4. One-hot `machine_type` into `type_pump`, `type_compressor`, `type_conveyor`, `type_chiller` (integers 0/1).
5. Drop every row where a rolling column is NaN. That removes the first 5 days of each machine: 200 rows. The first date kept is `2025-01-06`.
6. Keep the 18 columns below, in this order.

| Column | Type | Meaning |
| --- | --- | --- |
| `machine_id` | text | `M001` to `M040`. Not a feature: use it for group splits and reports. |
| `date` | text `YYYY-MM-DD` | one row per machine per day. Not a feature: use it for the temporal split. |
| `load_pct` | float | load of the day, 5 to 100 % |
| `ambient_c` | float | outside temperature of the day, in Celsius |
| `temperature_c` | float | casing temperature, in Celsius |
| `vibration_mm_s` | float | vibration, in mm/s |
| `pressure_bar` | float | pressure, in bar |
| `power_kw` | float | power drawn, in kW. Target of the Week 5 regression; a feature for the fault task. |
| `machine_age` | int | years since installation, 2 to 13 |
| `temperature_c_delta30` | float | today's temperature minus the machine's own 30-day mean |
| `vibration_mm_s_delta30` | float | today's vibration minus the machine's own 30-day mean |
| `temperature_c_trend` | float | 3-day mean minus 14-day mean of temperature |
| `vibration_mm_s_trend` | float | 3-day mean minus 14-day mean of vibration |
| `type_pump` | int 0/1 | 1 if the machine is a pump |
| `type_compressor` | int 0/1 | 1 if the machine is a compressor |
| `type_conveyor` | int 0/1 | 1 if the machine is a conveyor |
| `type_chiller` | int 0/1 | 1 if the machine is a chiller |
| `fault_next_7d` | int 0/1 | the label: 1 if a fault happens in the next 7 days. 7.5 % of rows. |

Shape: `(14400, 18)`. No missing value.

## The temporal split used in every week

```python
train = features[features["date"] < "2025-10-01"]   # 10,720 rows, positive rate 7.9 %
test = features[features["date"] >= "2025-10-01"]   # 3,680 rows, positive rate 6.4 %
```

The 15 feature columns are every column except `machine_id`, `date` and `fault_next_7d`.

## Reference numbers (logistic regression, temporal split, test quarter)

| Features | AUC |
| --- | --- |
| 6 sensor columns only | 0.879 |
| 6 sensors + 4 rolling columns | 0.954 |
| all 15 feature columns | 0.961 |

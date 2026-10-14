# Sensor calibration and data quality guide

Every machine sends one reading per day to the diagnostics database. This guide explains what each field means and which values are impossible.

## Fields of a reading

| Field | Unit | Meaning | Valid range |
| --- | --- | --- | --- |
| `load_pct` | percent | Share of rated capacity used that day | 0 to 100 |
| `ambient_c` | degrees C | Air temperature in the machine room | -30 to 45 |
| `temperature_c` | degrees C | Casing or motor temperature | -20 to 120 |
| `vibration_mm_s` | mm/s | Vibration velocity (RMS) | 0.1 to 30 |
| `pressure_bar` | bar | Process pressure | 0.2 to 15 |
| `power_kw` | kW | Average electrical power drawn | 0.5 to 120 |

Any value outside the valid range is a sensor problem or a data entry error, never a real physical state. A temperature of 999 C is a common placeholder written by an old gateway when the sensor is disconnected.

## Missing values

- A missing temperature or vibration is usually a gateway timeout. It happens on about 2 percent of days.
- Never fill a missing value with zero. Zero is a real value and would look like a stopped machine.
- Fill with the machine's own median of the last 30 days, or leave empty and let the model ignore that row.

## Duplicate readings

The gateway sometimes resends a reading after a network retry. The result is two identical rows. Keep one. Two rows with the same `machine_id` and `date` but different values are a conflict: keep the last one and open a `sensor_fault` incident.

## Calibration schedule

| Sensor | Check | Frequency |
| --- | --- | --- |
| Temperature (PT100) | Compare with a reference probe, difference under 1 C | Every 6 months |
| Vibration (accelerometer) | Shaker table at 10 mm/s, difference under 5 percent | Every 12 months |
| Pressure | Dead-weight tester at 5 bar, difference under 0.1 bar | Every 12 months |
| Power meter | Compare with the site meter over one hour | Every 24 months |

## Signs of a faulty sensor

1. The same value repeated for more than 3 days (frozen sensor).
2. Exactly 0.0 on a running machine (broken cable).
3. A value that jumps by more than 50 percent in one day and comes back the next day (spike).
4. A value outside the valid range.

Report any of these as a `sensor_fault` incident with **low** severity.

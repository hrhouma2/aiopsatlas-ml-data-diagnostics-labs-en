# Conveyor and chiller troubleshooting guide

Applies to machines of type `conveyor` (M021 to M030) and `chiller` (M031 to M040).

## Conveyors (NorthPeak BX series)

### Normal operating ranges

| Signal | Normal | Warning | Stop the conveyor |
| --- | --- | --- | --- |
| Motor temperature | 15 to 45 C | 45 to 55 C | above 55 C |
| Vibration | below 3.0 mm/s | 3.0 to 4.5 mm/s | above 4.5 mm/s |
| Power at full load | 18 to 24 kW | 24 to 28 kW | above 28 kW |

Conveyors have no pressure circuit. The `pressure_bar` value of a conveyor is the hydraulic tensioner pressure, normally about 1.0 bar.

### Symptom: belt slipping or vibration

1. Check the belt tension. The belt should sag no more than 2 cm at the middle.
2. Check the gearbox oil level.
3. Listen to the drive shaft bearing at both ends. Replace bearings in pairs.
4. Clear material stuck under the belt.

### Symptom: sensor reads zero for days

A vibration sensor that shows exactly 0.0 for more than one day is broken, not quiet. A running conveyor always vibrates a little. Check the cable near the belt guard, where it often gets cut.

## Chillers (NorthPeak KX series)

### Normal operating ranges

| Signal | Normal | Warning | Stop the chiller |
| --- | --- | --- | --- |
| Compressor temperature | 10 to 45 C | 45 to 60 C | above 60 C |
| Vibration | below 3.5 mm/s | 3.5 to 5.5 mm/s | above 5.5 mm/s |
| Refrigerant pressure | 3.5 to 5.0 bar | 3.0 to 3.5 bar | below 3.0 bar |
| Power at full load | 55 to 68 kW | 68 to 75 kW | above 75 kW |

### Symptom: pressure falling slowly

A refrigerant leak makes the pressure fall by about 0.1 bar per day. Look for oily traces at the hose fittings. Do not top up refrigerant before the leak is fixed.

### Symptom: overheating in summer

Chillers work harder when the ambient temperature is high. A chiller temperature 10 C above the yearly average is normal in July. Compare with the same month last year before opening an incident.

### Symptom: coolant leak

1. Stop the chiller.
2. Tighten the hose fitting to 25 Nm.
3. Top up the coolant and run for 30 minutes.
4. Check the fitting again. If it still leaks, replace the hose.

# NorthPeak sites overview and glossary

NorthPeak Manufacturing runs 40 machines on three sites in Canada. The diagnostics team uses one shared database to watch them.

## Sites

| Site | Machines | Climate note |
| --- | --- | --- |
| Montreal | about half of the fleet | Cold winters, hot humid summers |
| Quebec City | about a quarter | Coldest site, about 2 C below Montreal |
| Toronto | about a quarter | Mildest site, about 1.5 C above Montreal |

## Machine types

| Type | Series | Count | Main job |
| --- | --- | --- | --- |
| pump | PX | 10 | Moves water and coolant between buildings |
| compressor | CX | 10 | Produces compressed air for tools and valves |
| conveyor | BX | 10 | Moves parts between work stations |
| chiller | KX | 10 | Cools the process water |

Machine ids go from `M001` to `M040`. Pumps are M001 to M010, compressors M011 to M020, conveyors M021 to M030 and chillers M031 to M040.

## Glossary

- **Bearing**: the part that lets a shaft turn smoothly. It wears out and makes noise and vibration before it fails.
- **Casing**: the outer metal body of a machine. Its temperature is the easiest one to measure.
- **Downtime**: hours during which the machine did not run because of an incident.
- **Fault**: a real problem on the machine. In the database, `fault_next_7d = 1` means an incident happened within the next 7 days.
- **Gateway**: the small computer that collects sensor values and sends them to the database once a day.
- **Incident**: a recorded event with a category, a severity and a description.
- **Load**: how hard the machine works, as a percent of its rated capacity.
- **Rated power**: the power the machine draws at 100 percent load, from the manufacturer's plate.
- **RMS vibration**: the standard way to measure vibration velocity, in mm/s.
- **Running hours**: total hours the machine has worked since installation.
- **Sensor fault**: the measurement is wrong but the machine is fine.
- **Thermal switch**: a safety part that stops the machine when the casing is too hot.

## Who uses the data

- Technicians read yesterday's readings each morning.
- Supervisors look at the weekly incident list.
- The diagnostics team builds models that predict faults 7 days ahead, so that a technician can look at the machine before it stops.

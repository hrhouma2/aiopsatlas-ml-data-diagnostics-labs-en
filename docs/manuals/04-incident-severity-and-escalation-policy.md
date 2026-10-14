# Incident severity and escalation policy

This policy applies to all NorthPeak sites: Montreal, Quebec City and Toronto.

## Severity levels

| Severity | Definition | Response time | Who is notified |
| --- | --- | --- | --- |
| low | Machine still runs. No safety risk. | Next working day | Site technician |
| medium | Machine runs with reduced load, or stopped for less than 8 hours. | Within 4 hours | Site technician and site supervisor |
| high | Machine stopped for more than 8 hours, or any safety risk. | Within 1 hour | Site supervisor, maintenance manager, safety officer |

## Rules

1. Any leak of oil, coolant or refrigerant near an electrical panel is **high**, whatever the size.
2. A sensor fault alone is **low**. The machine is fine; only the measurement is wrong.
3. Two incidents of the same category on the same machine within 30 days: raise the second one to at least **medium**.
4. Three incidents of any category on the same machine within 90 days: the maintenance manager must review the machine for replacement.
5. Bearing wear on a compressor is at least **medium**. It gets worse within a week.

## Escalation contacts

- Site technician: the person on duty, listed on the site board.
- Site supervisor: Montreal, D. Lavoie; Quebec City, P. Gagnon; Toronto, R. Singh.
- Maintenance manager: C. Bouchard, all sites.
- Safety officer: N. Petrov, all sites.

## Recording an incident

Every incident record must contain:

- the machine id (for example `M014`),
- the date in ISO format (`2025-03-08`),
- one category from: `bearing_wear`, `overheating`, `leak`, `electrical`, `sensor_fault`,
- the severity,
- the downtime in hours,
- the repair cost in Canadian dollars,
- the name of the technician,
- a short description in plain English.

Incidents without a technician name are returned to the site for completion.

## Data privacy

Incident descriptions must not contain personal information about employees other than the technician's name. Do not write badge numbers, phone numbers or home addresses in a description.

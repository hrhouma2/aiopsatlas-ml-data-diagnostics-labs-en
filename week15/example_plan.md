# Final project plan

## 1. Title

Which NorthPeak machines should a technician visit this week?

## 2. Question

For each of the 40 machines, what is the chance of a fault in the next 7 days, and why? The user is the site supervisor who plans the Monday visits.

## 3. Data

`data/clean/readings.csv` (14,600 rows, all of 2025), `data/clean/machines.csv` (40 rows), `data/clean/incidents.csv` (155 rows) for the check of predictions against real incidents, and `docs/manuals/` (6 files) for the explanations. I will not use `power_kw` as a feature: it is a result of the load, not a cause of faults.

## 4. Method

Feature table with rolling deltas and trends per machine (Week 4). Random forest with a temporal split, train before 2025-10-01 (Week 6). Baseline: `DummyClassifier(strategy="prior")` (Week 5). Top-3 manual chunks from Chroma and a grounded `llama3.2` prompt (Weeks 9 and 10). One Streamlit page with a status colour per machine (Week 14).

## 5. Evaluation

AUC and recall at threshold 0.3 on the days from 2025-10-01, against the baseline. Cost table: a false alarm counts one visit, a missed fault counts the mean downtime of the category. For the LLM part: 10 golden questions, a keyword check on each answer, and a count of "I don't know" on the 2 questions the manuals cannot answer.

## 6. Risks

Leakage from a random split: I use a temporal split and check that no test date is before the last train date. The LLM invents a range: the page shows the manual chunk next to the answer, and the range check is done in Python. Time: the app is the last step, a table of probabilities is the fallback.

## 7. Timeline

| Date | Milestone | Done? |
| --- | --- | --- |
| Jan 7 | plan reviewed by a classmate | |
| Jan 9 | feature table, baseline and random forest, first AUC | |
| Jan 11 | RAG explanation, golden set run, numbers in the report | |
| Jan 13 | slides ready, dry run of the 5 minutes | |
| Jan 14 | final submission: code, report, presentation | |

## 8. Ethics check

Bias: I report recall per site and per machine age group, not only the global number. Costs: a missed fault costs about 9 hours of downtime on average, a false alarm costs one visit; the threshold is chosen with this table. Personal data: the app never shows the `technician` column. Decision: the supervisor decides, the app suggests. Limits: the model saw one year of one fleet; the LLM is a 3B model that can misread a table.

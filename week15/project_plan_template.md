# Final project plan

> Copy this file to `week15/my_plan.md`, replace every line that starts with `>`, and keep the headings.
> One page. Short sentences. Numbers where you can. Check it with `python week15/check_plan.py week15/my_plan.md`.

## 1. Title

> One line. Say what the project does, not what it uses. Example: "Which NorthPeak machines should a technician visit this week?"

## 2. Question

> One question your project answers. It must be answerable with the NorthPeak data.
> Say who would use the answer (a technician, a site supervisor, the maintenance manager).

## 3. Data

> Which files or tables: `data/clean/readings.csv`, `incidents.csv`, `machines.csv`, `diagnostics.sqlite`, `docs/manuals/`.
> Which columns. How many rows. Which period. What you will NOT use, and why.

## 4. Method

> The steps, in order, with the week each one comes from.
> Example: feature table (Week 4) -> random forest with temporal split (Week 6) -> RAG explanation (Weeks 9 and 10) -> Streamlit page (Week 14).
> Name the baseline you will compare against.

## 5. Evaluation

> Which metric, on which split, against which baseline.
> Example: AUC and recall at threshold 0.3 on the days from 2025-10-01, against a `DummyClassifier`.
> For an LLM part: how many golden questions, what you check (a keyword, a JSON key, a count).

## 6. Risks

> Three risks and what you do about each.
> Examples: data leakage from a random split; the LLM inventing a range; running out of time on the app.

## 7. Timeline

| Date | Milestone | Done? |
| --- | --- | --- |
| > Jan 7 | > plan reviewed by a classmate | |
| > Jan 9 | > data work and baseline finished | |
| > Jan 11 | > model or LLM part finished, first numbers in the report | |
| > Jan 13 | > slides ready, dry run of the 5 minutes | |
| Jan 14 | final submission: code, report, presentation | |

## 8. Ethics check

> One line for each: bias between sites or machine ages; cost of a false alarm vs a missed fault; personal data (technician names); who takes the final decision; limits you will state in the report.

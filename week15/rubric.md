# Final project rubric - 100 points

Used for the peer review of Exercise 2 and for the final grade. Read the three levels of a row, pick the one that matches, write the points and one sentence of evidence.

| Criterion | Points | Level 3 - full points | Level 2 - about half | Level 1 - few or zero |
| --- | --- | --- | --- | --- |
| 1. Problem | 10 | One clear question, a named user, answerable with the NorthPeak data | Question is clear but too wide, or the user is vague | No question, or a question the data cannot answer |
| 2. Data work | 15 | Right files and columns; cleaning and feature steps shown and checked with real counts | Data loaded and used; checks missing or partial | Data misread, wrong join, or numbers that do not add up |
| 3. Modelling or LLM part | 20 | Method fits the question; baseline present; temporal split or grounded prompt; settings documented | Method works but no baseline, or a random split on time data, or an ungrounded prompt | Method does not run, or does not answer the question |
| 4. Evaluation | 15 | Right metric for the question; results on held-out data; a short error analysis | A metric is reported but on the training data, or without a baseline | No metric, or a metric that does not fit (accuracy on 7 percent positives) |
| 5. Reproducibility | 15 | `README` with commands in order; seeds fixed; versions listed; a smoke test that passes | Runs after some guessing; some versions or seeds missing | Cannot be run by the reviewer |
| 6. Communication | 15 | Report of 3 to 5 pages in the approach / results / limitations order; 5-minute talk with one chart per idea | Report or talk is complete but long, unordered, or hides the limits | Report missing, or the talk is a code walkthrough |
| 7. Ethics | 10 | Names the bias risk, the cost of both error types, the private fields, and who decides | Mentions ethics in general words only | No mention, or personal data shown without need |

## How to score

- Level 3 gives the full points of the row. Level 2 gives about half. Level 1 gives 0 to 3 points.
- Write one sentence of evidence per row. "I ran `python week14/smoke_test.py` and got 7/7" is evidence. "Looks fine" is not.
- Total the seven rows. 70 and above is a pass.

## Peer review form

```text
Project reviewed: ......................................  Reviewer: ......................

1. Problem            ..../10   evidence: ..............................................
2. Data work          ..../15   evidence: ..............................................
3. Modelling or LLM   ..../20   evidence: ..............................................
4. Evaluation         ..../15   evidence: ..............................................
5. Reproducibility    ..../15   evidence: ..............................................
6. Communication      ..../15   evidence: ..............................................
7. Ethics             ..../10   evidence: ..............................................

Total                 ..../100

One thing to keep: .......................................................................
One thing to fix before Jan 14: ...........................................................
```

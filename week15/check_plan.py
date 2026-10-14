"""Week 15, Exercise 1 - Check a filled project plan.

Run from the kit root:

    python week15/check_plan.py week15/my_plan.md

Checks that the eight headings of the template are still there, that no
placeholder line (starting with ">") is left, that each section has some
text, and that the plan names at least one data file and one week.
Exit code 0 when everything passes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADINGS = ["Title", "Question", "Data", "Method", "Evaluation", "Risks", "Timeline", "Ethics check"]
DATA_WORDS = ["readings", "incidents", "machines", "diagnostics.sqlite", "manuals", "features"]


def sections(text: str) -> dict[str, str]:
    """Map each '## N. Heading' to the text under it."""
    parts = re.split(r"^## \d+\. (.+)$", text, flags=re.MULTILINE)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def check_plan(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    found = sections(text)
    problems = []
    for h in HEADINGS:
        if h not in found:
            problems.append(f"missing heading: ## {h}")
        elif not found[h]:
            problems.append(f"empty section: {h}")
    placeholders = [line for line in text.splitlines() if line.strip().startswith(">")]
    if placeholders:
        problems.append(f"{len(placeholders)} placeholder line(s) still start with '>'")
    low = text.lower()
    if not any(w in low for w in DATA_WORDS):
        problems.append("the Data section names no NorthPeak file or table")
    if not re.search(r"week\s*\d+", low):
        problems.append("the Method section names no week of the course")
    return problems


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "week15/my_plan.md"
    issues = check_plan(target)
    for issue in issues:
        print("FAIL ", issue)
    if not issues:
        print(f"PASS  {target}: 8 sections filled, no placeholder left")
    sys.exit(1 if issues else 0)

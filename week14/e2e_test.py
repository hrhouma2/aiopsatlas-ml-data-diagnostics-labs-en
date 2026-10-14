"""Week 14, Exercise 3 - End-to-end test of the two Streamlit pages, without a browser.

Run from the kit root, with the venv active and Ollama running:

    python week14/e2e_test.py

Streamlit ships a test runner, `AppTest`. It executes the page like the
browser would, then lets us read the widgets and the output. We select a
machine, move the date, click "Explain" and check that an answer appeared.
`AppTest.from_file` resolves paths against this file, so "app.py" means
week14/app.py; the data paths inside the apps still need the kit root as cwd.
"""

from __future__ import annotations

import sys

from streamlit.testing.v1 import AppTest

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# --- Exercise 1 page -------------------------------------------------------
at = AppTest.from_file("app.py", default_timeout=60).run()
check("app.py runs without exception", not at.exception, str(at.exception) if at.exception else "")
check("app.py has 40 machines", len(at.selectbox[0].options) == 40, f"{len(at.selectbox[0].options)} options")
check("app.py default machine is M001", at.selectbox[0].value == "M001", at.selectbox[0].value)

at.selectbox[0].set_value("M014").run()
texts = [m.value for m in at.markdown] + [c.value for c in at.caption]
check("app.py switches to M014", any("compressor" in t for t in texts), "caption mentions compressor")
check("app.py shows a chart", len(at.get("plotly_chart")) == 1, f"{len(at.get('plotly_chart'))} chart(s)")

# --- Exercise 2 page -------------------------------------------------------
at2 = AppTest.from_file("app_full.py", default_timeout=180).run()
check("app_full.py runs without exception", not at2.exception, str(at2.exception) if at2.exception else "")

at2.selectbox[0].set_value("M004")
at2.select_slider[0].set_value(__import__("datetime").date(2025, 12, 18)).run()
banner = " ".join(m.value for m in at2.markdown)
check("M004 on 2025-12-18 is red", "Act now" in banner and "0.97" in banner, "banner says 'Act now' and 0.97")

at2.button[0].click().run()
check("Explain button ran", not at2.exception, str(at2.exception) if at2.exception else "")
answer = " ".join(m.value for m in at2.markdown)
check("explanation mentions vibration or coupling", "vibration" in answer.lower() or "coupling" in answer.lower(), "keyword found")
print("\nExplanation shown on the page (your text will differ):")
print(max((m.value for m in at2.markdown if "<div" not in m.value), key=len, default="(none)"))

print()
print("all passed" if failures == 0 else f"{failures} check(s) failed")
sys.exit(1 if failures else 0)

"""Week 10, Exercise 2 - Structured extraction from incident descriptions.

Run from the kit root, with the venv active and Ollama running:

    python week10/exercise_2_solution.py

Steps:
  1. describe the output you want as a Pydantic model;
  2. ask llama3.2 for that exact shape with with_structured_output();
  3. extract {category, severity, part} from one description;
  4. run the extractor on the first 10 incidents;
  5. compare the predicted category with the true column;
  6. look at the errors honestly.
"""

from typing import Literal

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

MODEL = "llama3.2"
N_INCIDENTS = 10


# Step 1 - the shape of the answer
class IncidentFacts(BaseModel):
    """What we want to read out of one technician note."""

    category: Literal["bearing_wear", "overheating", "leak", "electrical", "sensor_fault"] = Field(
        description="the kind of fault"
    )
    severity: Literal["low", "medium", "high"] = Field(description="how serious the incident is")
    part: str = Field(description="the machine part named in the text, or 'unknown'")


print("== Step 1: the Pydantic model ==")
print(IncidentFacts.model_json_schema()["properties"]["category"])
print()

# Step 2 - the extractor chain
print("== Step 2: with_structured_output ==")
SYSTEM = """You read one incident description from a factory and fill the fields.
category: bearing_wear = noise, vibration or bearing play; overheating = temperature too high,
thermal switch, cooling; leak = oil, coolant or a pressure drop with a seal; electrical = breaker,
fault code, power loss, terminal; sensor_fault = impossible reading, frozen or flatlined sensor,
sensor replaced.
severity: high = machine stopped or safety risk; medium = reduced load or short stop;
low = machine still runs.
part: the physical part named in the text (for example drive shaft, gearbox, hose fitting).
Use 'unknown' if no part is named."""

llm = ChatOllama(model=MODEL, temperature=0)
prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", "{description}")])
extractor = prompt | llm.with_structured_output(IncidentFacts)
print(type(extractor).__name__)
print()

# Step 3 - one description
print("== Step 3: one description ==")
incidents = pd.read_csv("data/clean/incidents.csv")
row = incidents.iloc[9]
print("text  :", row.description)
facts = extractor.invoke({"description": row.description})
print("object:", facts)
print("dict  :", facts.model_dump())
print("type of facts.category:", type(facts.category).__name__)
print()

# Step 4 - ten incidents
print("== Step 4: the first 10 incidents ==")
sample = incidents.head(N_INCIDENTS)
results = extractor.batch([{"description": d} for d in sample.description])
table = pd.DataFrame(
    {
        "incident_id": sample.incident_id.values,
        "true_category": sample.category.values,
        "pred_category": [r.category for r in results],
        "true_severity": sample.severity.values,
        "pred_severity": [r.severity for r in results],
        "part": [r.part for r in results],
    }
)
print(table.to_string(index=False))
print()

# Step 5 - compare with the truth
print("== Step 5: accuracy ==")
cat_ok = (table.true_category == table.pred_category).sum()
sev_ok = (table.true_severity == table.pred_severity).sum()
print(f"category correct: {cat_ok}/{N_INCIDENTS}")
print(f"severity correct: {sev_ok}/{N_INCIDENTS}")
print()

# Step 6 - the errors
print("== Step 6: where it went wrong ==")
wrong_cat = table[table.true_category != table.pred_category]
if wrong_cat.empty:
    print("no category error in this run")
for r in wrong_cat.itertuples():
    text = incidents.loc[incidents.incident_id == r.incident_id, "description"].iloc[0]
    print(f"{r.incident_id}: category true={r.true_category} pred={r.pred_category}")
    print("   text:", text)
wrong_sev = table[table.true_severity != table.pred_severity]
print(f"severity wrong on {len(wrong_sev)} incidents; two examples:")
for r in wrong_sev.head(2).itertuples():
    text = incidents.loc[incidents.incident_id == r.incident_id, "description"].iloc[0]
    print(f"{r.incident_id}: severity true={r.true_severity} pred={r.pred_severity}")
    print("   text:", text)
table.to_csv("week10/extraction_results.csv", index=False)
print("saved week10/extraction_results.csv")

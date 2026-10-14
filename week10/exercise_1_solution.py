"""Week 10, Exercise 1 - A chain that summarises an incident.

Run from the kit root, with the venv active and Ollama running:

    python week10/exercise_1_solution.py

Steps:
  1. call llama3.2 through LangChain's ChatOllama, temperature 0;
  2. write a prompt template with two variables;
  3. build the chain  prompt | llm | parser;
  4. run it on one incident of data/clean/incidents.csv;
  5. run it on five incidents with .batch();
  6. check every summary with a robust test (machine id present, short).
"""

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

MODEL = "llama3.2"

# Step 1 - the model as a LangChain component
print("== Step 1: ChatOllama ==")
llm = ChatOllama(model=MODEL, temperature=0)
reply = llm.invoke("What does a chiller do? One sentence.")
print(type(reply).__name__, "->", reply.content)
print()

# Step 2 - a prompt template with variables
print("== Step 2: the template ==")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the maintenance assistant of NorthPeak Manufacturing. "
            "Summarise the incident for a supervisor in ONE sentence of 25 words or fewer. "
            "Always name the machine id.",
        ),
        (
            "user",
            "Machine {machine_id} ({machine_type}), {date}, category {category}, "
            "severity {severity}.\nTechnician note: {description}",
        ),
    ]
)
print("input variables:", prompt.input_variables)
print()

# Step 3 - the chain
print("== Step 3: the chain ==")
chain = prompt | llm | StrOutputParser()
print(type(chain).__name__)
print()

# Step 4 - one incident
print("== Step 4: one incident ==")
incidents = pd.read_csv("data/clean/incidents.csv")
machines = pd.read_csv("data/clean/machines.csv")
incidents = incidents.merge(machines[["machine_id", "machine_type"]], on="machine_id")
first = incidents.iloc[0].to_dict()
print("input :", first["description"])
summary = chain.invoke(first)
print("output:", summary)
print()

# Step 5 - five incidents at once
print("== Step 5: five incidents with .batch() ==")
rows = incidents.head(5).to_dict(orient="records")
summaries = chain.batch(rows)
for row, text in zip(rows, summaries):
    print(f"{row['incident_id']} | {text}")
print()

# Step 6 - a robust check
print("== Step 6: check ==")
for row, text in zip(rows, summaries):
    has_id = row["machine_id"] in text
    short = len(text.split()) <= 25
    print(f"{row['incident_id']}: machine id present={has_id}  words={len(text.split())}  short={short}")

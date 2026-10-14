"""Week 11, Exercise 2 - A text-to-SQL agent with a validator and a read-only connection.

Run from the kit root, with the venv active and Ollama running:

    python week11/exercise_2_solution.py

Steps:
  1. open data/diagnostics.sqlite read-only and write the schema for the prompt;
  2. ask llama3.2 for the SQL of one question;
  3. the validator: one SELECT, no ';', no write keyword, LIMIT added;
  4. run the SQL and read the rows back with their column names;
  5. the agent loop: validate, run, retry on error, at most 3 tries;
  6. five questions, their SQL, their rows and a one-sentence answer.
"""

import re
import sqlite3

import ollama
import pandas as pd

MODEL = "llama3.2"
OPTIONS = {"temperature": 0}
DB_URI = "file:data/diagnostics.sqlite?mode=ro"
MAX_TRIES = 3
ROW_LIMIT = 50

SCHEMA = """Tables of the SQLite database:
machines(machine_id TEXT, machine_type TEXT, site TEXT, install_year INTEGER, rated_power_kw REAL)
  machine_type is one of: pump, compressor, conveyor, chiller.
  site is one of: Montreal, Quebec City, Toronto.
readings(reading_id INTEGER, machine_id TEXT, date TEXT, load_pct REAL, ambient_c REAL,
  temperature_c REAL, vibration_mm_s REAL, pressure_bar REAL, power_kw REAL, fault_next_7d INTEGER)
  one row per machine per day of 2025, date as 'YYYY-MM-DD'.
incidents(incident_id TEXT, machine_id TEXT, date TEXT, category TEXT, severity TEXT,
  downtime_hours REAL, repair_cost_cad REAL, technician TEXT, description TEXT)
  category is one of: bearing_wear, overheating, leak, electrical, sensor_fault.
  severity is one of: low, medium, high."""

SYSTEM = f"""You translate questions into SQLite SQL.
{SCHEMA}
Rules: write ONE SELECT statement only. No explanation, no markdown, no semicolon.
machine_type and site are only in the machines table: join on machine_id when you need them.
When the question says "per" or "by", put the grouping column in the SELECT list."""

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|ATTACH|DETACH|CREATE|REPLACE|VACUUM)\b", re.I
)

# Step 1 - read-only connection and schema
print("== Step 1: read-only connection ==")
con = sqlite3.connect(DB_URI, uri=True)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
print("tables:", tables)
print("schema lines in the prompt:", len(SCHEMA.splitlines()))
print()


# Step 2 - ask the model for SQL
def clean(text):
    """Remove markdown fences, stray backticks and one trailing semicolon."""
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.I).strip().strip("`").strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql


def ask_sql(question, history=()):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}, *history]
    reply = ollama.chat(model=MODEL, messages=messages, options=OPTIONS)
    return clean(reply.message.content)


print("== Step 2: the model writes SQL ==")
question = "How many high severity incidents happened in Toronto?"
sql = ask_sql(question)
print(sql)
print()


# Step 3 - the validator
def validate_sql(sql):
    """Return (safe_sql, 'ok') or (None, reason)."""
    if ";" in sql:
        return None, "contains ';' (one statement only)"
    if not re.match(r"^SELECT\b", sql, re.I):
        return None, "does not start with SELECT"
    if FORBIDDEN.search(sql):
        return None, "forbidden keyword"
    if not re.search(r"\bLIMIT\s+\d+", sql, re.I):
        sql = f"{sql} LIMIT {ROW_LIMIT}"
    return sql, "ok"


print("== Step 3: the validator ==")
for candidate in [
    sql,
    "SELECT * FROM incidents; DROP TABLE incidents",
    "DELETE FROM incidents WHERE machine_id = 'M014'",
    "SELECT machine_id FROM machines",
]:
    safe, reason = validate_sql(candidate)
    print(f"{reason:<35} <- {candidate[:60]}")
    if safe and safe != candidate:
        print(f"{'':<35}    became: {safe}")
print()


# Step 4 - run and read back
def run_sql(safe_sql):
    cursor = con.execute(safe_sql)
    columns = [d[0] for d in cursor.description]
    return columns, cursor.fetchall()


print("== Step 4: run and read back ==")
safe, _ = validate_sql(sql)
columns, rows = run_sql(safe)
print("columns:", columns)
print("rows   :", rows)
print()


# Step 5 - the agent loop
def answer_with_sql(question, max_tries=MAX_TRIES):
    """Ask, validate, run. On a problem, tell the model and try again."""
    history = []
    for attempt in range(1, max_tries + 1):
        sql = ask_sql(question, history)
        safe, reason = validate_sql(sql)
        print(f"  try {attempt}: {sql}")
        if safe is None:
            print(f"         rejected: {reason}")
            feedback = f"Rejected: {reason}. Write one SELECT statement only."
        else:
            try:
                columns, rows = run_sql(safe)
            except sqlite3.Error as error:
                print(f"         sqlite error: {error}")
                feedback = f"SQLite error: {error}. Fix the query."
            else:
                grouped = re.search(r"\b(per|by|each)\b", question, re.I)
                if grouped and (len(rows) == 1 or len(columns) == 1):
                    print("         shape check: groups asked, but no group label in the result")
                    feedback = (
                        "Your result has no group label. The question asks for one row per group. "
                        "Put the grouping column first in the SELECT list and GROUP BY it."
                    )
                else:
                    return safe, columns, rows
        history += [{"role": "assistant", "content": sql}, {"role": "user", "content": feedback}]
    return None, [], []


print("== Step 5: the loop on a hard question ==")
print("Q: What is the mean power per machine type?")
safe, columns, rows = answer_with_sql("What is the mean power per machine type?")
print("  columns:", columns)
print("  rows   :", rows)
print()


# Step 6 - five questions, read back to the user
def read_back(question, columns, rows):
    """One number: let the model phrase it. Several rows: show the table, not a sentence."""
    if len(rows) == 1 and len(columns) == 1:
        reply = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question in ONE short sentence with the number given, "
                    "rounded to one decimal. Do not compute anything.",
                },
                {"role": "user", "content": f"Question: {question}\nResult: {rows[0][0]}"},
            ],
            options=OPTIONS,
        )
        return reply.message.content.strip()
    return "\n" + pd.DataFrame(rows, columns=columns).to_string(index=False)


QUESTIONS = [
    "How many machines are there per site?",
    "Which 5 machines have the most incidents?",
    "What is the average temperature of compressors?",
    "How many high severity incidents happened in Toronto?",
    "What is the total repair cost per category?",
]

print("== Step 6: five questions ==")
for q in QUESTIONS:
    print("Q:", q)
    safe, columns, rows = answer_with_sql(q)
    if safe is None:
        print("  no valid SQL after", MAX_TRIES, "tries")
        continue
    print("  rows   :", len(rows))
    print("  answer :", read_back(q, columns, rows))
    print()
con.close()

"""Week 8, Exercise 1 - Predict the incident category from its description.

Run from the kit root, with the venv active:

    python week08/exercise_1_solution.py

Step 1: look at the text column and the five categories.
Step 2: turn the descriptions into TF-IDF vectors.
Step 3: read the vector of one incident.
Step 4: train a logistic regression and measure accuracy.
Step 5: confusion matrix and per-class report.
Step 6: the words that pull each category.
Step 7: classify a sentence a technician just wrote.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 90)

# Step 1 - the text and the labels
incidents = pd.read_csv("data/clean/incidents.csv")
print("== Step 1: the text column ==")
print(incidents[["incident_id", "category", "description"]].head(5))
print()
print(incidents["category"].value_counts())
print(f"unique descriptions: {incidents['description'].nunique()} of {len(incidents)}")
print()

X_text = incidents["description"]
y = incidents["category"]
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, stratify=y, random_state=42
)
print(f"train rows: {len(X_train_text)}   test rows: {len(X_test_text)}")
print()

# Step 2 - TF-IDF vectors
print("== Step 2: TF-IDF ==")
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print(f"vocabulary size : {len(vectorizer.vocabulary_)}")
print(f"train matrix    : {X_train.shape}   non-zero cells: {X_train.nnz}")
print(f"first 12 words  : {vectorizer.get_feature_names_out()[:12].tolist()}")
print()

# Step 3 - the vector of one incident
print("== Step 3: the vector of INC-0001 ==")
first = vectorizer.transform([incidents.loc[0, "description"]])
words = vectorizer.get_feature_names_out()
weights = sorted(zip(first.indices, first.data), key=lambda t: -t[1])
print(incidents.loc[0, "description"])
for idx, w in weights:
    print(f"  {words[idx]:<12} {w:.3f}")
print()

# Step 4 - logistic regression
print("== Step 4: logistic regression ==")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"accuracy on the test rows: {accuracy_score(y_test, pred):.3f}")
print()

# Step 5 - confusion matrix
print("== Step 5: confusion matrix ==")
labels = sorted(y.unique())
cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels)
print(cm)
print()
print(classification_report(y_test, pred, digits=2))

# Step 6 - the words that pull each category
print("== Step 6: top words per category ==")
for cls, row in zip(model.classes_, model.coef_):
    top = np.argsort(row)[::-1][:4]
    print(f"  {cls:<13} {', '.join(words[i] for i in top)}")
print()

# Step 7 - a sentence a technician just wrote
print("== Step 7: new sentences ==")
new_notes = [
    "Grinding noise near the gearbox, vibration alarm at 6.8 mm/s.",
    "Puddle of oil under the motor mount this morning.",
    "Breaker tripped again, terminal looks burnt.",
]
for note, guess in zip(new_notes, model.predict(vectorizer.transform(new_notes))):
    print(f"  {guess:<13} <- {note}")

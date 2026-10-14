"""Week 9, Exercise 1 - Chunk the 6 manuals, embed them, store them in Chroma.

Run from the kit root, with the venv active, Ollama running and
`ollama pull nomic-embed-text` done:

    python week09/exercise_1_solution.py

Step 1: split one manual on blank lines.
Step 2: merge the paragraphs into chunks of about 600 characters.
Step 3: chunk the 6 manuals, with metadata and ids.
Step 4: embed one chunk with nomic-embed-text.
Step 5: create the Chroma collection on disk and add the chunks.
Step 6: a first similarity search.
Step 7: close and reopen: the vectors are still there.

The chunking functions live in common/rag.py. Every later week reuses them,
so the store you build here is the store of the whole course.
"""

import sys
from collections import Counter
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so that `common` is importable
from common.rag import chunk_manuals, machine_type_of, merge_paragraphs, split_paragraphs  # noqa: E402

# Step 1 - paragraphs of one manual
print("== Step 1: split one manual ==")
path = Path("docs/manuals/01-pump-troubleshooting.md")
text = path.read_text(encoding="utf-8")
paragraphs = split_paragraphs(text)
print(f"{path.name}: {len(text)} characters, {len(paragraphs)} paragraphs")
for p in paragraphs[:4]:
    print(f"  [{len(p):>3} chars] {p[:60]!r}")
print()

# Step 2 - chunks of about 600 characters
print("== Step 2: merge into chunks ==")
chunks = merge_paragraphs(paragraphs, max_chars=600)
print(f"{len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"  chunk {i}: {len(c):>4} chars, starts with {c.splitlines()[0]!r}")
print()

# Step 3 - all 6 manuals with metadata
print("== Step 3: chunk the 6 manuals ==")
texts, metadatas, ids = chunk_manuals()
print(f"{len(texts)} chunks")
print("per source      :", dict(Counter(m["source"] for m in metadatas)))
print("per machine_type:", dict(Counter(m["machine_type"] for m in metadatas)))
print("first ids       :", ids[:3])
print("first metadata  :", metadatas[0])
print("machine_type_of('03-conveyor-and-chiller-troubleshooting.md') ->", machine_type_of("03-conveyor-and-chiller-troubleshooting.md"))
print()

# Step 4 - one embedding
print("== Step 4: embed one chunk ==")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query(texts[0])
print(f"vector length: {len(vector)}   first 3 values: {[round(v, 4) for v in vector[:3]]}")
print()

# Step 5 - the Chroma collection on disk
print("== Step 5: store in Chroma ==")
store = Chroma(
    collection_name="manuals",
    embedding_function=embeddings,
    persist_directory="data/chroma",
)
if store._collection.count() > 0:
    print(f"collection already has {store._collection.count()} vectors, deleting it to rebuild")
    store.delete_collection()
    store = Chroma(collection_name="manuals", embedding_function=embeddings, persist_directory="data/chroma")
store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
print(f"vectors in the collection: {store._collection.count()}")
print(f"files on disk in data/chroma: {sorted(p.name for p in Path('data/chroma').iterdir())}")
print()

# Step 6 - a first search
print("== Step 6: similarity search ==")
question = "At what vibration level must I stop a pump?"
hits = store.similarity_search_with_score(question, k=3)
for doc, distance in hits:
    print(f"  distance {distance:.3f}  {doc.metadata['source']}  [{doc.metadata['machine_type']}]")
    print(f"      {doc.page_content[:90]!r}")
print()

# Step 7 - reopen from disk
print("== Step 7: reopen the store ==")
del store
reopened = Chroma(collection_name="manuals", embedding_function=embeddings, persist_directory="data/chroma")
print(f"vectors after reopening: {reopened._collection.count()}")
one = reopened.get(ids=["04-incident-severity-and-escalation-policy.md-0"])
print(f"stored metadata of that id: {one['metadatas'][0]}")

"""Week 10, Exercise 3 - A retrieval chain with an "I don't know" guard.

Run from the kit root, with the venv active and Ollama running
(models llama3.2 and nomic-embed-text pulled):

    python week10/exercise_3_solution.py

The Chroma store is the one Week 9 built in data/chroma (collection "manuals").
If it is missing, build_or_load_manuals_store() builds it from docs/manuals.

Steps:
  1. build or load the vector store;
  2. turn it into a retriever and look at what it returns;
  3. write the grounded prompt;
  4. assemble the retrieval chain with LCEL;
  5. ask questions the manuals can answer, and one they cannot;
  6. verify every answer: a real citation, or "I don't know".
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
MANUALS = Path("docs/manuals")
PERSIST_DIR = "data/chroma"
COLLECTION = "manuals"
MAX_CHARS = 600
TOP_K = 3


def machine_type_of(file_name):
    """Metadata from the file name, same rule as Week 9."""
    if "pump" in file_name:
        return "pump"
    if "compressor" in file_name:
        return "compressor"
    if "conveyor" in file_name:
        return "conveyor,chiller"
    return "all"


def chunk_text(text, max_chars=MAX_CHARS):
    """Split on blank lines, merge paragraphs up to max_chars, keep 1 paragraph of overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], []
    for paragraph in paragraphs:
        candidate = "\n\n".join(current + [paragraph])
        if current and len(candidate) > max_chars:
            chunks.append("\n\n".join(current))
            current = [current[-1], paragraph]
        else:
            current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_manual_chunks():
    documents, ids = [], []
    for path in sorted(MANUALS.glob("*.md")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            metadata = {"source": path.name, "machine_type": machine_type_of(path.name)}
            documents.append(Document(page_content=chunk, metadata=metadata))
            ids.append(f"{path.name}-{i}")
    return documents, ids


def build_or_load_manuals_store():
    """Load data/chroma if it already holds the manuals, else embed them once."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=PERSIST_DIR)
    if store._collection.count() > 0:
        print(f"loaded existing store: {store._collection.count()} chunks")
        return store
    documents, ids = load_manual_chunks()
    store.add_documents(documents, ids=ids)
    print(f"built store: {len(documents)} chunks from {len(set(ids_source(i) for i in ids))} manuals")
    return store


def ids_source(chunk_id):
    return chunk_id.rsplit("-", 1)[0]


# Step 1 - the store
print("== Step 1: the vector store ==")
store = build_or_load_manuals_store()
print()

# Step 2 - the retriever
print("== Step 2: the retriever ==")
retriever = store.as_retriever(search_kwargs={"k": TOP_K})
question = "Above which vibration level must a compressor be stopped?"
hits = retriever.invoke(question)
for doc in hits:
    print(f"[{doc.metadata['source']}] {doc.page_content[:70]!r}")
print()

# Step 3 - the grounded prompt
print("== Step 3: the prompt ==")
GUARD = (
    "Answer only from the context. If the context does not contain the answer, "
    "say: I don't know. Cite the source file in brackets."
)
prompt = ChatPromptTemplate.from_messages(
    [("system", GUARD), ("user", "Context:\n{context}\n\nQuestion: {question}")]
)
print(prompt.messages[0].prompt.template)
print()


# Step 4 - the chain
def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


print("== Step 4: the retrieval chain ==")
llm = ChatOllama(model=MODEL, temperature=0)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))
print()

# Step 5 - questions the manuals can and cannot answer
print("== Step 5: five questions ==")
QUESTIONS = [
    "Above which vibration level must a compressor be stopped?",
    "What is the response time for a high severity incident?",
    "What does a temperature of 999 C mean in a reading?",
    "Who is the site supervisor in Toronto?",
    "What is the price of a new bearing for a pump?",
]
answers = {}
for q in QUESTIONS:
    answers[q] = chain.invoke(q)
    print("Q:", q)
    print("A:", answers[q])
    print()


# Step 6 - verify each answer
def verify(question, answer):
    """A good answer cites one of the retrieved files, or says I don't know."""
    sources = {d.metadata["source"] for d in retriever.invoke(question)}
    cited = [s for s in sources if s in answer]
    refused = "i don't know" in answer.lower()
    if refused:
        return "REFUSED"
    if cited:
        return f"CITED {cited[0]}"
    return "NO CITATION - check by hand"


print("== Step 6: verification ==")
for q, a in answers.items():
    print(f"{verify(q, a):<45} <- {q}")

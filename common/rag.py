"""The canonical NorthPeak manuals vector store, built in Week 9 and reused from Week 10 on.

Usage, from the kit root, with Ollama running and `nomic-embed-text` pulled:

    from common.rag import build_or_load_manuals_store
    store = build_or_load_manuals_store()
    hits = store.similarity_search("pump vibration above 7 mm/s", k=3)

Or on the command line:

    python common/rag.py

The recipe (every week uses exactly this one, so results are comparable):
  documents  = the 6 markdown files of docs/manuals/
  chunking   = split on blank lines into paragraphs, then merge paragraphs
               into chunks of at most about 600 characters, with 1 paragraph
               of overlap between two consecutive chunks
  metadata   = {"source": file name, "machine_type": pump | compressor |
               "conveyor,chiller" | all, inferred from the file name}
  ids        = "<source>-<i>"
  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  store      = Chroma(collection_name="manuals", persist_directory="data/chroma")
"""

from __future__ import annotations

import re
from pathlib import Path

MANUALS_DIR = Path("docs/manuals")
PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "manuals"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_CHARS = 600

# The file-name prefix says which machine type a manual is about.
MACHINE_TYPE_BY_PREFIX = {"01": "pump", "02": "compressor", "03": "conveyor,chiller"}


def machine_type_of(file_name: str) -> str:
    """Infer the machine type of a manual from its file name."""
    return MACHINE_TYPE_BY_PREFIX.get(file_name[:2], "all")


def split_paragraphs(text: str) -> list[str]:
    """Split a markdown text on blank lines. Tables and lists stay whole."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def merge_paragraphs(paragraphs: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Merge paragraphs into chunks of at most `max_chars`, 1 paragraph of overlap."""
    chunks: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join(current + [paragraph])
        if current and len(candidate) > max_chars:
            chunks.append("\n\n".join(current))
            current = [current[-1], paragraph]
        else:
            current = current + [paragraph]
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_manuals(manuals_dir: Path | str = MANUALS_DIR, max_chars: int = MAX_CHARS):
    """Return (texts, metadatas, ids) for the 6 manuals, in file-name order."""
    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for path in sorted(Path(manuals_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(merge_paragraphs(split_paragraphs(text), max_chars)):
            texts.append(chunk)
            metadatas.append({"source": path.name, "machine_type": machine_type_of(path.name)})
            ids.append(f"{path.name}-{i}")
    return texts, metadatas, ids


def get_embeddings():
    """The embedding function shared by every week."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_or_load_manuals_store(persist_directory: str = PERSIST_DIR, rebuild: bool = False):
    """Open data/chroma; build the collection from the manuals if it is empty."""
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )
    if rebuild and store._collection.count() > 0:
        store.delete_collection()
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=persist_directory,
        )
    if store._collection.count() == 0:
        texts, metadatas, ids = chunk_manuals()
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return store


if __name__ == "__main__":
    texts, metadatas, ids = chunk_manuals()
    print(f"{len(texts)} chunks from {len(set(m['source'] for m in metadatas))} manuals")
    store = build_or_load_manuals_store()
    print(f"collection '{COLLECTION_NAME}' in {PERSIST_DIR}: {store._collection.count()} vectors")

"""
RAG (Retrieval-Augmented Generation) utilities.

Documents are split into chunks, embedded locally with FastEmbed
(no API key needed) and stored in a FAISS index persisted to disk.
"""

import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_DIR = os.getenv("RAG_INDEX_DIR", "vector_store")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

_embeddings = None
_vector_store = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

    return _embeddings


def get_vector_store():
    """Return the vector store, loading it from disk if it exists."""
    global _vector_store

    if _vector_store is None and Path(INDEX_DIR, "index.faiss").exists():
        _vector_store = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True  # index is created locally by this app
        )

    return _vector_store


def load_file(file_name: str, data: bytes) -> list[Document]:
    """Load an uploaded file (PDF / TXT / MD) into LangChain documents."""
    suffix = Path(file_name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True)

        docs = loader.load()
    finally:
        os.remove(tmp_path)

    # Keep the original file name as the source instead of the temp path
    for doc in docs:
        doc.metadata["source"] = file_name

    return docs


def add_documents(docs: list[Document]) -> int:
    """Split documents into chunks, embed them and add them to the index."""
    global _vector_store

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]

    if not chunks:
        return 0

    store = get_vector_store()

    if store is None:
        _vector_store = FAISS.from_documents(chunks, get_embeddings())
    else:
        store.add_documents(chunks)

    _vector_store.save_local(INDEX_DIR)

    return len(chunks)


def list_sources() -> list[str]:
    """Return the names of all indexed files."""
    store = get_vector_store()

    if store is None:
        return []

    sources = {doc.metadata.get("source", "unknown") for doc in store.docstore._dict.values()}

    return sorted(sources)


def clear_index():
    """Delete all indexed documents."""
    global _vector_store

    _vector_store = None

    for name in ("index.faiss", "index.pkl"):
        path = Path(INDEX_DIR, name)
        if path.exists():
            path.unlink()


def search(query: str, k: int = TOP_K) -> list[Document]:
    """Return the k chunks most relevant to the query."""
    store = get_vector_store()

    if store is None:
        return []

    return store.similarity_search(query, k=k)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved chunks so the LLM can cite them."""
    parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source} (page {page + 1})" if page is not None else source
        parts.append(f"[{i}] Source: {label}\n{doc.page_content}")

    return "\n\n".join(parts)

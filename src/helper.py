from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
import torch
import os


def load_pdf_file(data: str) -> List[Dict[str, Any]]:
    """Load PDFs from `data` directory and return list of simple docs.

    Returns a list of dicts with keys `page_content` and `metadata`.
    """
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    # convert to simple dicts to avoid depending on langchain schema objects
    simple_docs = [
        {
            "page_content": d.page_content,
            "metadata": getattr(d, "metadata", {}),
        }
        for d in docs
    ]
    return simple_docs


def filter_to_minimal_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return minimal docs containing only `source` in metadata and page_content."""
    minimal_docs: List[Dict[str, Any]] = []
    for doc in docs:
        src = doc.get("metadata", {}).get("source")
        minimal_docs.append({"page_content": doc.get("page_content", ""), "metadata": {"source": src}})
    return minimal_docs


def text_split(extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split documents into chunks and return the same simple-doc format."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    # langchain text splitter expects Document-like objects with page_content and metadata
    # create lightweight objects for compatibility
    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    docs_for_split = [_Doc(d["page_content"], d.get("metadata", {})) for d in extracted_data]
    chunks = text_splitter.split_documents(docs_for_split)
    simple_chunks = [{"page_content": c.page_content, "metadata": getattr(c, "metadata", {})} for c in chunks]
    return simple_chunks


def download_hugging_face_embeddings(model_name: str | None = None):
    """Return a HuggingFaceEmbeddings instance (uses CUDA if available)."""
    model_name = model_name or os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    return embeddings
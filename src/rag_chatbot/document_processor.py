"""PDF ingestion and vector-store construction.

Reads uploaded PDF files, splits them into overlapping text chunks, tags
each chunk with source metadata, and indexes everything in a Chroma
vector store for retrieval-augmented generation.
"""

import os
import tempfile
from typing import Any

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from rag_chatbot.config import CHROMA_PERSIST_DIRECTORY, CHUNK_OVERLAP, CHUNK_SIZE
from rag_chatbot.models import get_cached_embeddings


def process_pdf_files(pdf_files: list[dict[str, Any]]) -> Chroma | None:
    """Convert a list of uploaded PDFs into a searchable Chroma vector store.

    Each PDF is written to a temporary file (required by
    ``PyPDFLoader``), loaded, split into overlapping text chunks, and
    tagged with its filename as the ``"source"`` metadata field.  All
    chunks are then embedded and inserted into a persistent Chroma
    collection.

    Args:
        pdf_files: A list of dicts, each with ``"name"`` (str) and
            ``"content"`` (bytes) keys representing an uploaded PDF.

    Returns:
        A ``Chroma`` vector store instance on success, or ``None`` if an
        error occurs during processing.
    """
    try:
        all_chunks: list = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        for pdf_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file["content"])
                temp_path = tmp.name

            try:
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            for document in documents:
                document.metadata["source"] = pdf_file["name"]

            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

        if st.session_state.embeddings is None:
            st.session_state.embeddings = get_cached_embeddings()

        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=st.session_state.embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )
        return vector_store

    except Exception as exc:
        st.error(f"PDF processing failed: {exc}")
        return None

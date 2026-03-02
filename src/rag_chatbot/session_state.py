"""Streamlit session-state initialisation.

Centralises the creation of every session-state key so that the rest of
the application can safely assume the keys exist and carry sensible
default values.
"""

import os

import streamlit as st

from rag_chatbot.config import CACHE_DIRECTORY, DEFAULT_SIMILARITY_THRESHOLD


def initialize_session_state() -> None:
    """Populate ``st.session_state`` with default values for all keys.

    This function is idempotent: it only sets a key when it does not
    already exist, so calling it on every Streamlit rerun is safe and
    inexpensive.

    Session-state keys are grouped as follows:

    **Chat:**
        ``history``, ``current_question``, ``is_follow_up``

    **Documents:**
        ``uploaded_pdfs``, ``pdf_docs``, ``pdf_processed``,
        ``vector_store``

    **Conversation chain:**
        ``conversation``

    **Embeddings:**
        ``embeddings``, ``question_embeddings``

    **Semantic cache:**
        ``question_cache``, ``cache_dir``, ``use_cache``,
        ``cache_index_loaded``, ``cache_embedding_keys``,
        ``cache_embedding_matrix``

    **UI / debug:**
        ``debug_mode``, ``similarity_threshold``
    """
    defaults = {
        # Chat
        "history": [],
        "current_question": "",
        "is_follow_up": False,
        # Documents
        "uploaded_pdfs": [],
        "pdf_docs": [],
        "pdf_processed": False,
        "vector_store": None,
        # Conversation chain
        "conversation": None,
        # Embeddings
        "embeddings": None,
        "question_embeddings": {},
        # Semantic cache
        "question_cache": {},
        "cache_index_loaded": False,
        "cache_embedding_keys": [],
        "cache_embedding_matrix": None,
        "use_cache": True,
        # UI / debug
        "debug_mode": False,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # The cache directory requires a filesystem side-effect (mkdir), so it
    # is handled separately from the pure-value defaults above.
    if "cache_dir" not in st.session_state:
        os.makedirs(CACHE_DIRECTORY, exist_ok=True)
        st.session_state.cache_dir = CACHE_DIRECTORY

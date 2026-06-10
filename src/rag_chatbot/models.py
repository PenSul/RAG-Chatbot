"""Cached LLM and embedding model resources.

Uses Streamlit's ``st.cache_resource`` decorator to ensure that
heavyweight model objects are created only once and reused across
application reruns, avoiding expensive re-initialisation.
"""

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from rag_chatbot.config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_REPEAT_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
)


@st.cache_resource(show_spinner=False)
def get_cached_embeddings(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> OllamaEmbeddings:
    """Create and cache an ``OllamaEmbeddings`` instance.

    The embedding model is loaded once and held in Streamlit's resource
    cache for the lifetime of the server process.

    Args:
        model_name: Name of the Ollama embedding model to load.

    Returns:
        A reusable ``OllamaEmbeddings`` object.
    """
    return OllamaEmbeddings(model=model_name)


@st.cache_resource(show_spinner=False)
def get_cached_llm(
    model_name: str = LLM_MODEL_NAME,
    temperature: float = LLM_TEMPERATURE,
    top_k: int = LLM_TOP_K,
    top_p: float = LLM_TOP_P,
    repeat_penalty: float = LLM_REPEAT_PENALTY,
) -> ChatOllama:
    """Create and cache a ``ChatOllama`` instance.

    The chat model is loaded once and held in Streamlit's resource cache
    for the lifetime of the server process.

    Args:
        model_name: Name of the Ollama chat model.
        temperature: Sampling temperature — lower values produce more
            deterministic output.
        top_k: Number of highest-probability tokens considered during
            sampling.
        top_p: Nucleus-sampling cumulative probability mass.
        repeat_penalty: Multiplicative penalty applied to tokens that
            have already appeared in the context.

    Returns:
        A reusable ``ChatOllama`` object.
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )

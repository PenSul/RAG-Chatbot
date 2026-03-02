"""Semantic question-answer cache.

Provides a two-tier caching strategy for previously answered questions:

1. **Exact match** — an MD5 hash of the normalised question plus the set
   of loaded PDF names is checked first (O(1) dictionary lookup).
2. **Semantic match** — if no exact hit is found, the question's embedding
   is compared against a dense matrix of cached embeddings using
   vectorised cosine similarity (O(n) but very fast in practice via
   NumPy).

Cached entries are persisted to disk as JSON files so that they survive
across application restarts.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import streamlit as st

from rag_chatbot.models import get_cached_embeddings


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


def compute_query_embedding(text: str) -> list[float]:
    """Compute the embedding vector for a piece of text.

    Lazily initialises the shared embedding model on first call and
    stores the reference in ``st.session_state.embeddings`` for reuse.

    Args:
        text: The input string to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    if st.session_state.embeddings is None:
        st.session_state.embeddings = get_cached_embeddings()
    return st.session_state.embeddings.embed_query(text)


# ---------------------------------------------------------------------------
# In-memory embedding index maintenance
# ---------------------------------------------------------------------------


def rebuild_embedding_index() -> None:
    """Rebuild the dense NumPy matrix used for fast cosine-similarity search.

    Reads all entries currently held in
    ``st.session_state.question_embeddings`` and packs them into a
    contiguous float32 array alongside a parallel list of cache keys.

    This function should be called whenever the in-memory cache is loaded
    or mutated in bulk (e.g. after loading from disk).  Incremental
    updates during normal ``save_answer_to_cache`` calls bypass this
    function for efficiency.
    """
    question_embeddings = st.session_state.question_embeddings

    if not question_embeddings:
        st.session_state.cache_embedding_keys = []
        st.session_state.cache_embedding_matrix = None
        return

    keys = list(question_embeddings.keys())
    matrix = np.array(
        [question_embeddings[k] for k in keys],
        dtype=np.float32,
    )
    st.session_state.cache_embedding_keys = keys
    st.session_state.cache_embedding_matrix = matrix


def ensure_cache_index_loaded() -> None:
    """Load cached questions and embeddings from disk into memory once.

    Scans the cache directory for ``*.embedding.json`` files, loads each
    one together with its companion ``*.json`` answer file, and populates
    the in-memory structures.  The function is guarded by the
    ``cache_index_loaded`` flag so that the expensive filesystem scan
    happens at most once per Streamlit session.
    """
    if st.session_state.cache_index_loaded:
        return

    cache_dir = st.session_state.cache_dir
    if not os.path.isdir(cache_dir):
        st.session_state.cache_index_loaded = True
        return

    for entry in os.scandir(cache_dir):
        if not entry.is_file() or not entry.name.endswith(".embedding.json"):
            continue

        cache_key = entry.name.replace(".embedding.json", "")
        answer_path = os.path.join(cache_dir, f"{cache_key}.json")

        if not os.path.exists(answer_path):
            continue

        try:
            with open(answer_path, "r", encoding="utf-8") as file_handle:
                answer_data = json.load(file_handle)
            with open(entry.path, "r", encoding="utf-8") as file_handle:
                embedding_data = json.load(file_handle)

            st.session_state.question_cache[cache_key] = answer_data
            st.session_state.question_embeddings[cache_key] = embedding_data.get(
                "embedding", [],
            )
        except (json.JSONDecodeError, KeyError, OSError):
            # Silently skip corrupt entries to keep the application running.
            continue

    rebuild_embedding_index()
    st.session_state.cache_index_loaded = True


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


def generate_cache_key(question: str) -> str:
    """Produce a deterministic cache key for a question.

    The key is an MD5 hex digest derived from the lowercased, stripped
    question concatenated with the sorted names of all currently loaded
    PDF files.  This ensures that the same question asked against a
    different document set yields a different key.

    Args:
        question: The user's question text.

    Returns:
        A 32-character hexadecimal MD5 digest string.
    """
    pdf_names = sorted(pdf["name"] for pdf in st.session_state.uploaded_pdfs)
    pdf_identifier = "-".join(pdf_names)
    composite = f"{question.lower().strip()}-{pdf_identifier}"
    return hashlib.md5(composite.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_answer_to_cache(
    question: str,
    answer: str,
    sources: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Persist a question-answer pair to both in-memory and on-disk caches.

    The dense embedding matrix is updated incrementally (a single
    ``np.vstack`` call) rather than rebuilt from scratch, keeping the
    save operation fast.

    Args:
        question: The original user question.
        answer: The generated (and post-processed) answer text.
        sources: Optional list of source-document metadata dicts, each
            containing keys such as ``"source"``, ``"page"``, and
            ``"content"``.
    """
    cache_key = generate_cache_key(question)
    question_embedding = compute_query_embedding(question.lower().strip())

    cache_entry = {
        "question": question,
        "answer": answer,
        "sources": sources if sources else [],
        "timestamp": datetime.now().isoformat(),
    }

    # --- In-memory update ------------------------------------------------
    st.session_state.question_cache[cache_key] = cache_entry
    st.session_state.question_embeddings[cache_key] = question_embedding

    embedding_vector = np.asarray(question_embedding, dtype=np.float32)
    if st.session_state.cache_embedding_matrix is None:
        st.session_state.cache_embedding_keys = [cache_key]
        st.session_state.cache_embedding_matrix = embedding_vector.reshape(1, -1)
    else:
        st.session_state.cache_embedding_keys.append(cache_key)
        st.session_state.cache_embedding_matrix = np.vstack(
            [st.session_state.cache_embedding_matrix, embedding_vector],
        )

    # --- Disk persistence ------------------------------------------------
    answer_path = os.path.join(st.session_state.cache_dir, f"{cache_key}.json")
    embedding_path = os.path.join(
        st.session_state.cache_dir,
        f"{cache_key}.embedding.json",
    )

    with open(answer_path, "w", encoding="utf-8") as file_handle:
        json.dump(cache_entry, file_handle)

    with open(embedding_path, "w", encoding="utf-8") as file_handle:
        json.dump({"embedding": question_embedding}, file_handle)


def find_semantically_similar_question(
    query: str,
    threshold: Optional[float] = None,
) -> tuple[Optional[dict], float]:
    """Search the cache for a semantically similar question.

    Computes cosine similarity between the *query* embedding and every
    cached question embedding using a single matrix-vector product,
    returning the closest match if it meets or exceeds the *threshold*.

    Args:
        query: The user's current question.
        threshold: Minimum cosine-similarity score to accept a match.
            Defaults to ``st.session_state.similarity_threshold``.

    Returns:
        A 2-tuple ``(cached_entry, similarity_score)``.  If no match
        meets the threshold, ``cached_entry`` is ``None`` and the score
        is ``0.0``.
    """
    if threshold is None:
        threshold = st.session_state.similarity_threshold

    ensure_cache_index_loaded()

    matrix = st.session_state.cache_embedding_matrix
    keys = st.session_state.cache_embedding_keys

    if matrix is None or len(keys) == 0:
        return None, 0.0

    query_embedding = np.asarray(
        compute_query_embedding(query.lower().strip()),
        dtype=np.float32,
    )

    # Vectorised cosine similarity: (M · q) / (‖rows‖ · ‖q‖)
    query_norm = np.linalg.norm(query_embedding) + 1e-12
    row_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    similarities = (matrix @ query_embedding) / (row_norms * query_norm)

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])
    best_key = keys[best_index]

    if st.session_state.debug_mode:
        cached_question = st.session_state.question_cache[best_key]["question"]
        st.write(f"Most similar cached question: {cached_question}")
        st.write(f"Similarity score: {best_score:.4f}")

    if best_score >= threshold:
        return st.session_state.question_cache[best_key], best_score

    return None, 0.0


def load_cached_answer(question: str) -> tuple[Optional[dict], float]:
    """Attempt to retrieve a cached answer for a question.

    The lookup proceeds in three stages, from cheapest to most expensive:

    1. **In-memory exact match** — the deterministic MD5 cache key is
       looked up in the session-state dictionary.
    2. **On-disk exact match** — the corresponding JSON files are read
       from the cache directory.
    3. **Semantic match** — embedding cosine similarity is computed
       against the entire cache.

    Args:
        question: The user's question.

    Returns:
        A 2-tuple ``(cached_entry, similarity_score)``.  An exact match
        returns a score of ``1.0``.  ``cached_entry`` is ``None`` when
        no match is found at any stage.
    """
    cache_key = generate_cache_key(question)

    # Stage 1: in-memory exact match
    if cache_key in st.session_state.question_cache:
        return st.session_state.question_cache[cache_key], 1.0

    # Stage 2: on-disk exact match
    answer_path = os.path.join(st.session_state.cache_dir, f"{cache_key}.json")
    embedding_path = os.path.join(
        st.session_state.cache_dir,
        f"{cache_key}.embedding.json",
    )

    if os.path.exists(answer_path) and os.path.exists(embedding_path):
        with open(answer_path, "r", encoding="utf-8") as file_handle:
            answer_data = json.load(file_handle)
        with open(embedding_path, "r", encoding="utf-8") as file_handle:
            embedding_data = json.load(file_handle)

        st.session_state.question_cache[cache_key] = answer_data
        st.session_state.question_embeddings[cache_key] = embedding_data["embedding"]
        rebuild_embedding_index()
        return answer_data, 1.0

    # Stage 3: semantic match
    return find_semantically_similar_question(question)

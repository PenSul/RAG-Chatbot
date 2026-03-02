"""Streamlit application entry point for the RAG Chatbot.

Provides the interactive web interface: a sidebar for document management
and cache settings, and a main panel with a chat-style question-and-answer
experience backed by Retrieval-Augmented Generation.
"""

import os
import time

import streamlit as st

from rag_chatbot.cache_manager import (
    load_cached_answer,
    rebuild_embedding_index,
    save_answer_to_cache,
)
from rag_chatbot.config import (
    SIMILARITY_SLIDER_DEFAULT,
    SIMILARITY_SLIDER_MAX,
    SIMILARITY_SLIDER_MIN,
    SIMILARITY_SLIDER_STEP,
    SOURCE_CONTENT_PREVIEW_LENGTH,
)
from rag_chatbot.conversation import (
    create_conversational_chain,
    process_multi_question_query,
)
from rag_chatbot.document_processor import process_pdf_files
from rag_chatbot.question_parser import detect_multiple_questions
from rag_chatbot.response_processor import format_response_with_citations
from rag_chatbot.session_state import initialize_session_state
from rag_chatbot.stream_handler import StreamHandler


# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------


def _process_uploaded_files(uploaded_files) -> None:
    """Ingest uploaded PDF files and build the vector store and chain.

    If documents have already been processed in this session the chat
    history and PDF list are reset before the new batch is ingested.

    Args:
        uploaded_files: The list of ``UploadedFile`` objects from
            Streamlit's file uploader widget.
    """
    with st.spinner("Processing documents..."):
        if st.session_state.pdf_processed:
            st.session_state.history = []
            st.session_state.uploaded_pdfs = []

        pdf_files: list[dict] = []
        for uploaded_file in uploaded_files:
            pdf_record = {
                "name": uploaded_file.name,
                "content": uploaded_file.getvalue(),
            }
            pdf_files.append(pdf_record)
            st.session_state.uploaded_pdfs.append(pdf_record)

        st.session_state.vector_store = process_pdf_files(pdf_files)

        if st.session_state.vector_store:
            st.session_state.pdf_processed = True
            st.session_state.conversation = create_conversational_chain(
                st.session_state.vector_store,
            )
            st.success(f"Processed {len(pdf_files)} PDF(s)!")


def _render_action_buttons() -> None:
    """Render the *Clear Chat* and *Reset All* buttons side by side."""
    column_clear, column_reset = st.columns(2)

    with column_clear:
        if st.session_state.pdf_processed and st.button("Clear Chat"):
            st.session_state.history = []
            if hasattr(st.session_state.conversation, "memory"):
                st.session_state.conversation.memory.clear()
            st.rerun()

    with column_reset:
        if st.session_state.pdf_processed and st.button("Reset All"):
            st.session_state.history = []
            st.session_state.uploaded_pdfs = []
            st.session_state.pdf_processed = False
            st.session_state.vector_store = None
            st.session_state.conversation = None
            st.rerun()


def _render_cache_settings() -> None:
    """Render the cache-management controls in the sidebar."""
    st.subheader("Cache Settings")

    st.session_state.use_cache = st.checkbox(
        "Use Answer Cache",
        value=st.session_state.use_cache,
    )

    st.session_state.similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=SIMILARITY_SLIDER_MIN,
        max_value=SIMILARITY_SLIDER_MAX,
        value=SIMILARITY_SLIDER_DEFAULT,
        step=SIMILARITY_SLIDER_STEP,
        help="Lower values will match more questions but might be less accurate",
    )

    if st.button("Clear Cache"):
        st.session_state.question_cache = {}
        st.session_state.question_embeddings = {}
        st.session_state.cache_embedding_keys = []
        st.session_state.cache_embedding_matrix = None
        st.session_state.cache_index_loaded = True

        for entry in os.scandir(st.session_state.cache_dir):
            if entry.is_file() and entry.name.endswith(".json"):
                try:
                    os.remove(entry.path)
                except OSError:
                    pass

        st.success("Cache cleared!")


def _render_sidebar() -> None:
    """Draw the complete sidebar: document upload, controls, cache, debug."""
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Process PDFs"):
            _process_uploaded_files(uploaded_files)

        if st.session_state.uploaded_pdfs:
            st.subheader("Loaded PDFs")
            for index, pdf in enumerate(st.session_state.uploaded_pdfs, start=1):
                st.write(f"{index}. {pdf['name']}")

        _render_action_buttons()

        if st.session_state.pdf_processed:
            _render_cache_settings()

        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)


# ---------------------------------------------------------------------------
# Chat-interface helpers
# ---------------------------------------------------------------------------


def _display_cached_response(
    cache_data: dict,
    message_placeholder,
) -> str:
    """Render a cached answer and its source documents.

    Args:
        cache_data: The cached entry containing ``"answer"`` and
            optionally ``"sources"``.
        message_placeholder: Streamlit placeholder for the answer text.

    Returns:
        The cached answer string.
    """
    message_placeholder.markdown(cache_data["answer"])

    if cache_data.get("sources"):
        with st.expander("Source Documents (Cached)"):
            for index, source in enumerate(cache_data["sources"], start=1):
                st.write(
                    f"Source {index} - "
                    f"{source.get('source', 'Unknown')} - "
                    f"Page {source.get('page', 'N/A')}:",
                )
                st.text(source.get("content", "No content available"))

    return cache_data["answer"]


def _generate_fresh_response(prompt: str, message_placeholder) -> str:
    """Invoke the LLM chain, stream the answer, and return the result.

    The response is post-processed (citations appended, question echo
    removed) and, when caching is enabled, saved to the semantic cache.

    Args:
        prompt: The user's question.
        message_placeholder: Streamlit placeholder for streamed output.

    Returns:
        The post-processed answer text.
    """
    start_time = time.time()
    stream_handler = StreamHandler(message_placeholder)

    st.session_state.is_follow_up = len(st.session_state.history) > 1
    is_multi, _ = detect_multiple_questions(prompt)

    if st.session_state.debug_mode:
        with st.expander("Debug Info"):
            st.write({
                "is_follow_up_question": st.session_state.is_follow_up,
                "is_multi_question": is_multi,
                "history_length": len(st.session_state.history),
                "original_question": prompt,
            })

    if is_multi:
        result = process_multi_question_query(
            st.session_state.conversation,
            prompt,
            stream_handler,
        )
    else:
        result = st.session_state.conversation(
            {"question": prompt},
            callbacks=[stream_handler],
        )
        result["answer"] = format_response_with_citations(
            result["answer"],
            prompt,
            result.get("source_documents", []),
        )

    processing_time = time.time() - start_time
    if st.session_state.debug_mode:
        with st.expander("Debug Info"):
            st.write(f"Processing time: {processing_time:.4f} seconds")

    stream_handler.finalize()
    processed_response = result["answer"]

    if st.session_state.debug_mode:
        with st.expander("Response Processing"):
            st.write("Raw response:")
            st.write(result["answer"])
            st.write("Processed response:")
            st.write(processed_response)

    # Display source documents and collect metadata for the cache.
    source_docs_for_cache: list[dict] = []
    if "source_documents" in result and result["source_documents"]:
        with st.expander("Source Documents"):
            for index, doc in enumerate(result["source_documents"], start=1):
                content_preview = doc.page_content
                if len(content_preview) > SOURCE_CONTENT_PREVIEW_LENGTH:
                    content_preview = (
                        content_preview[:SOURCE_CONTENT_PREVIEW_LENGTH] + "..."
                    )

                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "content": content_preview,
                }
                source_docs_for_cache.append(source_info)
                st.write(
                    f"Source {index} - {source_info['source']} - "
                    f"Page {source_info['page']}:",
                )
                st.text(source_info["content"])

    if st.session_state.use_cache:
        save_answer_to_cache(prompt, processed_response, source_docs_for_cache)

    return processed_response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the RAG Chatbot Streamlit application.

    Orchestrates session initialisation, sidebar rendering, chat-history
    display, and the request-response loop for user questions.  The
    function is called on every Streamlit rerun.
    """
    st.title("Chatbot")
    initialize_session_state()
    _render_sidebar()

    if not st.session_state.pdf_processed:
        st.info("Please upload and process PDF documents to begin chatting.")
        return

    # Render existing chat history.
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Wait for new user input.
    prompt = st.chat_input("Ask about the PDFs")
    if not prompt:
        return

    st.session_state.current_question = prompt
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # --- Cache lookup ---
            cache_data = None
            similarity_score = 0.0

            if st.session_state.use_cache:
                cache_start = time.time()
                cache_data, similarity_score = load_cached_answer(prompt)

                if cache_data and st.session_state.debug_mode:
                    retrieval_time = time.time() - cache_start
                    similar_question = cache_data.get("question", "")
                    st.info(
                        f"Cache hit! Retrieved in {retrieval_time:.4f} seconds",
                    )
                    st.info(f"Similarity score: {similarity_score:.4f}")
                    st.info(f"Similar question: '{similar_question}'")

            message_placeholder = st.empty()

            # --- Generate or retrieve answer ---
            if cache_data:
                processed_response = _display_cached_response(
                    cache_data,
                    message_placeholder,
                )
            else:
                processed_response = _generate_fresh_response(
                    prompt,
                    message_placeholder,
                )

            st.session_state.history.append(
                {"role": "assistant", "content": processed_response},
            )

        except Exception as exc:
            error_message = f"Error generating response: {exc}"
            st.error(error_message)
            st.session_state.history.append(
                {"role": "assistant", "content": error_message},
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""@file main.py
@brief Multi-PDF Chatbot with RAG using Ollama and LangChain
@details This application provides a conversational interface for querying multiple PDF documents
using Retrieval-Augmented Generation (RAG) with local LLMs through Ollama.
"""

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

import os
import re
import hashlib
import json
import time
import tempfile
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # kept for compatibility; no longer used in hot path


# ----------------------------
# Cached resources (speed)
# ----------------------------
@st.cache_resource(show_spinner=False)
def _get_embeddings_resource(model_name: str = "all-minilm:l6-v2") -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name)


@st.cache_resource(show_spinner=False)
def _get_llm_resource(
    model_name: str = "phi4-mini",
    temperature: float = 0.15,
    top_k: int = 10,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )


# ----------------------------
# Streaming handler
# ----------------------------
class StreamHandler(BaseCallbackHandler):
    """@class StreamHandler
    @brief Handles streaming of LLM responses token by token.

    @details Provides real-time display of generated tokens in the Streamlit interface.
    """

    def __init__(self, container, initial_text=""):
        """@brief Constructor for StreamHandler.
        @param container Streamlit container to display tokens
        @param initial_text Initial text to start with
        """
        self.container = container
        self.text = initial_text
        self._last_render = 0.0
        self._render_interval_s = 0.03  # throttle renders slightly for smoother UI

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """@brief Called when a new token is generated.
        @param token The newly generated token string
        """
        self.text += token
        now = time.time()
        if (now - self._last_render) >= self._render_interval_s:
            self.container.markdown(self.text + "▌")
            self._last_render = now

    def finalize(self):
        """@brief Finalizes the streaming display by removing the cursor."""
        self.container.markdown(self.text)


# ----------------------------
# Session state initialization
# ----------------------------
def init_session_state():
    """@brief Initializes Streamlit session state variables.

    @details Sets up all necessary session state variables with default values.
    """
    if "history" not in st.session_state:
        st.session_state.history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "is_follow_up" not in st.session_state:
        st.session_state.is_follow_up = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = []
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
    if "question_cache" not in st.session_state:
        st.session_state.question_cache = {}
    if "cache_dir" not in st.session_state:
        cache_dir = "./cache"
        os.makedirs(cache_dir, exist_ok=True)
        st.session_state.cache_dir = cache_dir
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "question_embeddings" not in st.session_state:
        st.session_state.question_embeddings = {}

    # Fast similarity-search structures
    if "cache_index_loaded" not in st.session_state:
        st.session_state.cache_index_loaded = False
    if "cache_embedding_keys" not in st.session_state:
        st.session_state.cache_embedding_keys = []
    if "cache_embedding_matrix" not in st.session_state:
        st.session_state.cache_embedding_matrix = None

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = 0.85

    # Persist cache toggle across reruns
    if "use_cache" not in st.session_state:
        st.session_state.use_cache = True


# ----------------------------
# Embeddings + cache helpers
# ----------------------------
def get_embedding(text):
    """@brief Gets embedding vector for a text string.

    @param text Text to embed
    @return Embedding vector
    """
    # Keep original session_state behavior, but initialize via st.cache_resource for speed.
    if st.session_state.embeddings is None:
        st.session_state.embeddings = _get_embeddings_resource("all-minilm:l6-v2")

    return st.session_state.embeddings.embed_query(text)


def _rebuild_cache_embedding_matrix():
    """Builds a dense matrix for fast similarity search from question_embeddings."""
    if not st.session_state.question_embeddings:
        st.session_state.cache_embedding_keys = []
        st.session_state.cache_embedding_matrix = None
        return

    keys = list(st.session_state.question_embeddings.keys())
    # Ensure float32 for speed/memory
    mat = np.array(
        [st.session_state.question_embeddings[k] for k in keys], dtype=np.float32
    )
    st.session_state.cache_embedding_keys = keys
    st.session_state.cache_embedding_matrix = mat


def _ensure_cache_index_loaded():
    """Load cached Q/A + embeddings from disk once per session for fast semantic search."""
    if st.session_state.cache_index_loaded:
        return

    cache_dir = st.session_state.cache_dir
    if not os.path.isdir(cache_dir):
        st.session_state.cache_index_loaded = True
        return

    # Load all embeddings + corresponding cache entries
    for entry in os.scandir(cache_dir):
        if not entry.is_file():
            continue
        if not entry.name.endswith(".embedding.json"):
            continue

        cache_key = entry.name.replace(".embedding.json", "")
        cache_file = os.path.join(cache_dir, f"{cache_key}.json")
        if not os.path.exists(cache_file):
            continue

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            with open(entry.path, "r") as f:
                embedding_data = json.load(f)

            st.session_state.question_cache[cache_key] = cache_data
            st.session_state.question_embeddings[cache_key] = embedding_data.get(
                "embedding", []
            )
        except Exception:
            # Corrupt cache entries are skipped (keeps app running smoothly)
            continue

    _rebuild_cache_embedding_matrix()
    st.session_state.cache_index_loaded = True


def get_cache_key(question):
    """@brief Generates a hash key for caching question-answer pairs.

    @param question The user's question
    @return MD5 hash string representing the cache key

    @details Creates a unique key based on the question and currently loaded PDFs.
    """
    pdf_files_str = "-".join(
        sorted([pdf["name"] for pdf in st.session_state.uploaded_pdfs])
    )
    key_data = f"{question.lower().strip()}-{pdf_files_str}"
    return hashlib.md5(key_data.encode()).hexdigest()


def save_to_cache(question, answer, sources=None):
    """@brief Saves a question-answer pair to the cache.

    @param question The user's question
    @param answer The generated answer
    @param sources List of source documents (optional)

    @details Stores both in memory and on disk for persistence.
    """
    cache_key = get_cache_key(question)
    question_embedding = get_embedding(question.lower().strip())

    cache_data = {
        "question": question,
        "answer": answer,
        "sources": sources if sources else [],
        "timestamp": datetime.now().isoformat(),
    }

    # Save to in-memory cache
    st.session_state.question_cache[cache_key] = cache_data
    st.session_state.question_embeddings[cache_key] = question_embedding

    # Update fast-search matrix incrementally (avoid full rebuild on every save)
    emb_vec = np.asarray(question_embedding, dtype=np.float32)
    if st.session_state.cache_embedding_matrix is None:
        st.session_state.cache_embedding_keys = [cache_key]
        st.session_state.cache_embedding_matrix = emb_vec.reshape(1, -1)
    else:
        st.session_state.cache_embedding_keys.append(cache_key)
        st.session_state.cache_embedding_matrix = np.vstack(
            [st.session_state.cache_embedding_matrix, emb_vec]
        )

    cache_file = os.path.join(st.session_state.cache_dir, f"{cache_key}.json")
    embedding_file = os.path.join(
        st.session_state.cache_dir, f"{cache_key}.embedding.json"
    )

    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    with open(embedding_file, "w") as f:
        json.dump({"embedding": question_embedding}, f)


def find_similar_question(query, threshold=None):
    """@brief Finds semantically similar questions in the cache.

    @param query The user's question
    @param threshold Similarity threshold (optional)
    @return Most similar cached question or None

    @details Uses cosine similarity between embeddings to find similar questions.
    """
    if threshold is None:
        threshold = st.session_state.similarity_threshold

    _ensure_cache_index_loaded()

    mat = st.session_state.cache_embedding_matrix
    keys = st.session_state.cache_embedding_keys

    if mat is None or len(keys) == 0:
        return None, 0.0

    query_embedding = np.asarray(get_embedding(query.lower().strip()), dtype=np.float32)

    # Vectorized cosine similarity: (mat @ q) / (||mat|| * ||q||)
    q_norm = np.linalg.norm(query_embedding) + 1e-12
    mat_norms = np.linalg.norm(mat, axis=1) + 1e-12
    sims = (mat @ query_embedding) / (mat_norms * q_norm)

    idx = int(np.argmax(sims))
    max_sim = float(sims[idx])
    max_sim_key = keys[idx]

    if st.session_state.debug_mode:
        st.write(
            f"Most similar question: {st.session_state.question_cache[max_sim_key]['question']}"
        )
        st.write(f"Similarity score: {max_sim:.4f}")

    if max_sim >= threshold:
        return st.session_state.question_cache[max_sim_key], max_sim

    return None, 0.0


def load_from_cache(question):
    """@brief Attempts to load an answer from the cache using semantic similarity.

    @param question The user's question
    @return Cached data if found, None otherwise

    @details First tries exact match, then semantic similarity search.
    """
    cache_key = get_cache_key(question)

    # In-memory exact match
    if cache_key in st.session_state.question_cache:
        return st.session_state.question_cache[cache_key], 1.0

    # Disk exact match
    cache_file = os.path.join(st.session_state.cache_dir, f"{cache_key}.json")
    embedding_file = os.path.join(
        st.session_state.cache_dir, f"{cache_key}.embedding.json"
    )

    if os.path.exists(cache_file) and os.path.exists(embedding_file):
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        with open(embedding_file, "r") as f:
            embedding_data = json.load(f)

        st.session_state.question_cache[cache_key] = cache_data
        st.session_state.question_embeddings[cache_key] = embedding_data["embedding"]

        # keep matrix in sync
        _rebuild_cache_embedding_matrix()
        return cache_data, 1.0

    # Semantic match
    similar_question, similarity = find_similar_question(question)
    return similar_question, similarity


# ----------------------------
# Question parsing + response processing
# ----------------------------
def detect_multiple_questions(question):
    """@brief Detects if the input contains multiple questions.

    @param question The input text
    @return Tuple of (is_multi_question, list_of_questions)

    @details Uses heuristic rules to identify multiple questions.
    """
    parts = re.split(r"\?", question)
    question_words = [
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "whose",
        "whom",
        "is",
        "are",
        "was",
        "were",
        "will",
        "can",
        "could",
        "should",
        "would",
        "do",
        "does",
        "did",
    ]
    questions = []
    for part in parts:
        if part.strip():
            words = part.lower().split()
            if any(word in question_words for word in words):
                questions.append(part.strip() + "?")
    if len(questions) > 1:
        return True, questions
    return False, [question]


def process_pdfs(pdf_files):
    """@brief Processes multiple PDF files into a combined vector store.

    @param pdf_files List of uploaded PDF files
    @return Chroma vector store instance or None if processing fails

    @details Handles text extraction, chunking, and embedding generation.
    """
    try:
        all_chunks = []

        # Create splitter once (minor speedup)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100, length_function=len
        )

        for pdf_file in pdf_files:
            # Safer + avoids name collisions; still writes to disk because PyPDFLoader expects a path
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

            for doc in documents:
                doc.metadata["source"] = pdf_file["name"]

            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

        if st.session_state.embeddings is None:
            st.session_state.embeddings = _get_embeddings_resource("all-minilm:l6-v2")

        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=st.session_state.embeddings,
            persist_directory="./chroma_db",
        )
        return vector_store
    except Exception as e:
        st.error(f"PDF processing failed: {str(e)}")
        return None


def post_process_response(response, question, source_docs=None):
    """@brief Cleans and formats the LLM response with source citations.

    @param response The raw LLM response
    @param question The original question
    @param source_docs List of source documents (optional)
    @return Cleaned response string with citations

    @details Removes question repetition and adds source citations.
    """
    if response.strip() == "I don't have enough information to answer this question.":
        return response

    if "don't have enough information" in response.lower():
        parts = re.split(
            r"(I don\'t have enough information to answer [^.]*\.)",
            response,
            flags=re.IGNORECASE,
        )
        if len(parts) > 1:
            cleaned_parts = []
            for part in parts:
                if part.strip():
                    if "don't have enough information" in part.lower():
                        cleaned_parts.append(part.strip())
                    else:
                        cleaned_part = clean_question_repetition(part, question)
                        if cleaned_part.strip():
                            cleaned_parts.append(cleaned_part.strip())
            response = " ".join(cleaned_parts)

    response = clean_question_repetition(response, question)

    if source_docs and len(source_docs) > 0:
        unique_sources = {}
        for doc in source_docs:
            source_name = doc.metadata.get("source", "Unknown Document")
            page_num = doc.metadata.get("page", "N/A")
            key = f"{source_name}-{page_num}"
            if key not in unique_sources:
                unique_sources[key] = {"source": source_name, "page": page_num}

        if unique_sources:
            citation_text = "\n\n\nSources:\n"
            for i, (_, source) in enumerate(unique_sources.items()):
                citation_text += f"{i + 1}. {source['source']}, page {source['page']}\n"
            response += citation_text

    return response


def clean_question_repetition(text, question):
    """@brief Removes question repetition from response text.

    @param text The text to clean
    @param question The original question
    @return Cleaned text

    @details Uses word overlap detection to identify redundant question restatements.
    """
    question_words = set(re.findall(r"\b\w+\b", question.lower()))
    question_words = {
        w
        for w in question_words
        if len(w) > 3
        and w
        not in {
            "what",
            "when",
            "where",
            "who",
            "how",
            "does",
            "is",
            "are",
            "the",
            "and",
            "that",
            "this",
            "with",
            "for",
            "from",
            "have",
            "had",
        }
    }
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences and len(sentences) > 1:
        first_sentence_words = set(re.findall(r"\b\w+\b", sentences[0].lower()))
        overlap = len(question_words.intersection(first_sentence_words)) / max(
            1, len(question_words)
        )
        if overlap > 0.5:
            return " ".join(sentences[1:])
    return text


def setup_conversation(vector_store):
    """@brief Configures the conversational retrieval chain.

    @param vector_store Chroma vector store for document retrieval
    @return Configured ConversationalRetrievalChain

    @details Sets up LLM with optimized parameters and custom prompts.
    """
    # Cached LLM object reduces repeated initialization overhead across reruns
    llm = _get_llm_resource(
        model_name="phi4-mini",
        temperature=0.15,
        top_k=10,
        top_p=0.9,
        repeat_penalty=1.1,
    )

    condense_question_prompt = PromptTemplate.from_template(
        """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Chat History: {chat_history}
    Follow-Up Question: {question}
    Standalone Question:"""
    )

    qa_prompt = PromptTemplate.from_template(
        """
    Answer questions about documents based ONLY on the following context:
    {context}
    Question: {question}
    Answer:"""
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        rephrase_question=False,
        return_source_documents=True,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        output_key="answer",
        verbose=True,
    )


def handle_multi_question_query(conversation, prompt, stream_handler=None):
    """@brief Processes queries containing multiple questions with source citations.

    @param conversation The conversation chain
    @param prompt The user's input prompt
    @param stream_handler Optional streaming callback
    @return Combined response dictionary with citations
    """
    is_multi, questions = detect_multiple_questions(prompt)
    if not is_multi or len(questions) <= 1:
        result = conversation(
            {"question": prompt}, callbacks=[stream_handler] if stream_handler else None
        )
        if "source_documents" in result:
            result["answer"] = post_process_response(
                result["answer"], prompt, result["source_documents"]
            )
        return result

    responses = []
    all_sources = []
    for q in questions:
        result = conversation({"question": q}, callbacks=None)
        processed_answer = post_process_response(
            result["answer"], q, result.get("source_documents", [])
        )
        responses.append((q, processed_answer))
        if "source_documents" in result:
            all_sources.extend(result["source_documents"])

    combined_answer = ""
    for q, answer in responses:
        q_text = q.strip()
        if not q_text.endswith("?"):
            q_text += "?"
        combined_answer += f"{q_text}\n{answer}\n\n"

    # Streaming a pre-built string char-by-char is very slow in Streamlit.
    # Batch updates keep the "feel" while being much faster.
    if stream_handler:
        chunk_size = 200
        for i in range(0, len(combined_answer), chunk_size):
            stream_handler.on_llm_new_token(combined_answer[i : i + chunk_size])
        stream_handler.finalize()

    return {"answer": combined_answer.strip(), "source_documents": all_sources}


# ----------------------------
# Main app
# ----------------------------
def main():
    """@brief Main application entry point.

    @details Sets up the Streamlit interface and handles user interactions.
    """
    st.title("Chatbot")
    init_session_state()

    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True
        )

        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing documents..."):
                if st.session_state.pdf_processed:
                    st.session_state.history = []
                    st.session_state.uploaded_pdfs = []

                pdf_files = []
                for uploaded_file in uploaded_files:
                    pdf_file = {
                        "name": uploaded_file.name,
                        "content": uploaded_file.getvalue(),
                    }
                    pdf_files.append(pdf_file)
                    st.session_state.uploaded_pdfs.append(pdf_file)

                st.session_state.vector_store = process_pdfs(pdf_files)
                if st.session_state.vector_store:
                    st.session_state.pdf_processed = True
                    st.session_state.conversation = setup_conversation(
                        st.session_state.vector_store
                    )
                    st.success(f"Processed {len(pdf_files)} PDF(s)!")

        if st.session_state.uploaded_pdfs:
            st.subheader("Loaded PDFs")
            for i, pdf in enumerate(st.session_state.uploaded_pdfs):
                st.write(f"{i + 1}. {pdf['name']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.pdf_processed and st.button("Clear Chat"):
                st.session_state.history = []
                if hasattr(st.session_state.conversation, "memory"):
                    st.session_state.conversation.memory.clear()
                st.rerun()
        with col2:
            if st.session_state.pdf_processed and st.button("Reset All"):
                st.session_state.history = []
                st.session_state.uploaded_pdfs = []
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.conversation = None
                st.rerun()

        # Cache management
        if st.session_state.pdf_processed:
            st.subheader("Cache Settings")
            st.session_state.use_cache = st.checkbox(
                "Use Answer Cache", value=st.session_state.use_cache
            )

            st.session_state.similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Lower values will match more questions but might be less accurate",
            )

            if st.button("Clear Cache"):
                st.session_state.question_cache = {}
                st.session_state.question_embeddings = {}
                st.session_state.cache_embedding_keys = []
                st.session_state.cache_embedding_matrix = None
                st.session_state.cache_index_loaded = True  # we've just cleared it

                for entry in os.scandir(st.session_state.cache_dir):
                    if entry.is_file() and entry.name.endswith(".json"):
                        try:
                            os.remove(entry.path)
                        except OSError:
                            pass
                st.success("Cache cleared!")

        # Debug toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)

    # Chat Interface
    if st.session_state.pdf_processed:
        # Render chat history
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("Ask about the PDFs"):
            st.session_state.current_question = prompt
            st.session_state.history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    cache_hit = False
                    cache_data = None
                    similarity_score = 0.0
                    similar_question = ""

                    # Cache lookup (if enabled)
                    if st.session_state.use_cache:
                        start_time = time.time()
                        cache_data, similarity_score = load_from_cache(prompt)
                        if cache_data:
                            cache_hit = True
                            similar_question = cache_data.get("question", "")
                            if st.session_state.debug_mode:
                                retrieval_time = time.time() - start_time
                                st.info(
                                    f"Cache hit! Retrieved in {retrieval_time:.4f} seconds"
                                )
                                st.info(f"Similarity score: {similarity_score:.4f}")
                                st.info(f"Similar question: '{similar_question}'")

                    message_placeholder = st.empty()

                    if cache_hit:
                        message_placeholder.markdown(cache_data["answer"])
                        if cache_data.get("sources"):
                            with st.expander("Source Documents (Cached)"):
                                for i, source in enumerate(cache_data["sources"]):
                                    st.write(
                                        f"Source {i + 1} - {source.get('source', 'Unknown')} - "
                                        f"Page {source.get('page', 'N/A')}:"
                                    )
                                    st.text(
                                        source.get("content", "No content available")
                                    )

                        processed_response = cache_data["answer"]
                    else:
                        start_time = time.time()
                        stream_handler = StreamHandler(message_placeholder)

                        st.session_state.is_follow_up = (
                            len(st.session_state.history) > 1
                        )
                        is_multi, _ = detect_multiple_questions(prompt)

                        if st.session_state.debug_mode:
                            with st.expander("Debug Info"):
                                st.write(
                                    {
                                        "is_follow_up_question": st.session_state.is_follow_up,
                                        "is_multi_question": is_multi,
                                        "history_length": len(st.session_state.history),
                                        "original_question": prompt,
                                    }
                                )

                        if is_multi:
                            result = handle_multi_question_query(
                                st.session_state.conversation, prompt, stream_handler
                            )
                        else:
                            result = st.session_state.conversation(
                                {"question": prompt}, callbacks=[stream_handler]
                            )
                            result["answer"] = post_process_response(
                                result["answer"],
                                prompt,
                                result.get("source_documents", []),
                            )

                        processing_time = time.time() - start_time
                        if st.session_state.debug_mode:
                            with st.expander("Debug Info"):
                                st.write(
                                    f"Processing time: {processing_time:.4f} seconds"
                                )

                        stream_handler.finalize()
                        raw_response = result["answer"]
                        processed_response = raw_response

                        if st.session_state.debug_mode:
                            with st.expander("Response Processing"):
                                st.write("Raw response:")
                                st.write(raw_response)
                                st.write("Processed response:")
                                st.write(processed_response)

                        # Prepare source documents for cache and display
                        source_docs_for_cache = []
                        if "source_documents" in result and result["source_documents"]:
                            with st.expander("Source Documents"):
                                for i, doc in enumerate(result["source_documents"]):
                                    source_info = {
                                        "source": doc.metadata.get("source", "Unknown"),
                                        "page": doc.metadata.get("page", "N/A"),
                                        "content": (
                                            doc.page_content[:300] + "..."
                                            if len(doc.page_content) > 300
                                            else doc.page_content
                                        ),
                                    }
                                    source_docs_for_cache.append(source_info)
                                    st.write(
                                        f"Source {i + 1} - {source_info['source']} - "
                                        f"Page {source_info['page']}:"
                                    )
                                    st.text(source_info["content"])

                        # Save to cache (if enabled)
                        if st.session_state.use_cache:
                            save_to_cache(
                                prompt, processed_response, source_docs_for_cache
                            )

                    # Add to chat history
                    st.session_state.history.append(
                        {"role": "assistant", "content": processed_response}
                    )

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.history.append(
                        {"role": "assistant", "content": error_msg}
                    )
    else:
        st.info("Please upload and process PDF documents to begin chatting.")


if __name__ == "__main__":
    main()

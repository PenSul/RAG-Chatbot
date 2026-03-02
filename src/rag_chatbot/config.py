"""Centralised configuration constants for the RAG Chatbot application.

All tunable parameters — model names, chunking settings, cache behaviour,
prompt templates, and heuristic thresholds — are collected here so that
they can be adjusted in a single place without touching business logic.
"""

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "all-minilm:l6-v2"

LLM_MODEL_NAME: str = "phi4-mini"
LLM_TEMPERATURE: float = 0.15
LLM_TOP_K: int = 10
LLM_TOP_P: float = 0.9
LLM_REPEAT_PENALTY: float = 1.1

# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
RETRIEVER_RESULT_COUNT: int = 3

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------

CACHE_DIRECTORY: str = "./cache"
CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD: float = 0.85
SIMILARITY_SLIDER_DEFAULT: float = 0.95
SIMILARITY_SLIDER_MIN: float = 0.50
SIMILARITY_SLIDER_MAX: float = 0.99
SIMILARITY_SLIDER_STEP: float = 0.01

# ---------------------------------------------------------------------------
# Streaming / UI
# ---------------------------------------------------------------------------

STREAM_RENDER_INTERVAL_SECONDS: float = 0.03
MULTI_QUESTION_STREAM_CHUNK_SIZE: int = 200
SOURCE_CONTENT_PREVIEW_LENGTH: int = 300

# ---------------------------------------------------------------------------
# Question-detection heuristics
# ---------------------------------------------------------------------------

QUESTION_INDICATOR_WORDS: frozenset = frozenset({
    "what", "why", "how", "when", "where", "who", "which",
    "whose", "whom", "is", "are", "was", "were", "will",
    "can", "could", "should", "would", "do", "does", "did",
})

COMMON_STOP_WORDS: frozenset = frozenset({
    "what", "when", "where", "who", "how", "does", "is", "are",
    "the", "and", "that", "this", "with", "for", "from", "have", "had",
})

MIN_MEANINGFUL_WORD_LENGTH: int = 3
QUESTION_OVERLAP_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CONDENSE_QUESTION_TEMPLATE: str = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
Chat History: {chat_history}
Follow-Up Question: {question}
Standalone Question:"""

QA_PROMPT_TEMPLATE: str = """
Answer questions about documents based ONLY on the following context:
{context}
Question: {question}
Answer:"""

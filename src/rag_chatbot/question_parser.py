"""Heuristic question detection and response text cleaning.

Provides utilities for splitting compound user inputs into individual
questions and for removing redundant question restatements that LLMs
sometimes prepend to their answers.
"""

import re

from rag_chatbot.config import (
    COMMON_STOP_WORDS,
    MIN_MEANINGFUL_WORD_LENGTH,
    QUESTION_INDICATOR_WORDS,
    QUESTION_OVERLAP_THRESHOLD,
)


def detect_multiple_questions(text: str) -> tuple[bool, list[str]]:
    """Determine whether *text* contains more than one question.

    The detection heuristic splits on ``?`` characters and retains any
    fragment whose words overlap with a predefined set of English
    question-indicator words (e.g. *what*, *why*, *how*).

    Args:
        text: The raw user input.

    Returns:
        A 2-tuple ``(is_multi_question, questions)`` where
        *is_multi_question* is ``True`` when two or more questions are
        detected and *questions* is a list of the individual question
        strings.  When only one (or zero) questions are found, the
        original *text* is returned as the sole list element.
    """
    fragments = re.split(r"\?", text)
    questions: list[str] = []

    for fragment in fragments:
        stripped = fragment.strip()
        if not stripped:
            continue
        words = stripped.lower().split()
        if any(word in QUESTION_INDICATOR_WORDS for word in words):
            questions.append(stripped + "?")

    if len(questions) > 1:
        return True, questions
    return False, [text]


def remove_question_repetition(text: str, question: str) -> str:
    """Remove a leading sentence that merely restates the user's question.

    Many LLMs echo the question before answering.  This function detects
    that pattern by measuring the word-level overlap between the first
    sentence of *text* and the original *question*.  If the overlap ratio
    exceeds ``QUESTION_OVERLAP_THRESHOLD``, the first sentence is
    dropped.

    Only "meaningful" words — those longer than
    ``MIN_MEANINGFUL_WORD_LENGTH`` characters and not in
    ``COMMON_STOP_WORDS`` — are considered when computing overlap.

    Args:
        text: The LLM-generated response text.
        question: The original user question.

    Returns:
        The response text with the redundant leading sentence removed, or
        the original text unchanged if no repetition was detected.
    """
    question_words = {
        word
        for word in re.findall(r"\b\w+\b", question.lower())
        if len(word) > MIN_MEANINGFUL_WORD_LENGTH and word not in COMMON_STOP_WORDS
    }

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences or len(sentences) <= 1:
        return text

    first_sentence_words = set(re.findall(r"\b\w+\b", sentences[0].lower()))
    overlap_ratio = len(question_words & first_sentence_words) / max(
        1,
        len(question_words),
    )

    if overlap_ratio > QUESTION_OVERLAP_THRESHOLD:
        return " ".join(sentences[1:])

    return text

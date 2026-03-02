"""Conversational retrieval chain setup and multi-question orchestration.

Wraps LangChain's ``ConversationalRetrievalChain`` with custom prompts
for question condensation and document-grounded answer generation, and
provides a handler that automatically splits compound questions into
individually answered sub-queries.
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from rag_chatbot.config import (
    CONDENSE_QUESTION_TEMPLATE,
    MULTI_QUESTION_STREAM_CHUNK_SIZE,
    QA_PROMPT_TEMPLATE,
    RETRIEVER_RESULT_COUNT,
)
from rag_chatbot.models import get_cached_llm
from rag_chatbot.question_parser import detect_multiple_questions
from rag_chatbot.response_processor import format_response_with_citations
from rag_chatbot.stream_handler import StreamHandler


def create_conversational_chain(vector_store) -> ConversationalRetrievalChain:
    """Build a LangChain conversational retrieval chain.

    The chain combines four components:

    * A locally hosted LLM (via Ollama) for answer generation.
    * A Chroma vector-store retriever for context retrieval.
    * A conversation buffer memory so the LLM can refer to earlier
      exchanges within the same session.
    * Custom prompt templates for question condensation and
      document-grounded QA.

    Args:
        vector_store: A ``Chroma`` vector store loaded with document
            chunks.

    Returns:
        A fully configured ``ConversationalRetrievalChain`` ready for
        invocation.
    """
    llm = get_cached_llm()

    condense_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    qa_prompt = PromptTemplate.from_template(QA_PROMPT_TEMPLATE)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_RESULT_COUNT},
        ),
        memory=memory,
        rephrase_question=False,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        output_key="answer",
        verbose=True,
    )


def process_multi_question_query(
    conversation: ConversationalRetrievalChain,
    prompt: str,
    stream_handler: StreamHandler | None = None,
) -> dict:
    """Handle a user prompt that may contain multiple questions.

    If only a single question is detected the prompt is forwarded to the
    conversation chain as-is (with optional streaming).  When multiple
    questions are found, each is answered individually; the answers are
    concatenated and — if a *stream_handler* is provided — the combined
    text is pushed to the UI in fixed-size chunks for visual consistency.

    Args:
        conversation: The active ``ConversationalRetrievalChain``.
        prompt: The user's raw input text.
        stream_handler: Optional ``StreamHandler`` for real-time display.

    Returns:
        A dict with at least ``"answer"`` (str) and
        ``"source_documents"`` (list) keys.
    """
    is_multi, questions = detect_multiple_questions(prompt)

    if not is_multi or len(questions) <= 1:
        result = conversation(
            {"question": prompt},
            callbacks=[stream_handler] if stream_handler else None,
        )
        if "source_documents" in result:
            result["answer"] = format_response_with_citations(
                result["answer"],
                prompt,
                result["source_documents"],
            )
        return result

    # Answer each sub-question independently.
    individual_answers: list[tuple[str, str]] = []
    all_source_documents: list = []

    for question in questions:
        result = conversation({"question": question}, callbacks=None)
        processed_answer = format_response_with_citations(
            result["answer"],
            question,
            result.get("source_documents", []),
        )
        individual_answers.append((question, processed_answer))
        if "source_documents" in result:
            all_source_documents.extend(result["source_documents"])

    # Build the combined answer text.
    combined_parts: list[str] = []
    for question_text, answer_text in individual_answers:
        normalised_question = question_text.strip()
        if not normalised_question.endswith("?"):
            normalised_question += "?"
        combined_parts.append(f"{normalised_question}\n{answer_text}")

    combined_answer = "\n\n".join(combined_parts)

    # Stream the pre-built answer in chunks for visual consistency.
    if stream_handler:
        chunk_size = MULTI_QUESTION_STREAM_CHUNK_SIZE
        for offset in range(0, len(combined_answer), chunk_size):
            stream_handler.on_llm_new_token(
                combined_answer[offset:offset + chunk_size],
            )
        stream_handler.finalize()

    return {
        "answer": combined_answer.strip(),
        "source_documents": all_source_documents,
    }

# RAG Chatbot

A multi-PDF conversational chatbot powered by **Retrieval-Augmented Generation (RAG)**, running entirely on your local machine with [Ollama](https://ollama.ai/) and [LangChain](https://www.langchain.com/).

Upload one or more PDF documents, and the chatbot will answer questions grounded in their content — with source citations, semantic answer caching, and real-time streaming.

---

## Features

- **Multi-PDF ingestion** — upload and query several documents at once.
- **Local LLM inference** — no API keys, no cloud services; everything runs on-device via Ollama.
- **Streaming responses** — tokens are rendered in real time for a responsive chat experience.
- **Semantic answer cache** — repeated or similar questions are served instantly from an embedding-based cache with configurable similarity threshold.
- **Multi-question detection** — compound questions are automatically split and answered individually.
- **Source citations** — every answer references the originating document and page number.
- **Debug mode** — toggle an on-screen diagnostic panel for retrieval timing, similarity scores, and processing details.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or later |
| **Ollama** | Installed and running locally — see [ollama.ai](https://ollama.ai/) |
| **LLM model** | Pulled into Ollama (default: `phi4-mini`) |
| **Embedding model** | Pulled into Ollama (default: `all-minilm:l6-v2`) |

Pull the required models before first use:

```bash
ollama pull phi4-mini
ollama pull all-minilm:l6-v2
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/PenSul/RAG-Chatbot.git
   cd rag-chatbot
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install the package in editable mode:**

   ```bash
   pip install -e .
   ```

   This installs all dependencies listed in `pyproject.toml` and makes the `rag_chatbot` package importable.

---

## Usage

1. **Make sure Ollama is running** (e.g. `ollama serve` in a separate terminal).

2. **Start the application:**

   ```bash
   streamlit run src/rag_chatbot/app.py
   ```

3. **Open the URL** printed to the terminal (typically `http://localhost:8501`).

4. **Upload one or more PDFs** via the sidebar, click **Process PDFs**, and start chatting.

---

## Project Structure

```
src/rag_chatbot/
├── __init__.py              # Package metadata
├── app.py                   # Streamlit UI entry point
├── cache_manager.py         # Semantic question-answer cache (disk + in-memory)
├── config.py                # Centralised constants and prompt templates
├── conversation.py          # LangChain conversational chain setup
├── document_processor.py    # PDF loading, chunking, and vector-store creation
├── models.py                # Cached Ollama LLM and embedding resources
├── question_parser.py       # Multi-question detection and text cleaning
├── response_processor.py    # Post-processing and citation formatting
├── session_state.py         # Streamlit session-state initialisation
└── stream_handler.py        # Token-by-token streaming callback
```

---

## Configuration

All tunable parameters live in [`src/rag_chatbot/config.py`](src/rag_chatbot/config.py), including model names, chunking sizes, cache paths, similarity thresholds, and prompt templates. Modify that single file to adapt the chatbot to different models or retrieval strategies.

---

## License

This project is licensed under the [MIT License](LICENSE).

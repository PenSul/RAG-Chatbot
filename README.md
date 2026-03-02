## Prerequisites

1. **Python 3.8+**: Ensure Python 3.8 or later is installed.
2. **Ollama**: Install and set up Ollama to run local LLMs. Follow the instructions [here](https://ollama.ai/).
3. **Required Libraries**: Install the necessary Python packages.

## Installation

1. Navigate to the project directory:
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Streamlit application:
   ```bash
   streamlit run main.py
   ```
2. Open the provided URL in your browser (typically `http://localhost:8501`).
3. Upload PDF documents and start chatting with the bot!

## Notes

- Ensure Ollama is running and the required LLM (e.g., `phi4-mini:latest` as used in the code) is downloaded.
- Debug mode can be enabled for additional insights into the bot's processing.

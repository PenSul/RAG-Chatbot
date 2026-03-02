"""Streaming callback handler for real-time LLM response display.

Provides a LangChain callback that pushes tokens into a Streamlit
container as they are generated, creating a typewriter-style effect.
Rendering is throttled to avoid flooding the browser with rapid DOM
updates.
"""

import time

from langchain.callbacks.base import BaseCallbackHandler

from rag_chatbot.config import STREAM_RENDER_INTERVAL_SECONDS


class StreamHandler(BaseCallbackHandler):
    """Display LLM-generated tokens in real time within a Streamlit container.

    Tokens are accumulated in an internal buffer and the container is
    re-rendered at most once every ``STREAM_RENDER_INTERVAL_SECONDS``
    seconds, which keeps the UI smooth even when the model produces
    tokens faster than the browser can repaint.

    Attributes:
        container: The Streamlit placeholder used for rendering.
        text: The accumulated response text so far.
    """

    def __init__(self, container, initial_text: str = "") -> None:
        """Initialise the stream handler.

        Args:
            container: A Streamlit ``st.empty()`` (or similar) container
                that will be updated with each incoming token.
            initial_text: Optional seed text to prepend to the stream.
        """
        self.container = container
        self.text = initial_text
        self._last_render_time: float = 0.0
        self._render_interval: float = STREAM_RENDER_INTERVAL_SECONDS

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Append a newly generated token and conditionally refresh the display.

        The display is only updated if at least ``_render_interval``
        seconds have elapsed since the last render, preventing excessive
        DOM writes.

        Args:
            token: The token string produced by the LLM.
            **kwargs: Additional callback keyword arguments (unused).
        """
        self.text += token
        now = time.time()
        if (now - self._last_render_time) >= self._render_interval:
            self.container.markdown(self.text + "▌")
            self._last_render_time = now

    def finalize(self) -> None:
        """Render the final accumulated text without the typing cursor."""
        self.container.markdown(self.text)

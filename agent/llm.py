"""LLM access, centralized in one place.

The agent never instantiates a model inline — it asks here — so the model
choice, temperature and (later) retry/fallback live in one seam, and tests can
inject a fake model instead of calling DeepSeek.

Real runs need ``DEEPSEEK_API_KEY`` in the environment (e.g. a local ``.env``).
"""
from __future__ import annotations

from langchain_deepseek import ChatDeepSeek

DEFAULT_MODEL = "deepseek-chat"


def create_sql_model(temperature: float = 0.0):
    """Return the chat model used to generate SQL.

    Temperature defaults to 0 so SQL generation is as deterministic as the model
    allows. Reads the API key from the environment (DEEPSEEK_API_KEY).
    """
    return ChatDeepSeek(model=DEFAULT_MODEL, temperature=temperature)

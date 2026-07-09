"""Token + latency capture via LangChain's callback hook -- the data behind cost /
latency monitoring, kept OUT of the generate code.

``UsageCallback`` is attached to the graph run (``config={"callbacks": [cb]}``); its
``on_chat_model_start`` / ``on_llm_end`` fire for every model call nested inside the
nodes, so no ``model.invoke`` site needs to know about tracing. Token counts come off
the ``LLMResult`` (``llm_output.token_usage`` for DeepSeek, with a fallback to each
message's ``usage_metadata``); a provider that reports neither contributes zeros.
"""
from __future__ import annotations

import time

from langchain_core.callbacks import BaseCallbackHandler


def _tokens_from_llm_result(response) -> tuple[int, int]:
    """(input_tokens, output_tokens) from an LLMResult, however the provider reports it.

    Covers ``llm_output.token_usage`` / ``llm_output.usage`` (OpenAI-compatible, incl.
    DeepSeek) and the LangChain-standard per-message ``usage_metadata`` (cross-provider).
    Extension point: a provider that reports usage only elsewhere (e.g.
    ``response_metadata.token_usage``) or only as ``total_tokens`` would read as 0 here.
    """
    out = getattr(response, "llm_output", None) or {}
    tu = out.get("token_usage") or out.get("usage") or {}
    input_t = tu.get("prompt_tokens", 0)
    output_t = tu.get("completion_tokens", 0)
    if input_t or output_t:
        return input_t, output_t
    # fallback: sum usage_metadata off each generated message
    for gens in getattr(response, "generations", []) or []:
        for g in gens:
            um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
            input_t += um.get("input_tokens", 0)
            output_t += um.get("output_tokens", 0)
    return input_t, output_t


class UsageCallback(BaseCallbackHandler):
    """Records one usage event (latency + tokens) per LLM call in a run."""

    def __init__(self):
        self.events: list[dict] = []
        self._starts: dict = {}

    # chat models fire on_chat_model_start; plain LLMs fire on_llm_start. run_id pairs
    # a start with its end -- the LangChain callback contract always supplies it (a UUID),
    # so the None key can't collide across concurrent calls in practice.
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._starts[run_id] = time.monotonic()

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._starts[run_id] = time.monotonic()

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        start = self._starts.pop(run_id, None)
        input_t, output_t = _tokens_from_llm_result(response)
        self.events.append({
            "latency_ms": round((time.monotonic() - start) * 1000) if start else 0,
            "input_tokens": input_t,
            "output_tokens": output_t,
        })

    def summary(self) -> dict:
        return aggregate_usage(self.events)


def aggregate_usage(events: list[dict]) -> dict:
    """Sum a list of per-call usage events into one summary."""
    in_t = sum(e.get("input_tokens", 0) for e in events)
    out_t = sum(e.get("output_tokens", 0) for e in events)
    return {
        "llm_calls": len(events),
        "input_tokens": in_t,
        "output_tokens": out_t,
        "total_tokens": in_t + out_t,
        "latency_ms": sum(e.get("latency_ms", 0) for e in events),
    }

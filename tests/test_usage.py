"""Tests for the usage callback (no real LLM; fake LLMResults stand in)."""
import pytest

from agent.usage import (DEEPSEEK_USD_PER_1M, UsageCallback, _tokens_from_llm_result,
                         aggregate_usage, estimate_cost_usd)


class _LLMResult:
    def __init__(self, llm_output=None, generations=None):
        self.llm_output = llm_output
        self.generations = generations or []


class _Gen:
    def __init__(self, usage_metadata):
        self.message = type("M", (), {"usage_metadata": usage_metadata})()


def test_tokens_from_llm_output():
    r = _LLMResult(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    assert _tokens_from_llm_result(r) == (10, 3)


def test_tokens_fallback_to_usage_metadata():
    r = _LLMResult(generations=[[_Gen({"input_tokens": 5, "output_tokens": 2})]])
    assert _tokens_from_llm_result(r) == (5, 2)


def test_tokens_zero_when_absent():
    assert _tokens_from_llm_result(_LLMResult()) == (0, 0)


def test_callback_records_one_event_per_call_with_latency():
    cb = UsageCallback()
    cb.on_chat_model_start({}, [], run_id="r1")
    cb.on_llm_end(_LLMResult(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 3}}),
                  run_id="r1")
    assert len(cb.events) == 1
    assert cb.events[0]["input_tokens"] == 10 and cb.events[0]["output_tokens"] == 3
    assert cb.events[0]["latency_ms"] >= 0


def test_callback_summary_sums_across_calls():
    cb = UsageCallback()
    for rid, (p, c) in {"a": (10, 3), "b": (4, 2)}.items():
        cb.on_chat_model_start({}, [], run_id=rid)
        cb.on_llm_end(_LLMResult(llm_output={"token_usage": {"prompt_tokens": p, "completion_tokens": c}}),
                      run_id=rid)
    s = cb.summary()
    assert s == {"llm_calls": 2, "input_tokens": 14, "output_tokens": 5,
                 "total_tokens": 19, "latency_ms": s["latency_ms"]}


def test_aggregate_empty():
    assert aggregate_usage([]) == {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                   "total_tokens": 0, "latency_ms": 0}


def test_estimate_cost_usd_applies_per_million_rates():
    # cost = input_tokens/1M * input_rate + output_tokens/1M * output_rate; guards the math
    # (per-million scaling on both dimensions) without hard-coding a possibly-stale price.
    usage = {"input_tokens": 2_000_000, "output_tokens": 500_000}
    expected = 2 * DEEPSEEK_USD_PER_1M["input"] + 0.5 * DEEPSEEK_USD_PER_1M["output"]
    assert estimate_cost_usd(usage) == pytest.approx(expected)


def test_estimate_cost_usd_zero_when_no_tokens():
    assert estimate_cost_usd({}) == 0.0

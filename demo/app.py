"""Streamlit demo for the Cadence agent -- reliability-forward, not a chat box.

Run with:
    streamlit run demo/app.py

Shows three things, all real ``run_agent`` output (nothing hard-coded):
  1. Ask   -- a question -> the answer + the generated SQL + the result table
  2. Trace -- the per-node pipeline that produced it
  3. Semantic layer ON vs OFF -- the governed metric definition, side by side

Needs ``DEEPSEEK_API_KEY`` (e.g. a local .env) to actually run the agent; the page loads
and renders the UI without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from agent.db.build_saas_db import build  # noqa: E402
from agent.generation import AnswerResult  # noqa: E402
from agent.graph import run_agent  # noqa: E402

EXAMPLES = [
    "What is our total MRR?",                       # semantic layer changes this one
    "How many accounts are in the us-east region?",
    "What's the weather in Singapore today?",       # out of domain -> reasoned refusal
]

_LLM_NODES = {"query_enhance", "planner", "generate_sql", "semantic_consistency", "python_generate"}


@st.cache_resource
def _db() -> str:
    """Build the demo SaaS DB once per session (no API key needed)."""
    import tempfile
    return str(build(Path(tempfile.mkdtemp()) / "saas.db"))


@st.cache_resource
def _model():
    """Create the model lazily -- only when a run is triggered, so the page loads
    (and the smoke test runs) without a DEEPSEEK_API_KEY."""
    from agent.llm import create_sql_model
    return create_sql_model()


def _run(question: str, *, semantic_layer: bool) -> AnswerResult:
    return run_agent(_db(), question, model=_model(), semantic_layer=semantic_layer)


def _render(res: AnswerResult) -> None:
    if not res.sql:
        # a reasoned refusal -- the agent says why instead of inventing an answer
        st.warning(f"Refused: {res.answer or res.clarification or 'no answer'}")
        return
    st.markdown(f"**Answer:** {res.answer}")
    st.code(res.sql, language="sql")
    if res.execution and res.execution.ok and res.execution.rows:
        st.dataframe([dict(zip(res.execution.columns, row)) for row in res.execution.rows])
    n_llm = sum(1 for s in res.trace if isinstance(s, dict) and s.get("node") in _LLM_NODES)
    with st.expander(f"Pipeline trace -- {len(res.trace)} steps, {n_llm} LLM calls"):
        for step in res.trace:
            st.write(step)


def _pick(example: str) -> None:
    st.session_state.question = example


def main() -> None:
    st.set_page_config(page_title="Cadence -- reliability-first NL->SQL", layout="wide")
    st.title("Cadence")
    st.caption("A reliability-first NL->SQL data agent. Everything below is real, live agent output.")

    if "question" not in st.session_state:
        st.session_state.question = EXAMPLES[0]

    st.write("Try an example:")
    for col, ex in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(ex, on_click=_pick, args=(ex,), use_container_width=True)

    st.text_input("Ask a question about the SaaS metrics:", key="question")
    ask, compare = st.columns(2)
    run_single = ask.button("Ask", type="primary", use_container_width=True)
    run_compare = compare.button("Compare: semantic layer ON vs OFF", use_container_width=True)

    question = st.session_state.question
    try:
        if run_single:
            with st.spinner("Running the agent..."):
                res = _run(question, semantic_layer=True)  # governed behaviour by default
            _render(res)
        if run_compare:
            with st.spinner("Running the same question both ways..."):
                off = _run(question, semantic_layer=False)
                on = _run(question, semantic_layer=True)
            left, right = st.columns(2)
            with left:
                st.subheader("Semantic layer OFF")
                _render(off)
            with right:
                st.subheader("Semantic layer ON (governed)")
                _render(on)
    except Exception as exc:  # most likely a missing DEEPSEEK_API_KEY
        st.error(f"Could not run the agent: {exc}. Set DEEPSEEK_API_KEY (e.g. in a local .env).")


if __name__ == "__main__":
    main()

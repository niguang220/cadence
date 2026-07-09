"""Command-line entry point: ask the agent a question.

    python -m agent "what is total MRR by plan tier?"
    python -m agent --retrieval-only "..."   # offline health check, no LLM call

The query path needs ``DEEPSEEK_API_KEY`` in the environment (a local ``.env`` is
picked up too). ``--retrieval-only`` runs schema retrieval alone (database +
embedding model, no LLM), so it works fully offline -- handy for verifying the
Docker image baked its assets. Builds the bundled SaaS database on first use.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent", description="Ask the natural-language-to-SQL data agent a question.")
    parser.add_argument("question", help="the natural-language question to answer")
    parser.add_argument("-k", type=int, default=5,
                        help="how many schema tables to retrieve (default: 5)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="run schema retrieval only (no LLM call); an offline health check")
    args = parser.parse_args(argv)

    from agent.db.build_saas_db import DB_PATH, build
    db = DB_PATH if DB_PATH.exists() else build()

    if args.retrieval_only:
        # database + embedding model, no LLM and no network -- proves the baked assets
        from agent.db.introspect import introspect
        from agent.hybrid_retriever import retrieve
        print(f"tables: {retrieve(args.question, introspect(str(db)), k=args.k)}")
        return 0

    # imported lazily so retrieval-only and --help don't pay the model-load cost
    from dotenv import load_dotenv
    load_dotenv()  # a local .env is convenient; in Docker the key comes from -e
    from agent.pipeline import answer_question

    result = answer_question(db, args.question, k=args.k)
    if result.clarification:                     # the agent asked instead of guessing
        print(result.clarification)
        return 0
    print(f"tables: {result.retrieved_tables}")
    print(f"SQL:    {result.sql}")
    print(f"answer: {result.answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

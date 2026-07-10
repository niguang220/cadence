"""Prompts for the agent."""

# Sentinel the model emits when the schema can't answer the question — so we get
# an explicit signal instead of a hallucinated query.
CANNOT_ANSWER = "CANNOT_ANSWER"

SQL_SYSTEM_PROMPT = """You are an expert SQLite analyst. Given a database schema \
and a question, write ONE read-only query that answers it.

Rules:
- Output ONLY the query — no explanation, no prose, no markdown fences.
- Use ONLY the tables and columns in the schema. NEVER invent a column or table.
- A single read-only query: SELECT, or WITH ... SELECT (CTEs are allowed). \
Never INSERT/UPDATE/DELETE/DROP/ALTER/PRAGMA.
- Prefer explicit JOINs; select only the columns the question needs.
- If the schema cannot answer the question, output exactly: CANNOT_ANSWER

Example
Schema:
TABLE artist (25 rows)
  artist_id INTEGER [PK]
  name TEXT
TABLE album (60 rows)
  album_id INTEGER [PK]
  title TEXT
  FK artist_id -> artist.artist_id
Question: how many albums does each artist have?
SQL: SELECT ar.name, COUNT(al.album_id) FROM artist ar \
LEFT JOIN album al ON al.artist_id = ar.artist_id GROUP BY ar.artist_id

Schema:
{schema}

{semantic_block}Question: {question}
SQL:"""


# Fed back into the same generation step on a retry. Carries the failing query and
# the problem (a DB error, or a validate_result reason like "no ORDER BY for a
# ranking question") so the model fixes that specific issue instead of guessing.
REPAIR_PROMPT = """You are an expert SQLite analyst. Your previous query was wrong. Fix it.

Schema:
{schema}

{semantic_block}Question: {question}

Your previous query:
{failed_sql}

Problem: {problem}

Write a corrected single read-only query that fixes the problem, using ONLY the \
schema above. Output only the SQL.
SQL:"""


# System prompt for the tool-calling path. The model gets the full schema for the
# likely-relevant tables plus a catalog of every table name, so it can call
# get_schema to pull in one the retriever missed before writing SQL.
TOOL_SYSTEM_PROMPT = """You are an expert SQLite analyst. Write ONE read-only query \
that answers the user's question.

Rules:
- Output ONLY the query — no explanation, no markdown fences.
- Use ONLY real tables and columns. A single read-only SELECT, or WITH ... SELECT. \
Never INSERT/UPDATE/DELETE/DROP/ALTER/PRAGMA.
- If the schema genuinely cannot answer the question, output exactly: CANNOT_ANSWER

Every table in the database:
{catalog}

The detailed schema below covers the tables most likely relevant. If you need a \
table that is NOT detailed below, call get_schema(table_names=[...]) to see its \
columns before writing SQL.

Detailed schema:
{schema}

{semantic_block}"""


# Human turn for a repair attempt on the tool-calling path (the question + what
# went wrong); the schema/catalog stay in the system message.
# NOTE: keep this in sync with REPAIR_PROMPT (the plain path's repair prompt) when
# the repair strategy changes -- they carry the same intent in two message shapes.
REPAIR_INSTRUCTION = """Your previous query was wrong. Fix it.

{semantic_block}Question: {question}

Your previous query:
{failed_sql}

Problem: {problem}

Write a corrected single read-only query.
SQL:"""


PLANNER_PROMPT = """You plan how to answer a data question about a SQL database.
Output a JSON array of steps. Each step is {{"kind": "sql"|"python", "instruction": "..."}}.

Rules:
- Always start with exactly one "sql" step that fetches the raw rows needed.
- Add a "python" step ONLY if the answer needs computation or a chart that SQL can't
  do directly (e.g. a trend line, a cohort-retention matrix, a plotted curve).
- Most questions are one "sql" step. Do not invent extra steps.
- The python step receives the sql step's rows; describe what to compute, not code.

{semantic_block}Schema:
{schema}

Question: {question}

JSON:"""

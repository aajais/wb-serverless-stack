"""System prompt + user-message builder + SQL extractor."""

from __future__ import annotations

import re

SYSTEM_PROMPT = """You are a SQL expert. The user will give you a question and a SQLite schema. \
You answer with a single SQL query that, when executed against the provided database, \
returns the result that answers the question.

Rules:
- Output exactly one ```sql fenced block — nothing else.
- The query must be valid SQLite SQL.
- Do not invent columns or tables that are not in the schema.
- Do not add commentary, explanation, or alternatives.
"""


def build_user_prompt(question: str, schema_text: str, evidence: str | None = None) -> str:
    parts = [
        "## Database schema",
        "",
        schema_text.rstrip(),
        "",
    ]
    if evidence:
        parts += ["## Hints", evidence.strip(), ""]
    parts += [
        "## Question",
        question.strip(),
        "",
        "Return one SQLite SQL query as a ```sql fenced block.",
    ]
    return "\n".join(parts)


_SQL_BLOCK_RE = re.compile(r"```sql\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_FALLBACK_BLOCK_RE = re.compile(r"```\s*(.+?)```", re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the first ```sql ... ``` block out of the completion.

    Falls back to any fenced block, then to the raw text if no fences found.
    Always strips trailing semicolons + whitespace so multiset compares are stable.
    """
    if text is None:
        return ""
    match = _SQL_BLOCK_RE.search(text)
    if not match:
        match = _FALLBACK_BLOCK_RE.search(text)
    sql = match.group(1) if match else text
    return sql.strip().rstrip(";").strip()

"""Natural-language -> SQL on an in-memory SQLite copy of the dataset.

Safety is enforced by SQLite itself: an authorizer allows only SELECT/READ/FUNCTION,
the DB is read-only for the session, and a progress handler aborts slow queries.
"""
from __future__ import annotations

import sqlite3
import time

import pandas as pd

import config
from analytics import infer_roles, dataset_brief
from llm import LLMRouter, extract_code

_ALLOWED = {getattr(sqlite3, n) for n in
            ("SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION", "SQLITE_RECURSIVE") if hasattr(sqlite3, n)}


def _authorizer(action, *_args):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def _connect(df: pd.DataFrame, alias: str = "") -> sqlite3.Connection:
    safe = df.copy()
    safe.columns = [str(c) for c in safe.columns]
    for c, role in infer_roles(safe).items():
        if role == "datetime":
            safe[c] = pd.to_datetime(safe[c], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(":memory:")
    safe.to_sql("data", conn, index=False)
    if alias and alias.lower() != "data":  # let people query by file name too, e.g. FROM sample_sales
        try:
            conn.execute('CREATE VIEW "%s" AS SELECT * FROM data' % alias.replace('"', ""))
        except sqlite3.DatabaseError:
            pass
    conn.execute("PRAGMA query_only = ON")
    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 10_000_000)
    except Exception:
        pass
    conn.set_authorizer(_authorizer)  # set last: loading above needs write access
    return conn


def run_select(df: pd.DataFrame, sql: str, alias: str = "") -> pd.DataFrame:
    sql = sql.strip().rstrip(";").strip()
    if not sql.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT queries are allowed.")
    conn = _connect(df, alias)
    deadline = time.time() + config.SQL_TIMEOUT_S
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(config.SQL_MAX_ROWS)
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])
    except sqlite3.DatabaseError as exc:
        hint = (" The dataset is the table named `data` (e.g. SELECT * FROM data LIMIT 10)."
                if "no such table" in str(exc) else "")
        raise ValueError(f"SQL error: {exc}.{hint}") from None
    finally:
        conn.close()


def _prompt(question: str, df: pd.DataFrame, extra: str = "") -> str:
    return f"""You are an expert SQLite analyst. Write ONE read-only SQLite query.
Table name: data (quote column names with double quotes if they contain spaces or symbols).
Dataset:
{dataset_brief(df)}

Question: {question}
Rules: SELECT or WITH only; no comments; no explanation; no markdown; dates are ISO text; \
round decimals to 2 places; add ORDER BY and LIMIT 100 when ranking.
{extra}
Return only the SQL."""


def generate_and_run(router: LLMRouter, question: str, df: pd.DataFrame, alias: str = ""):
    """Returns (sql, result_df, error). Makes one repair attempt if execution fails."""
    sql = extract_code(router.invoke(_prompt(question, df)))
    try:
        return sql, run_select(df, sql, alias), ""
    except ValueError as exc:
        fixed = extract_code(router.invoke(_prompt(
            question, df, f"Your previous query failed.\nQuery: {sql}\nError: {exc}\nFix it.")))
        try:
            return fixed, run_select(df, fixed, alias), ""
        except ValueError as exc2:
            return fixed, None, str(exc2)

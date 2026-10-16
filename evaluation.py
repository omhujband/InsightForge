"""Honest evaluation: real, checkable measurements instead of LLM self-scores."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from analytics import infer_roles

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _floats(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def numbers_in_result(result: Any, limit: int = 500) -> list[float]:
    if isinstance(result, pd.DataFrame):
        nums = result.head(limit).select_dtypes("number").to_numpy().ravel().tolist()
        return [float(x) for x in nums if pd.notna(x)]
    if isinstance(result, pd.Series):
        return numbers_in_result(result.to_frame())
    if isinstance(result, bool):
        return []
    if isinstance(result, (int, float)):
        return [float(result)]
    return _floats(str(result))


def traceability(insights: str, result: Any) -> tuple[int, int]:
    """(cited numbers found in the result, cited numbers total). Small ints and years are ignored.

    A heuristic: a number in the explanation that is not in the result may be derived (e.g. a
    percentage) or invented - it flags what to double-check, it does not prove correctness.
    """
    known = numbers_in_result(result)
    cited = [x for x in _floats(insights) if not (abs(x) < 11 and float(x).is_integer())
             and not (1900 <= x <= 2100 and float(x).is_integer())]
    hit = sum(1 for x in cited if any(abs(x - k) <= max(0.011 * abs(k), 0.0051) for k in known))
    return hit, len(cited)


def build_benchmark(df: pd.DataFrame, n: int = 6) -> list[tuple[str, Any, str]]:
    """Questions whose true answer is computed directly with pandas (question, expected, kind)."""
    roles = infer_roles(df)
    nums = [c for c, r in roles.items() if r == "numeric"]
    cats = [c for c, r in roles.items() if r == "categorical"]
    items: list[tuple[str, Any, str]] = [("How many rows are in the dataset?", len(df), "number")]
    if nums:
        items.append((f"What is the average of {nums[0]}?", float(df[nums[0]].mean()), "number"))
        items.append((f"What is the maximum value of {nums[0]}?", float(df[nums[0]].max()), "number"))
    if len(nums) > 1:
        items.append((f"What is the total (sum) of {nums[1]}?", float(df[nums[1]].sum()), "number"))
    if cats:
        items.append((f"How many distinct values does {cats[0]} have?", int(df[cats[0]].nunique()), "number"))
        mode = df[cats[0]].mode()
        if len(mode):
            items.append((f"Which {cats[0]} appears most often?", str(mode.iloc[0]).strip(), "text"))
    return items[:n]


def check_answer(result: Any, expected: Any, kind: str) -> bool:
    if kind == "number":
        e = float(expected)
        return any(abs(c - e) <= max(1e-3 * abs(e), 0.0051) for c in numbers_in_result(result))
    text = result.to_string() if isinstance(result, pd.DataFrame) else str(result)
    return str(expected).lower() in text.lower()


def summarize_history(outcomes: list) -> dict:
    """Execution-level metrics from real runs (correctness is measured by the benchmark)."""
    if not outcomes:
        return {}
    ok = [o for o in outcomes if o.ok]
    tr = [o.traceable for o in ok if o.traceable and o.traceable[1] > 0]
    return {
        "runs": len(outcomes),
        "execution_success_pct": 100 * len(ok) / len(outcomes),
        "first_try_pct": 100 * sum(1 for o in ok if o.attempts == 1) / len(outcomes),
        "avg_attempts": sum(o.attempts for o in outcomes) / len(outcomes),
        "avg_sandbox_s": sum(o.runtime_s for o in outcomes) / len(outcomes),
        "traceable_pct": (100 * sum(h for h, _ in tr) / sum(t for _, t in tr)) if tr else None,
    }

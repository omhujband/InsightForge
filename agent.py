"""LLM workflows. The core question->answer pipeline is a LangGraph state machine:

    generate_code -> execute (sandbox) -> [error? repair -> execute ...] -> insights

Calculations are always done by code in the sandbox, never by the LLM; the LLM only
writes the code and, afterwards, explains the computed result.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

import config
import sandbox
from analytics import build_evidence, dataset_brief, find_relationships
from evaluation import traceability
from llm import LLMRouter, extract_code


# ----------------------------------------------------------------------------- outcome
@dataclass
class AnalysisOutcome:
    question: str
    dataset: str = ""
    code: str = ""
    result: Any = None
    kind: str = ""
    error: str = ""
    attempts: int = 0
    runtime_s: float = 0.0
    insights: str = ""
    provider: str = ""
    traceable: tuple[int, int] | None = None   # (numbers found in result, numbers cited)
    truncated: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return not self.error


def result_to_text(result: Any, max_chars: int = 3000) -> str:
    if isinstance(result, pd.DataFrame):
        text = result.head(40).to_string(index=False)
    else:
        text = str(result)
    return text[:max_chars]


# ----------------------------------------------------------------------------- prompts
def _codegen_prompt(question: str, df: pd.DataFrame, context: str) -> str:
    return f"""You are an expert pandas data analyst. Write Python that answers the question.

DataFrame `df` (already loaded; also available: pd, np, math):
{dataset_brief(df)}

{('Conversation context (for follow-up questions):\n' + context) if context else ''}
Question: {question}

Rules:
- Assign the final answer to a variable named `result` (DataFrame, Series or scalar).
- No imports, no file or network access, no df.query()/df.eval(), no .format(); use f-strings.
- Use boolean masks, groupby, agg, sort_values. Round floats to 2 decimals.
- Do not sum identifier columns. Treat obvious placeholders as missing when relevant.
- For rankings return the top rows with their values. Parse text dates with pd.to_datetime(..., errors="coerce", format="mixed").
- Return ONLY code, no explanation, no markdown."""


def _repair_prompt(question: str, df: pd.DataFrame, code: str, error: str) -> str:
    return f"""Your previous pandas code failed. Fix it.

Dataset:
{dataset_brief(df)}

Question: {question}
Previous code:
{code}
Error: {error}

Same rules as before: assign the answer to `result`, no imports, no file/network access, no df.query().
Return ONLY the corrected code."""


def _insight_prompt(question: str, result: Any) -> str:
    return f"""You are a careful senior business analyst. Explain the computed result below.

Question: {question}
Computed result:
{result_to_text(result)}

Write at most 5 short bullet points in this structure:
- **Findings:** only statements that quote numbers visible in the result.
- **Caveats:** data limits or ambiguity in the question (1 bullet).
- **Hypotheses / next checks:** clearly label anything that is not proven by the result.
Never invent numbers that are not in the result. Do not give recommendations as facts."""


# ----------------------------------------------------------------------------- pipeline
class _State(TypedDict, total=False):
    code: str
    error: str
    attempts: int
    result: Any
    kind: str
    runtime: float
    truncated: bool
    insights: str


def run_analysis(router: LLMRouter, df: pd.DataFrame, question: str, *, dataset: str = "",
                 context: str = "", with_insights: bool = True) -> AnalysisOutcome:
    """Run the full pipeline. Raises LLMUnavailable if no model can be reached."""
    max_attempts = 1 + config.MAX_REPAIR_ATTEMPTS

    def generate(state: _State) -> _State:
        code = extract_code(router.invoke(_codegen_prompt(question, df, context)))
        return {"code": code, "attempts": 1, "error": ""}

    def execute(state: _State) -> _State:
        res = sandbox.run_code(state["code"], df)
        if res.ok:
            return {"result": res.value, "kind": res.kind, "runtime": res.runtime_s, "error": "",
                    "truncated": bool(res.total_rows and res.total_rows > len(res.value))
                    if res.kind == "dataframe" else False}
        return {"error": res.error, "runtime": res.runtime_s}

    def repair(state: _State) -> _State:
        code = extract_code(router.invoke(_repair_prompt(question, df, state["code"], state["error"])))
        return {"code": code, "attempts": state["attempts"] + 1, "error": ""}

    def insights(state: _State) -> _State:
        return {"insights": router.invoke(_insight_prompt(question, state["result"])) if with_insights else ""}

    def after_execute(state: _State) -> str:
        if not state.get("error"):
            return "insights"
        return "repair" if state["attempts"] < max_attempts else END

    graph = StateGraph(_State)
    graph.add_node("generate", generate)
    graph.add_node("execute", execute)
    graph.add_node("repair", repair)
    graph.add_node("insights", insights)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", after_execute, {"insights": "insights", "repair": "repair", END: END})
    graph.add_edge("repair", "execute")
    graph.add_edge("insights", END)
    state = graph.compile().invoke({})

    outcome = AnalysisOutcome(
        question=question, dataset=dataset, code=state.get("code", ""), result=state.get("result"),
        kind=state.get("kind", ""), error=state.get("error", ""), attempts=state.get("attempts", 0),
        runtime_s=state.get("runtime", 0.0), insights=state.get("insights", ""),
        provider=router.last_provider or "", truncated=state.get("truncated", False))
    if outcome.ok and outcome.insights:
        outcome.traceable = traceability(outcome.insights, outcome.result)
    return outcome


# ----------------------------------------------------------------------------- other LLM helpers
def suggest_questions(router: LLMRouter, df: pd.DataFrame) -> list[str]:
    text = router.invoke(f"""You are a data analyst. Suggest exactly 5 useful business questions that can be \
answered from this dataset using the columns shown.
{dataset_brief(df, sample_rows=3)}
One question per line. No numbering, bullets or explanations.""")
    qs = [re.sub(r"^[\d\.\)\-\*•\s]+", "", ln).strip() for ln in text.splitlines()]
    return [q for q in qs if len(q) > 8][:5]


def root_cause(router: LLMRouter, df: pd.DataFrame, issue: str, target: str | None):
    """Returns (evidence_text, llm_answer). Evidence is computed first; the LLM must separate it from guesses."""
    evidence = build_evidence(df, target)
    answer = router.invoke(f"""You are a senior business analyst doing root-cause analysis.

Business problem: {issue}

Computed evidence from the data (these numbers are facts):
{evidence}

Dataset overview:
{dataset_brief(df, sample_rows=3)}

Write max 6 short bullets, grouped under:
**Evidence-backed observations** (cite the numbers above), **Hypotheses** (plausible causes the data cannot prove), \
**What to check next**. Do not present hypotheses as facts.""")
    return evidence, answer


def copilot_answer(router: LLMRouter, df: pd.DataFrame, question: str) -> str:
    return router.invoke(f"""You are an expert business analyst Copilot.
Dataset overview:
{dataset_brief(df, sample_rows=10)}

Computed context:
{build_evidence(df)}

Question: {question}
Answer concisely: Finding, Evidence (quote numbers from the context), Business impact, Suggested action.
If the data is insufficient, say so instead of guessing.""")


def relationship_advice(router: LLMRouter, dfs: dict[str, pd.DataFrame]) -> str:
    rels = find_relationships(dfs)
    shapes = {k: list(v.columns) for k, v in dfs.items()}
    return router.invoke(f"""Datasets and columns: {shapes}
Detected shared keys (with value overlap): {rels}
Recommend which datasets to join, on which key, with which join type, and 3 analyses this enables. Be concise.""")

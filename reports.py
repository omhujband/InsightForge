"""Text and PDF reports covering the whole session (return bytes; nothing is written to disk).

`items` may contain: analyses (list of AnalysisOutcome), forecast, cleaning (log list),
sql (query, DataFrame), root_cause (dict), copilot (list), rag (list).
"""
from __future__ import annotations

import re
import textwrap
import time
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

import analytics as A

LABELS = {"analyses": "AI analyses", "forecast": "Forecast", "cleaning": "Data cleaning log",
          "sql": "SQL query", "root_cause": "Root-cause analysis", "copilot": "Copilot Q&A", "rag": "Document Q&A"}


def _meta(name: str, df: pd.DataFrame) -> list[tuple[str, str]]:
    return [("Dataset", name), ("Rows", f"{len(df):,}"), ("Columns", str(df.shape[1])),
            ("Missing cells", f"{int(df.isna().sum().sum()):,}"), ("Duplicate rows", f"{int(df.duplicated().sum()):,}")]


def _df_text(res, rows=30) -> str:
    return res.head(rows).to_string(index=False) if isinstance(res, pd.DataFrame) else str(res)


# ----------------------------------------------------------------------------- text
def build_text_report(name: str, df: pd.DataFrame, items: dict | None = None) -> str:
    items = items or {}
    L = ["InsightForge - AI DATA ANALYST REPORT", time.strftime("Generated: %Y-%m-%d %H:%M"), ""]
    L += [f"{k}: {v}" for k, v in _meta(name, df)]
    L += ["", "KEY METRICS"] + [f"  {k}: {v:,.2f}" for k, v in A.kpi_candidates(df)]
    for i, o in enumerate(items.get("analyses", []), 1):
        L += ["", f"=== AI ANALYSIS {i}: {o.question}", _df_text(o.result)]
        if o.insights:
            L += ["", "AI interpretation (verify against the computed result):", o.insights]
    if "forecast" in items:
        fc = items["forecast"]
        L += ["", f"=== FORECAST: {fc['target']} ({fc['freq']}, {fc['horizon']} periods, model {fc['model']})", _df_text(fc["fdf"], 12)]
        if fc["bt"]:
            L.append(f"Backtest MAE={fc['bt']['mae']:,.1f}; vs naive: {fc['bt']['improvement_vs_naive_pct']:+.0f}%")
    if "cleaning" in items:
        L += ["", "=== DATA CLEANING LOG"] + [f"- {x}" for x in items["cleaning"]]
    if "sql" in items:
        L += ["", "=== SQL", items["sql"][0], _df_text(items["sql"][1])]
    if "root_cause" in items:
        rc = items["root_cause"]
        L += ["", f"=== ROOT CAUSE: {rc['issue']}", "Computed evidence:", rc["evidence"], "", rc["answer"]]
    for key, title in (("copilot", "COPILOT"), ("rag", "DOCUMENT Q&A")):
        for x in items.get(key, []):
            L += ["", f"=== {title}: {x['q']}", x["a"]]
    return "\n".join(L)


# ----------------------------------------------------------------------------- pdf
def _md(text: str) -> str:
    t = escape(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?m)^\s*[-*]\s+", "• ", t)
    return t.replace("\n", "<br/>")


def _table(df: pd.DataFrame, max_rows=25, max_cols=6, small=True):
    view = df.iloc[:max_rows, :max_cols]
    cells = [[str(c)[:18] for c in view.columns]] + [[str(v)[:18] for v in row] for row in view.to_numpy()]
    t = Table(cells, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8 if small else 9),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EADCCB"))]))
    return t, (len(df) > max_rows or df.shape[1] > max_cols), view


def build_pdf_report(name: str, df: pd.DataFrame, items: dict | None = None) -> bytes:
    items = items or {}
    S = getSampleStyleSheet()
    p = lambda text, style="Normal": Paragraph(_md(text), S[style])  # noqa: E731
    story = [p("InsightForge - AI Data Analyst Report", "Title"), p(time.strftime("Generated %Y-%m-%d %H:%M")), Spacer(1, 10)]

    def df_block(d: pd.DataFrame):
        t, cut, view = _table(d)
        story.append(t)
        if cut:
            story.append(p(f"(showing {len(view)} rows x {view.shape[1]} columns of {len(d)} x {d.shape[1]})"))

    meta = Table([[k, v] for k, v in _meta(name, df)], colWidths=[110, 330])
    meta.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                              ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1E7DA"))]))
    story += [p("1. Dataset overview", "Heading2"), meta, Spacer(1, 8)]
    kpis = A.kpi_candidates(df)
    if kpis:
        story += [p("Key metrics", "Heading3"), p("\n".join(f"- {k}: {v:,.2f}" for k, v in kpis))]
    story += [Spacer(1, 6), p("Column profile", "Heading3")]
    prof = A.column_profile(df)[["Column", "Role", "Dtype", "Missing %", "Unique", "Example"]]
    t, cut, _ = _table(prof, max_rows=40, max_cols=6)
    story.append(t)

    n = 2
    for i, o in enumerate(items.get("analyses", []), 1):
        story += [PageBreak() if i == 1 else Spacer(1, 14), p(f"{n}. AI analysis {i}", "Heading2"),
                  p(f"**Question:** {o.question}"), Spacer(1, 4), p("Computed result", "Heading3")]
        df_block(o.result) if isinstance(o.result, pd.DataFrame) else story.append(p(o.result))
        story += [Spacer(1, 4), p("Method (code that was executed)", "Heading3"),
                  Preformatted("\n".join(textwrap.fill(ln, 95) for ln in o.code.splitlines()), S["Code"])]
        if o.insights:
            story += [p("AI interpretation", "Heading3"),
                      p("Written by an AI model from the computed result above. Verify before acting."), p(o.insights)]
        n += 1
    if "forecast" in items:
        fc = items["forecast"]
        story += [PageBreak(), p(f"{n}. Forecast", "Heading2"),
                  p(f"Target: {fc['target']} | frequency: {fc['freq']} | horizon: {fc['horizon']} periods | model: {fc['model']}")]
        bt = fc["bt"]
        if bt:
            imp = bt["improvement_vs_naive_pct"]
            story.append(p(f"Backtest on last {bt['holdout_periods']} periods: MAE {bt['mae']:,.1f}"
                           + (f", MAPE {bt['mape_pct']:.1f}%" if bt["mape_pct"] is not None else "")
                           + (f"; {imp:+.0f}% vs a naive 'repeat last value' forecast." if imp is not None else ".")))
        fdf = fc["fdf"].copy()
        fdf["ds"] = fdf["ds"].dt.strftime("%Y-%m-%d")
        story.append(Spacer(1, 4))
        df_block(fdf.round(1))
        n += 1
    if "cleaning" in items:
        story += [Spacer(1, 12), p(f"{n}. Data cleaning log", "Heading2"), p("\n".join(f"- {x}" for x in items["cleaning"]))]
        n += 1
    if "sql" in items:
        story += [Spacer(1, 12), p(f"{n}. SQL query", "Heading2"), Preformatted(items["sql"][0], S["Code"])]
        df_block(items["sql"][1])
        n += 1
    if "root_cause" in items:
        rc = items["root_cause"]
        story += [Spacer(1, 12), p(f"{n}. Root-cause analysis", "Heading2"), p(f"**Problem:** {rc['issue']}"),
                  p("Computed evidence", "Heading3"), Preformatted(rc["evidence"], S["Code"]),
                  p("AI interpretation (hypotheses are not proven)", "Heading3"), p(rc["answer"])]
        n += 1
    for key, title in (("copilot", "Copilot Q&A"), ("rag", "Document Q&A")):
        if items.get(key):
            story += [Spacer(1, 12), p(f"{n}. {title}", "Heading2")]
            for x in items[key]:
                story += [p(f"**Q:** {x['q']}"), p(x["a"]), Spacer(1, 6)]
            n += 1
    if len(story) < 12 and not items:
        story.append(p("No analyses were included - run the AI Analyst, Forecast, etc. first to add them."))
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title="InsightForge - AI Data Analyst Report").build(story)
    return buf.getvalue()

# 📊 InsightForge - AI Data Analyst

Upload a dataset, ask questions in plain English, and get **verified calculations**, charts, AI-written
insights, forecasts and reports — without writing SQL or code. Built with **Python + Streamlit**.

> The key idea: **the AI never does the maths.** It writes code; the code runs in an isolated sandbox;
> the numbers you see are computed. The AI only explains the result afterwards.

---

## Features

| Area | What it does |
|---|---|
| **Data upload** | CSV, Excel (.xlsx/.xls), JSON; PDF/TXT for documents. Several files at once, with a built-in sample dataset. |
| **Dashboard & KPIs** | Row/column/missing/duplicate counts, KPI cards (identifier columns excluded; sums vs. averages chosen by column type), distributions, top categories, monthly trend, column profile with inferred roles. |
| **AI Analyst** | Natural-language question → pandas code → sandbox → answer table + auto-chart + insights. Suggested questions, follow-up questions with context, generated code always visible, automatic **self-repair** when code fails. |
| **SQL analytics** | English → SQLite query (editable). Safety is enforced by SQLite itself (read-only authorizer + timeout). Hand-written SQL works without an AI key. |
| **Data cleaning** | Proposes fixes with counts, you choose which to apply, shows an audit log and before/after metrics; keeps the original; can switch analysis to the cleaned copy. |
| **Forecasting** | Trend + seasonality model (Prophet optional) with a **backtest against a naive baseline**, and a warning when the model isn't better than guessing. |
| **Root-cause analysis** | Computes evidence first (correlations, group differences, recent trend), then asks the AI to separate *evidence-backed observations* from *hypotheses*. |
| **Dataset relationships** | Detects shared keys with value-overlap and cardinality (one-to-many…), join preview with row-explosion warning, AI join advice. |
| **Statistics** | Correlation heatmap, IQR outliers (adjustable), summary table. |
| **Copilot** | Free-form business Q&A grounded in computed dataset statistics. |
| **Document Q&A (RAG)** | PDF/TXT → chunks → retrieval → answer with source citations; abstains when nothing relevant is found. Works with Gemini embeddings or fully offline (TF-IDF). Re-indexes automatically when files change. |
| **Evaluation** | Real execution metrics from your session plus an **accuracy benchmark**: questions whose true answers are computed with pandas, compared with the AI's answers. |
| **Reports** | Text and PDF reports of the whole session: dataset overview, every AI analysis (computed result, exact code, AI interpretation clearly labelled), forecast with backtest, cleaning log, SQL, root cause, Copilot and document Q&A. Sections are selectable. |
| **Authentication & roles** | Supabase email/password login; Admin / Manager / Analyst / Viewer roles. Demo mode (no Supabase) has a role switcher. |
| **Multi-LLM** | Groq, Gemini, OpenRouter, OpenAI, xAI with automatic failover, key rotation and rate-limit cooldown. |

### Roles

| Role | Tabs |
|---|---|
| Viewer | Dashboard, Statistics |
| Manager | Dashboard, AI Analyst, Documents, Statistics, Reports |
| Analyst | Manager's + Evaluation, SQL, Cleaning, Forecast, Root Cause, Copilot |
| Admin | Everything, plus Relationships |

---

## How it works

```
                      ┌──────────────────────── Streamlit UI (app.py, views.py) ───────────────────────┐
   login (auth.py) →  │ Dashboard │ AI Analyst │ SQL │ Cleaning │ Forecast │ Root cause │ RAG │ Reports │
                      └────────────┬───────────────────────────────────────────────────────────────────┘
                                   │
        question ──► agent.py (LangGraph state machine)
                        generate_code ─► execute ──ok──► insights ─► answer
                                            │  error
                                            └─► repair (up to 2×) ─► execute
                                   │
              sandbox.py ──► child process: code_policy (AST rules) + timeout + memory cap + no secrets
              sql_engine.py ► SQLite in memory: authorizer allows SELECT only
              analytics.py ► profiling, cleaning, KPIs, forecast, relationships (no LLM)
              rag.py ───────► extract → chunk → TF-IDF / Gemini embeddings → cited answer
              llm.py ───────► provider router with failover
```

### Design decisions (and why)

* **Compute, then explain.** Numbers come from executed code, never from the model's head.
* **Sandboxed execution.** Generated code runs in a *separate process* with a static policy (no imports, no
  file/network access, no dunder tricks), a time limit, a memory limit and an environment stripped of API
  keys. Results come back as JSON, never pickle.
* **Self-repair loop.** If the code errors, the error is fed back to the model for another try — a real
  agentic loop, not a single prompt.
* **Honest metrics.** No LLM "quality scores". The Evaluation tab reports execution success, attempts, and a
  benchmark with known answers. Insights get a *number check* showing how many quoted numbers actually appear
  in the computed result.
* **Roles come from a trusted place.** In Supabase mode the role is read from `app_metadata.role`, which users
  cannot edit — not from a dropdown.
* **Works without keys.** Everything that doesn't need an LLM (dashboard, cleaning, forecast, statistics,
  hand-written SQL, reports) runs with no configuration at all.

---

## Project structure

```
app.py              entry point: auth gate, sidebar, data sources, role-filtered tabs
views.py            one render function per tab
agent.py            LangGraph pipeline + other LLM workflows (suggestions, root cause, copilot)
llm.py              provider router (Groq/Gemini/OpenRouter/OpenAI/xAI) with failover
sandbox.py          runs generated code in a child process (timeout, memory cap, clean env)
sandbox_runner.py   the child process itself
code_policy.py      static rules for generated code
sql_engine.py       NL→SQL and read-only SQLite execution
analytics.py        profiling, KPIs, cleaning, relationships, forecasting
rag.py              document extraction, chunking, retrieval, cited answers
evaluation.py       benchmark builder, answer checking, number-traceability check
charts.py           theme + automatic chart choice
reports.py          text / PDF reports
auth.py             Supabase auth, roles, demo mode
config.py           environment variables and limits
data/               sample sales, customers and a short company report
.streamlit/         theme and server settings
```

---

## Honest limitations

* The sandbox is **defence in depth, not a security boundary** (no container/VM). It is appropriate for a
  demo or internal tool; for untrusted public use, run the app inside a container with no network access.
* LLMs can still misread an ambiguous question. That is why the generated code is always shown and the
  Evaluation tab exists — check important answers.
* Sessions are in memory: refreshing the page logs you out and clears history.
* Forecasts are simple models; the backtest tells you when not to trust them.
* Default model names (e.g. `openai/gpt-oss-120b`, `gemini-2.5-flash`) can be changed in `.env` if a
  provider retires them.

See **working.md** for setup and a demo walkthrough.

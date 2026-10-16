"""Streamlit tab renderers. Each takes a context (router, df, name, dfs, docs, embedder)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import agent
import analytics as A
import charts
import evaluation as E
import rag
import reports
import sql_engine
from llm import LLMUnavailable

T = charts.TEMPLATE


def _need_llm(ctx) -> bool:
    if ctx.router.configured:
        return True
    st.info("This feature needs an LLM. Add a free Groq or Gemini key to `.env` and restart the app "
            "(see working.md). Features without the 🤖 icon work without any key.")
    return False


def _fmt(v):
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


# ----------------------------------------------------------------------------- dashboard
def render_dashboard(ctx):
    df, k = ctx.df, ctx.name
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    kpis = A.kpi_candidates(df)
    if kpis:
        st.subheader("Key metrics")
        cols = st.columns(len(kpis))
        for col, (label, val) in zip(cols, kpis):
            col.metric(label, f"{val:,.2f}")
        st.caption("Identifier columns are excluded. Amounts/volumes are summed; rates, prices and scores are averaged.")

    roles = A.infer_roles(df)
    nums = [c for c, r in roles.items() if r == "numeric"]
    cats = [c for c, r in roles.items() if r == "categorical"]
    dates = [c for c, r in roles.items() if r == "datetime"]
    left, right = st.columns(2)
    with left:
        if nums:
            col = st.selectbox("Distribution of", nums, key=f"dash_num_{k}")
            st.plotly_chart(px.histogram(df, x=col, template=T, title=f"Distribution: {col}"), width="stretch")
    with right:
        if cats:
            col = st.selectbox("Top categories of", cats, key=f"dash_cat_{k}")
            top = df[col].astype(str).str.strip().value_counts().head(10).reset_index()
            top.columns = [col, "Count"]
            st.plotly_chart(px.bar(top, x=col, y="Count", template=T, title=f"Top {col}"), width="stretch")
    if dates and nums:
        d1, d2 = st.columns(2)
        dcol = d1.selectbox("Date column", dates, key=f"dash_date_{k}")
        ncol = d2.selectbox("Metric", nums, key=f"dash_metric_{k}")
        ts = pd.DataFrame({"d": A.as_datetime(df, dcol), "v": pd.to_numeric(df[ncol], errors="coerce")}).dropna()
        if not ts.empty:
            agg = ts.set_index("d")["v"].resample("MS").sum().reset_index()
            st.plotly_chart(px.line(agg, x="d", y="v", markers=True, template=T,
                                    title=f"Monthly total of {ncol}", labels={"d": "Month", "v": ncol}), width="stretch")
    with st.expander("👀 Preview data"):
        st.dataframe(df.head(100), width="stretch")
    with st.expander("📋 Column profile (inferred roles)"):
        st.dataframe(A.column_profile(df), width="stretch")


# ----------------------------------------------------------------------------- AI analyst
def show_outcome(o: agent.AnalysisOutcome):
    if o.code:
        with st.expander("🔧 Generated code (what was actually executed)"):
            st.code(o.code, language="python")
    if not o.ok:
        st.error(f"Could not produce a reliable answer after {o.attempts} attempt(s).\n\n{o.error}")
        return
    st.caption(f"{o.attempts} attempt(s) · sandbox {o.runtime_s:.2f}s · model: {o.provider}")
    st.subheader("📌 Answer")
    res = o.result
    if isinstance(res, pd.DataFrame):
        st.dataframe(res.head(200), width="stretch")
        if o.truncated:
            st.caption("Result truncated to the first 1,000 rows.")
        st.download_button("⬇ Download result CSV", res.to_csv(index=False), "analysis_result.csv", "text/csv")
        fig = charts.auto_chart(res)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    elif o.kind == "json":
        st.json(res)
    else:
        st.success(_fmt(res))
    if o.insights:
        st.subheader("📈 AI interpretation")
        st.markdown(o.insights)
        if o.traceable and o.traceable[1]:
            hit, tot = o.traceable
            st.caption(f"Number check: {hit} of {tot} numbers quoted above appear in the computed result "
                       "(others may be derived or wrong - verify). The explanation is AI-written; the table is computed.")


def _run(ctx, question: str, context: str = ""):
    with st.status("Analyzing…", expanded=False) as status:
        try:
            o = agent.run_analysis(ctx.router, ctx.df, question, dataset=ctx.name, context=context)
        except LLMUnavailable as exc:
            status.update(label="AI model unavailable", state="error")
            st.error(str(exc))
            return
        status.update(label="Done" if o.ok else "Failed", state="complete" if o.ok else "error")
    st.session_state.last_outcome = o
    st.session_state.history.append(o)


def render_ai_analyst(ctx):
    if not _need_llm(ctx):
        return
    sugg = st.session_state.suggestions
    head, btn = st.columns([5, 1])
    head.subheader("💡 Suggested questions")
    if btn.button("🔄 Refresh", key="refresh_sugg") or ctx.name not in sugg:
        try:
            sugg[ctx.name] = agent.suggest_questions(ctx.router, ctx.df)
        except LLMUnavailable as exc:
            sugg[ctx.name] = []
            st.warning(f"Could not generate suggestions: {exc}")
    for i, q in enumerate(sugg.get(ctx.name, [])):
        if st.button(q, key=f"sugg_{i}"):
            st.session_state.pending_question = q

    with st.form("ask_form"):
        typed = st.text_input("Ask anything about your data", placeholder="e.g. Which region had the highest total revenue?")
        submitted = st.form_submit_button("🚀 Analyze", type="primary")
    question = st.session_state.pop("pending_question", None) or (typed.strip() if submitted and typed.strip() else None)
    if question:
        _run(ctx, question)

    o = st.session_state.last_outcome
    if o is not None and o.dataset == ctx.name:
        st.divider()
        st.markdown(f"**Question:** {o.question}")
        show_outcome(o)
        with st.form("followup_form"):
            follow = st.text_input("💬 Ask a follow-up (uses the answer above as context)")
            if st.form_submit_button("➡ Ask follow-up") and follow.strip():
                prev = f"Previous question: {o.question}\nPrevious result:\n{agent.result_to_text(o.result, 1500)}"
                _run(ctx, follow.strip(), prev)
                st.rerun()


# ----------------------------------------------------------------------------- documents (RAG)
def render_rag(ctx):
    st.caption("Ask questions across uploaded PDF/TXT documents. Answers cite their source passages.")
    if not ctx.docs:
        st.info("Upload a PDF or TXT file in the sidebar (or load the sample data) to use document Q&A.")
        return
    if not _need_llm(ctx):
        return
    sig = rag.signature(ctx.docs)
    if st.session_state.kb_sig != sig:
        with st.spinner("Indexing documents…"):
            kb, notes = rag.KnowledgeBase.build(ctx.docs, ctx.embedder)
        st.session_state.update(kb=kb, kb_sig=sig, kb_notes=notes, rag_history=[])
    kb = st.session_state.kb
    for n in st.session_state.kb_notes:
        st.warning(n)
    st.caption(f"Index: {len(kb.chunks)} passages from {len(ctx.docs)} file(s) · retrieval: {kb.backend}")
    with st.form("rag_form"):
        q = st.text_input("Question", placeholder="What is the discount policy?")
        go_ = st.form_submit_button("🔎 Search documents", type="primary")
    if go_ and q.strip():
        with st.spinner("Searching…"):
            try:
                text, hits = rag.answer(ctx.router, kb, q.strip())
                st.session_state.rag_history.append({"q": q.strip(), "a": text, "hits": hits})
            except LLMUnavailable as exc:
                st.error(str(exc))
    for item in reversed(st.session_state.rag_history):
        st.markdown(f"**❓ {item['q']}**")
        st.markdown(item["a"])
        if item["hits"]:
            with st.expander("📌 Retrieved passages"):
                for c, s in item["hits"]:
                    st.markdown(f"**{c.label}** · score {s:.2f}")
                    st.caption(c.text[:500])
        st.divider()


# ----------------------------------------------------------------------------- evaluation
def render_evaluation(ctx):
    st.caption("Measured from real runs. Execution metrics show reliability; the benchmark measures correctness.")
    hist = [o for o in st.session_state.history]
    s = E.summarize_history(hist)
    if s:
        c = st.columns(5)
        c[0].metric("Questions run", s["runs"])
        c[1].metric("Executed successfully", f"{s['execution_success_pct']:.0f}%")
        c[2].metric("Right on first try", f"{s['first_try_pct']:.0f}%")
        c[3].metric("Avg attempts", f"{s['avg_attempts']:.2f}")
        c[4].metric("Numbers traceable", "n/a" if s["traceable_pct"] is None else f"{s['traceable_pct']:.0f}%")
        st.dataframe(pd.DataFrame([{"Time": pd.to_datetime(o.timestamp, unit="s").strftime("%H:%M:%S"),
                                    "Question": o.question, "OK": o.ok, "Attempts": o.attempts,
                                    "Sandbox s": round(o.runtime_s, 2), "Model": o.provider} for o in hist]),
                     width="stretch")
        st.caption("'Executed successfully' means the code ran and returned something - not that the answer is right. "
                   "Use the benchmark below for correctness.")
    else:
        st.info("No questions have been run yet. Ask something in the 🤖 AI Analyst tab.")

    st.subheader("✅ Accuracy benchmark")
    st.caption("Generates questions whose true answers are computed directly with pandas, asks the AI the same "
               "questions, and checks the numbers match.")
    if not _need_llm(ctx):
        return
    if st.button("▶ Run benchmark on this dataset", key="run_bench"):
        rows = []
        bar = st.progress(0.0)
        bench = E.build_benchmark(ctx.df)
        for i, (q, expected, kind) in enumerate(bench):
            try:
                o = agent.run_analysis(ctx.router, ctx.df, q, dataset=ctx.name, with_insights=False)
                correct = o.ok and E.check_answer(o.result, expected, kind)
                got = agent.result_to_text(o.result, 80).replace("\n", " ") if o.ok else o.error[:80]
            except LLMUnavailable as exc:
                st.error(str(exc))
                break
            rows.append({"Question": q, "Expected": str(round(expected, 4) if isinstance(expected, float) else expected),
                         "AI answer": got, "Correct": correct, "Attempts": o.attempts})
            bar.progress((i + 1) / len(bench))
        st.session_state.benchmark = rows
    rows = st.session_state.get("benchmark")
    if rows:
        acc = 100 * sum(r["Correct"] for r in rows) / len(rows)
        st.metric("Benchmark accuracy", f"{acc:.0f}%", help=f"{len(rows)} questions")
        st.dataframe(pd.DataFrame(rows), width="stretch")


# ----------------------------------------------------------------------------- SQL
def render_sql(ctx):
    alias = Path(ctx.name).stem
    st.caption(f"Query the table **`data`** (or `{alias}`). SQLite is built into Python - nothing to install. "
               "Only read-only SELECT queries can run, enforced by the database itself.")
    with st.expander("Columns you can use"):
        st.code(", ".join(f'"{c}"' for c in ctx.df.columns))
    if "sql_text" not in st.session_state:
        st.session_state.sql_text = "SELECT * FROM data LIMIT 10"
    if ctx.router.configured:
        q = st.text_input("Ask in English", placeholder="Show total revenue by region", key="sql_q")
        if st.button("🤖 Generate & run SQL", key="sql_gen") and q.strip():
            try:
                sql, res, err = sql_engine.generate_and_run(ctx.router, q.strip(), ctx.df, alias)
                st.session_state.sql_text, st.session_state.sql_result, st.session_state.sql_err = sql, res, err
                st.session_state.sql_ds = ctx.name
            except LLMUnavailable as exc:
                st.error(str(exc))
    else:
        st.info("🤖 Natural-language SQL needs an LLM key. You can still write SQL by hand below.")
    sql = st.text_area("SQL (editable)", key="sql_text", height=120)
    if st.button("▶ Run SQL", key="sql_run") and sql.strip():
        try:
            st.session_state.sql_result, st.session_state.sql_err = sql_engine.run_select(ctx.df, sql, alias), ""
            st.session_state.sql_ds = ctx.name
        except ValueError as exc:
            st.session_state.sql_result, st.session_state.sql_err = None, str(exc)
    if st.session_state.get("sql_err"):
        st.error(st.session_state.sql_err)
    res = st.session_state.get("sql_result")
    if res is not None:
        st.dataframe(res, width="stretch")
        st.download_button("⬇ Download SQL result", res.to_csv(index=False), "sql_result.csv", "text/csv")
        fig = charts.auto_chart(res)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- cleaning
def render_cleaning(ctx):
    st.caption("Review what would change, choose the steps, then apply. Your original data is never modified.")
    actions = A.propose_cleaning(ctx.df)
    if not actions:
        st.success("No issues found: no padded text, placeholders, duplicates or missing values.")
        return
    chosen = [a.key for a in actions
              if st.checkbox(f"{a.title}  — **{a.count:,}** affected", value=a.default, key=f"clean_{ctx.name}_{a.key}")]
    if st.button("🧹 Apply selected steps", key="clean_apply"):
        before = int(ctx.df.isna().sum().sum())
        cleaned, log = A.apply_cleaning(ctx.df, chosen)
        st.session_state.cleaning[ctx.name] = {"df": cleaned, "log": log, "before": before,
                                               "after": int(cleaned.isna().sum().sum()),
                                               "rows": (len(ctx.df), len(cleaned))}
    res = st.session_state.cleaning.get(ctx.name)
    if res:
        c = st.columns(3)
        c[0].metric("Rows", f"{res['rows'][1]:,}", f"{res['rows'][1] - res['rows'][0]:,}")
        c[1].metric("Missing before", f"{res['before']:,}")
        c[2].metric("Missing after", f"{res['after']:,}")
        st.markdown("**Audit log**\n" + "\n".join(f"- {line}" for line in res["log"]))
        st.dataframe(res["df"].head(100), width="stretch")
        a, b = st.columns(2)
        a.download_button("⬇ Download cleaned CSV", res["df"].to_csv(index=False), "cleaned_data.csv", "text/csv")
        if b.button("✅ Use cleaned data for analysis", key="clean_use"):
            st.session_state.derived[f"{ctx.name} (cleaned)"] = res["df"]
            st.session_state.next_dataset = f"{ctx.name} (cleaned)"
            st.toast("Switched to the cleaned dataset")
            st.rerun()


# ----------------------------------------------------------------------------- forecast
def render_forecast(ctx):
    roles = A.infer_roles(ctx.df)
    dates = [c for c, r in roles.items() if r == "datetime"]
    nums = [c for c, r in roles.items() if r == "numeric"]
    if not dates or not nums:
        st.warning("Forecasting needs one date column and one numeric column.")
        return
    c = st.columns(4)
    dcol = c[0].selectbox("Date column", dates, key="fc_date")
    tcol = c[1].selectbox("Target", nums, key="fc_target")
    freq_name = c[2].selectbox("Frequency", list(A.FREQS), index=2, key="fc_freq")
    agg = c[3].selectbox("Aggregate per period", ["sum", "mean"], key="fc_agg")
    d = st.columns(3)
    horizon = d[0].slider("Periods to forecast", 3, 60, 6, key="fc_h")
    models = ["builtin"] + (["prophet"] if A.prophet_available() else [])
    model = d[1].selectbox("Model", models, format_func=lambda m: "Trend + seasonality" if m == "builtin" else "Prophet", key="fc_model")
    drop_last = d[2].checkbox("Drop incomplete last period", value=True, key="fc_drop")
    if st.button("🔮 Generate forecast", key="fc_go"):
        freq, m = A.FREQS[freq_name]
        try:
            ser = A.build_series(ctx.df, dcol, tcol, freq, agg, drop_last)
            if len(ser) < 8:
                st.error(f"Only {len(ser)} periods of history - too little to forecast. Try a finer frequency.")
            else:
                fdf, bt = A.forecast(ser, horizon, freq, m, model)
                st.session_state.forecast = {"ser": ser, "fdf": fdf, "bt": bt, "target": tcol, "model": model,
                                             "dataset": ctx.name, "freq": freq_name, "horizon": horizon}
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
    fc = st.session_state.get("forecast")
    if not fc:
        return
    bt = fc["bt"]
    if bt:
        m1, m2, m3 = st.columns(3)
        m1.metric("Backtest MAPE", "n/a" if bt["mape_pct"] is None else f"{bt['mape_pct']:.1f}%",
                  help=f"Model re-fit without the last {bt['holdout_periods']} periods (= your horizon, capped at 25% of history), then scored on them.")
        m2.metric("Backtest MAE", f"{bt['mae']:,.1f}")
        imp = bt["improvement_vs_naive_pct"]
        m3.metric("vs. naive 'repeat last value'", "n/a" if imp is None else f"{imp:+.0f}%",
                  help="Positive = more accurate than the simplest possible forecast.")
        if imp is not None and imp <= 0:
            st.warning("This model does not beat a naive baseline on held-out data - treat the forecast as unreliable. "
                       "Try a coarser frequency (weekly/monthly).")
    ser, fdf = fc["ser"], fc["fdf"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=fdf["ds"], y=fdf["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fdf["ds"], y=fdf["yhat_lower"], mode="lines", fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(166,83,60,0.18)", name="Interval"))
    fig.add_trace(go.Scatter(x=fdf["ds"], y=fdf["yhat"], mode="lines+markers", name="Forecast"))
    fig.update_layout(template=T, title=f"Forecast: {fc['target']}")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fdf, width="stretch")
    st.download_button("⬇ Download forecast CSV", fdf.to_csv(index=False), "forecast.csv", "text/csv")


# ----------------------------------------------------------------------------- root cause
def render_root_cause(ctx):
    if not _need_llm(ctx):
        return
    nums = [c for c, r in A.infer_roles(ctx.df).items() if r == "numeric"]
    issue = st.text_input("Describe the business problem", placeholder="Revenue dropped in recent months", key="rc_issue")
    target = st.selectbox("Metric to investigate", nums or [None], key="rc_target")
    if st.button("🔎 Analyze root cause", key="rc_go") and issue.strip():
        with st.spinner("Computing evidence and analyzing…"):
            try:
                ev, ans = agent.root_cause(ctx.router, ctx.df, issue.strip(), target)
                st.session_state.root_cause = {"dataset": ctx.name, "issue": issue.strip(), "evidence": ev, "answer": ans}
            except LLMUnavailable as exc:
                st.error(str(exc))
    rc = st.session_state.get("root_cause")
    if rc:
        with st.expander("📐 Computed evidence (facts used)", expanded=True):
            st.text(rc["evidence"])
        st.markdown(rc["answer"])
        st.caption("Correlation is not causation: hypotheses need to be tested before acting on them.")


# ----------------------------------------------------------------------------- relationships
def render_relationships(ctx):
    if len(ctx.dfs) < 2:
        st.info("Upload at least two datasets (or load the sample data) to discover relationships.")
        return
    rels = A.find_relationships(ctx.dfs)
    if not rels:
        st.info("No shared columns with overlapping values were found.")
        return
    st.dataframe(pd.DataFrame(rels), width="stretch")
    st.subheader("🔗 Join preview")
    names = list(ctx.dfs)
    c = st.columns(4)
    left = c[0].selectbox("Left", names, key="j_left")
    right = c[1].selectbox("Right", [n for n in names if n != left], key="j_right")
    keys = sorted(set(ctx.dfs[left].columns) & set(ctx.dfs[right].columns))
    if not keys:
        st.warning("These two datasets share no column names.")
    else:
        key = c[2].selectbox("Join key", keys, key="j_key")
        how = c[3].selectbox("Join type", ["left", "inner", "outer"], key="j_how")
        merged, nl, nr, nm = A.join_preview(ctx.dfs[left], ctx.dfs[right], key, how)
        st.caption(f"Left rows: {nl:,} · Right rows: {nr:,} → Result rows: {nm:,}")
        if nm > max(nl, nr) * 1.05 and how != "outer":
            st.warning("Row count grew: the key is not unique on one side, so rows are being duplicated.")
        st.dataframe(merged.head(50), width="stretch")
        if st.button("➕ Use joined data as a new dataset", key="j_use"):
            st.session_state.derived[f"{left} ⋈ {right}"] = merged
            st.session_state.next_dataset = f"{left} ⋈ {right}"
            st.toast("Switched to the joined dataset")
            st.rerun()
    if ctx.router.configured and st.button("🤖 Ask AI for join advice", key="j_ai"):
        try:
            st.markdown(agent.relationship_advice(ctx.router, ctx.dfs))
        except LLMUnavailable as exc:
            st.error(str(exc))


# ----------------------------------------------------------------------------- statistics
def render_statistics(ctx):
    df = ctx.df
    t1, t2, t3 = st.tabs(["Correlation", "Outliers", "Summary"])
    nums = [c for c, r in A.infer_roles(df).items() if r == "numeric"]
    with t1:
        if len(nums) < 2:
            st.info("At least two numeric columns are needed.")
        else:
            st.plotly_chart(px.imshow(df[nums].corr(), text_auto=".2f", aspect="auto", template=T,
                                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation matrix"),
                            width="stretch")
    with t2:
        if not nums:
            st.info("No numeric columns.")
        else:
            c1, c2 = st.columns([3, 1])
            col = c1.selectbox("Column", nums, key=f"out_col_{ctx.name}")
            k = c2.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.5, key="out_k")
            q1, q3, lo, hi, mask = A.outliers_iqr(df[col], k)
            m = st.columns(4)
            m[0].metric("Q1", f"{q1:,.2f}")
            m[1].metric("Q3", f"{q3:,.2f}")
            m[2].metric("Bounds", f"{lo:,.1f} – {hi:,.1f}")
            m[3].metric("Outliers", int(mask.sum()))
            vals = pd.to_numeric(df[col], errors="coerce")
            fig = px.histogram(pd.DataFrame({col: vals, "Type": mask.map({True: "Outlier", False: "Normal"})}),
                               x=col, color="Type", nbins=60, template=T,
                               color_discrete_map={"Normal": "#78866B", "Outlier": "#A6533C"})
            for bound in (lo, hi):
                fig.add_vline(x=bound, line_dash="dash", line_color="#382E2A")
            fig.update_layout(title=f"{col}: values outside the dashed bounds are outliers", barmode="stack")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df[mask].head(100), width="stretch")
    with t3:
        st.dataframe(df.describe(include="all").transpose().astype(str), width="stretch")


# ----------------------------------------------------------------------------- copilot
def render_copilot(ctx):
    if not _need_llm(ctx):
        return
    with st.form("copilot_form"):
        q = st.text_area("Ask a business question", placeholder="What are the biggest opportunities in this data?")
        go_ = st.form_submit_button("🤖 Ask Copilot", type="primary")
    if go_ and q.strip():
        with st.spinner("Thinking…"):
            try:
                st.session_state.copilot_history.append({"q": q.strip(), "a": agent.copilot_answer(ctx.router, ctx.df, q.strip()), "ds": ctx.name})
            except LLMUnavailable as exc:
                st.error(str(exc))
    for item in reversed(st.session_state.copilot_history[-10:]):
        with st.expander(item["q"], expanded=item is st.session_state.copilot_history[-1]):
            st.markdown(item["a"])


# ----------------------------------------------------------------------------- reports
def _report_items(ctx) -> dict:
    ss, name, items = st.session_state, ctx.name, {}
    an = [o for o in ss.history if o.ok and o.dataset == name]
    if an:
        items["analyses"] = an
    fc = ss.get("forecast")
    if fc and fc.get("dataset") == name:
        items["forecast"] = fc
    if ss.cleaning.get(name):
        items["cleaning"] = ss.cleaning[name]["log"]
    if ss.get("sql_result") is not None and ss.get("sql_ds") == name:
        items["sql"] = (ss.get("sql_text", ""), ss.sql_result)
    rc = ss.get("root_cause")
    if rc and rc.get("dataset") == name:
        items["root_cause"] = rc
    cp = [x for x in ss.copilot_history if x.get("ds") == name]
    if cp:
        items["copilot"] = cp
    if ss.rag_history:
        items["rag"] = [{"q": x["q"], "a": x["a"]} for x in ss.rag_history]
    return items


def render_reports(ctx):
    avail = _report_items(ctx)
    st.caption("The report includes everything you have run on this dataset in this session. "
               "Run more tabs (AI Analyst, Forecast, Cleaning, SQL, Root Cause, Copilot, Documents) to make it richer.")
    chosen = st.multiselect("Sections to include", list(avail), default=list(avail),
                            format_func=lambda k: reports.LABELS[k],
                            key=f"rep_sections_{ctx.name}_{'-'.join(avail)}")  # new sections => fresh default (all ticked)
    items = {k: v for k, v in avail.items() if k in chosen}
    text = reports.build_text_report(ctx.name, ctx.df, items)
    with st.expander("Preview", expanded=not avail):
        st.text(text)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download text report", text, "executive_report.txt", "text/plain")
    if c2.button("📄 Build PDF report", key="pdf_build"):
        st.session_state.pdf_bytes = reports.build_pdf_report(ctx.name, ctx.df, items)
    if st.session_state.get("pdf_bytes"):
        c2.download_button("⬇ Download PDF", st.session_state.pdf_bytes, "AI_Data_Analyst_Report.pdf", "application/pdf")
    if not avail:
        st.info("Nothing from the AI features has been run yet, so the report only has the dataset overview.")

"""InsightForge - AI Data Analyst - Streamlit entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="InsightForge - AI Data Analyst", page_icon="📊", layout="wide")

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import analytics  # noqa: E402
import auth  # noqa: E402
import charts  # noqa: E402
import config  # noqa: E402
import views  # noqa: E402
from llm import LLMRouter, get_embedder  # noqa: E402

TABS = [
    ("dashboard", "📊 Dashboard", views.render_dashboard),
    ("ai_analyst", "🤖 AI Analyst", views.render_ai_analyst),
    ("rag", "🤖 Documents", views.render_rag),
    ("evaluation", "⚡ Evaluation", views.render_evaluation),
    ("sql", "🗄 SQL", views.render_sql),
    ("cleaning", "🧹 Cleaning", views.render_cleaning),
    ("forecast", "🔮 Forecast", views.render_forecast),
    ("root_cause", "🤖 Root Cause", views.render_root_cause),
    ("relationships", "🔗 Relationships", views.render_relationships),
    ("statistics", "📈 Statistics", views.render_statistics),
    ("copilot", "🤖 Copilot", views.render_copilot),
    ("reports", "📄 Reports", views.render_reports),
]
DOC_ONLY = {"rag"}  # tabs that work without a tabular dataset

DEFAULTS = {
    "use_sample": False, "derived": {}, "history": [], "last_outcome": None, "suggestions": {},
    "kb": None, "kb_sig": None, "kb_notes": [], "rag_history": [], "copilot_history": [],
    "cleaning": {},
}


@st.cache_resource(show_spinner=False)
def get_router() -> LLMRouter:
    return LLMRouter(config.provider_specs())


@st.cache_resource(show_spinner=False)
def get_embedder_cached():
    return get_embedder()


@st.cache_data(show_spinner=False, max_entries=32)
def cached_load(name: str, data: bytes):
    return analytics.load_file(name, data)


def collect_sources(uploaded):
    items = [(f.name, f.getvalue()) for f in uploaded or []]
    if st.session_state.use_sample:
        items += [(p.name, p.read_bytes()) for p in sorted(config.DATA_DIR.glob("sample_*"))]
    dfs, docs, problems = {}, [], []
    for name, data in items:
        ext = Path(name).suffix.lower()
        if len(data) > config.MAX_UPLOAD_MB * 1_048_576:
            problems.append(f"{name}: larger than {config.MAX_UPLOAD_MB} MB.")
        elif ext in analytics.TABULAR_EXT:
            try:
                label = name if name not in dfs else f"{name} (2)"
                dfs[label] = cached_load(name, data)
            except Exception as exc:
                problems.append(f"{name}: {exc}")
        elif ext in analytics.DOC_EXT:
            docs.append((name, data))
    dfs.update(st.session_state.derived)
    return dfs, docs, problems


def main():
    charts.register_template()
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    st.sidebar.title("📊 InsightForge - AI Data Analyst")
    user = auth.require_login()
    router, embedder = get_router(), get_embedder_cached()

    if auth.mode() == "supabase":
        st.sidebar.success(f"{user.email}\nRole: **{user.role}**")
        if st.sidebar.button("🚪 Log out"):
            auth.logout()
    else:
        st.sidebar.info(f"Role: **{user.role}**")
    if router.configured:
        st.sidebar.caption("🤖 Model: " + ", ".join(router.provider_names[:2]))
    else:
        st.sidebar.warning("No LLM key configured: AI features are disabled (see working.md).")
    for err in router.init_errors:
        st.sidebar.warning(err)

    st.sidebar.divider()
    st.sidebar.subheader("📁 Data")
    uploaded = st.sidebar.file_uploader("Upload CSV, Excel, JSON, PDF or TXT", accept_multiple_files=True,
                                        type=["csv", "xlsx", "xls", "json", "pdf", "txt"])
    st.session_state.use_sample = st.sidebar.toggle("Use sample data", value=st.session_state.use_sample,
                                                    help="Sales + customers tables and a short company report.")
    dfs, docs, problems = collect_sources(uploaded)
    for p in problems:
        st.sidebar.error(p)

    if st.session_state.history:
        st.sidebar.divider()
        st.sidebar.subheader("📝 Analysis history")
        for o in reversed(st.session_state.history[-8:]):
            with st.sidebar.expander(o.question[:55]):
                st.write(agent_preview(o))
        if st.sidebar.button("🗑 Clear history"):
            st.session_state.history, st.session_state.last_outcome = [], None
            st.rerun()

    st.title("📊 InsightForge - AI Data Analyst")
    st.caption("Upload data → ask questions in plain English → verified calculations → insights, reports and forecasts")
    if not dfs and not docs:
        st.info("👈 Upload a dataset (CSV / Excel / JSON) or a document (PDF / TXT) in the sidebar, "
                "or switch on **Use sample data** to explore the app.")
        st.stop()

    name, df = None, None
    if dfs:
        if "next_dataset" in st.session_state:  # set by Cleaning / Relationships tabs
            st.session_state.dataset_picker = st.session_state.pop("next_dataset")
        name = st.selectbox("📂 Dataset", list(dfs), key="dataset_picker")
        df = dfs[name]
    ctx = SimpleNamespace(router=router, df=df, name=name, dfs=dfs, docs=docs, embedder=embedder, role=user.role)

    allowed = [t for t in TABS if auth.can(user.role, t[0])]
    for tab, (feature, _label, fn) in zip(st.tabs([t[1] for t in allowed]), allowed):
        with tab:
            if df is None and feature not in DOC_ONLY:
                st.info("Upload a tabular dataset to use this feature.")
            else:
                try:
                    fn(ctx)
                except Exception as exc:  # keep one broken tab from taking down the page
                    st.error(f"This section hit an unexpected error: {exc}")


def agent_preview(o) -> str:
    if not o.ok:
        return f"❌ {o.error[:200]}"
    return o.result.head(5).to_string(index=False) if isinstance(o.result, pd.DataFrame) else str(o.result)[:300]


main()

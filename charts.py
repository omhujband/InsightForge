"""Plotly helpers: brand theme and automatic chart choice from a result's shape."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

TEMPLATE = "terracotta_cream"
_PALETTE = ["#A6533C", "#C77D68", "#D7B98F", "#78866B", "#8F6B55"]


def register_template() -> None:
    if TEMPLATE in pio.templates:
        return
    t = go.layout.Template(pio.templates["plotly_white"])
    t.layout.update(paper_bgcolor="#FBF7F0", plot_bgcolor="#FBF7F0", font={"color": "#382E2A"},
                    colorway=_PALETTE, xaxis={"gridcolor": "#E7DCCE"}, yaxis={"gridcolor": "#E7DCCE"})
    pio.templates[TEMPLATE] = t


def auto_chart(df: pd.DataFrame):
    """Pick a sensible chart from the result's shape (no LLM guess). None if not chartable."""
    if not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] < 1:
        return None
    register_template()
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not nums:
        return None
    first = df.columns[0]
    d = df.head(40)
    if first in nums and len(nums) == 1:
        return px.histogram(df, x=first, template=TEMPLATE)
    others = [c for c in nums if c != first]
    if not others:
        return None
    x = first
    if pd.api.types.is_datetime64_any_dtype(df[x]) or "date" in str(x).lower():
        return px.line(df.sort_values(x), x=x, y=others[:3], template=TEMPLATE, markers=True)
    if x in nums:
        return px.scatter(df, x=x, y=others[0], template=TEMPLATE)
    return px.bar(d, x=x, y=others[:3], barmode="group", template=TEMPLATE)

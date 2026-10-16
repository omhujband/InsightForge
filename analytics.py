"""Deterministic data logic: loading, profiling, KPIs, cleaning, relationships, forecasting."""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

_DATE_RE = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")
_ID_NAME_RE = re.compile(r"(^|[_\s])(id|uuid|guid|key|idx|index|ref)$|^(id|uuid)[_\s]", re.I)
_MEAN_RE = re.compile(r"price|rate|rating|score|age|discount|percent|ratio|margin|temperature", re.I)
_SUM_RE = re.compile(r"revenue|sales|amount|profit|cost|total|quantity|qty|units|spend|income|expense|orders|count", re.I)
_PLACEHOLDERS = {"", "n/a", "na", "null", "none", "nan", "?", "#n/a", "missing"}

TABULAR_EXT = (".csv", ".xlsx", ".xls", ".json")
DOC_EXT = (".pdf", ".txt")


# ----------------------------------------------------------------------------- loading
def load_file(name: str, data: bytes) -> pd.DataFrame:
    ext = Path(name).suffix.lower()
    if ext == ".csv":
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(BytesIO(data), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode the CSV file.")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(BytesIO(data))
    elif ext == ".json":
        try:
            df = pd.read_json(BytesIO(data))
        except ValueError:
            df = pd.read_json(BytesIO(data), lines=True)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    if df.empty:
        raise ValueError("The file contains no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_text(s: pd.Series) -> bool:
    return (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)) and not pd.api.types.is_bool_dtype(s)


def as_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", format="mixed")


# ----------------------------------------------------------------------------- profiling
def infer_roles(df: pd.DataFrame) -> dict[str, str]:
    """Classify each column: id | numeric | datetime | categorical | text | boolean."""
    roles: dict[str, str] = {}
    for c in df.columns:
        s, nn = df[c], df[c].dropna()
        if pd.api.types.is_bool_dtype(s):
            roles[c] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            roles[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            ratio = nn.nunique() / max(len(nn), 1)
            is_int = pd.api.types.is_integer_dtype(s) or (len(nn) > 0 and bool((nn % 1 == 0).all()))
            if _ID_NAME_RE.search(str(c)) or (is_int and ratio > 0.98 and len(nn) > 20):
                roles[c] = "id"
            else:
                roles[c] = "numeric"
        else:
            sample = nn.astype(str).head(200)
            if len(sample) and sample.str.match(_DATE_RE).mean() >= 0.9:
                roles[c] = "datetime"
            elif _ID_NAME_RE.search(str(c)):
                roles[c] = "id"
            elif nn.nunique() <= max(50, 0.05 * len(nn)):
                roles[c] = "categorical"
            else:
                roles[c] = "text"
    return roles


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    roles = infer_roles(df)
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append({"Column": c, "Role": roles[c], "Dtype": str(s.dtype),
                     "Non-null": int(s.notna().sum()), "Missing": int(s.isna().sum()),
                     "Missing %": round(100 * s.isna().mean(), 1), "Unique": int(s.nunique()),
                     "Example": str(ex)[:40]})
    return pd.DataFrame(rows)


def kpi_candidates(df: pd.DataFrame, max_n: int = 4) -> list[tuple[str, float]]:
    """Pick meaningful KPIs: skip identifiers; sum money/volume columns, average rates."""
    nums = [c for c, r in infer_roles(df).items() if r == "numeric"]
    nums.sort(key=lambda c: (0 if _SUM_RE.search(c) and not _MEAN_RE.search(c) else 1))
    out = []
    for c in nums[:max_n]:
        if _SUM_RE.search(c) and not _MEAN_RE.search(c):
            out.append((f"Total {c}", float(df[c].sum())))
        else:
            out.append((f"Avg {c}", float(df[c].mean())))
    return out


def _col_summary(s: pd.Series, role: str) -> str:
    miss = int(s.isna().sum())
    if role in ("numeric", "id") and pd.api.types.is_numeric_dtype(s):
        d = f"min={s.min():.4g}, mean={s.mean():.4g}, max={s.max():.4g}"
    elif role == "datetime":
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
        d = f"range={dt.min()}..{dt.max()}"
    elif role in ("categorical", "boolean"):
        d = "top=" + ", ".join(f"{k}({v})" for k, v in s.astype(str).value_counts().head(5).items())
    else:
        d = f"e.g. {str(s.dropna().iloc[0])[:40]!r}" if s.notna().any() else "empty"
    return f"{d}; missing={miss}"


def dataset_brief(df: pd.DataFrame, max_cols: int = 40, sample_rows: int = 5) -> str:
    """Compact text description of a dataset, used in LLM prompts."""
    roles = infer_roles(df)
    lines = [f"Rows: {len(df)}; Columns: {df.shape[1]}"]
    for c in list(df.columns)[:max_cols]:
        lines.append(f"- {c} [{df[c].dtype}, {roles[c]}]: {_col_summary(df[c], roles[c])}")
    sample = df.iloc[:sample_rows, :15].to_string(max_colwidth=30)
    return "\n".join(lines) + f"\nSample rows:\n{sample}"


# ----------------------------------------------------------------------------- statistics
def outliers_iqr(s: pd.Series, k: float = 1.5):
    s = pd.to_numeric(s, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return q1, q3, lo, hi, (s < lo) | (s > hi)


def build_evidence(df: pd.DataFrame, target: str | None = None) -> str:
    """Computed facts that ground root-cause analysis (no LLM involved)."""
    roles = infer_roles(df)
    nums = [c for c, r in roles.items() if r == "numeric"]
    cats = [c for c, r in roles.items() if r == "categorical" and df[c].nunique() <= 30]
    dates = [c for c, r in roles.items() if r == "datetime"]
    target = target if target in nums else (nums[0] if nums else None)
    out: list[str] = []
    if target:
        t = pd.to_numeric(df[target], errors="coerce")
        out.append(f"Target metric: {target} (n={int(t.notna().sum())}, mean={t.mean():.4g}, "
                   f"median={t.median():.4g}, std={t.std():.4g})")
        corr = df[nums].corr()[target].drop(target, errors="ignore").dropna()
        if len(corr):
            top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(5)
            out.append("Strongest correlations with target: " + ", ".join(f"{k}={v:.2f}" for k, v in top.items()))
        for c in cats[:3]:
            g = t.groupby(df[c]).agg(["mean", "count"]).sort_values("mean")
            if len(g) >= 2:
                lo, hi = g.iloc[0], g.iloc[-1]
                out.append(f"By {c}: lowest mean {g.index[0]}={lo['mean']:.4g} (n={int(lo['count'])}), "
                           f"highest {g.index[-1]}={hi['mean']:.4g} (n={int(hi['count'])})")
        if dates:
            ts = pd.DataFrame({"d": as_datetime(df, dates[0]), "y": t}).dropna().set_index("d")["y"]
            m = ts.resample("MS").sum() if len(ts) else ts
            if len(m) >= 3:
                last = m.iloc[-3:]
                out.append("Last 3 periods (monthly sum): " + ", ".join(f"{i:%Y-%m}={v:.4g}" for i, v in last.items()))
    else:
        out.append("No numeric target column available.")
    return "\n".join(out)


# ----------------------------------------------------------------------------- relationships
def find_relationships(dfs: dict[str, pd.DataFrame]) -> list[dict]:
    names, out = list(dfs), []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for col in sorted(set(dfs[a].columns) & set(dfs[b].columns)):
                sa, sb = dfs[a][col].dropna().astype(str), dfs[b][col].dropna().astype(str)
                if sa.empty or sb.empty:
                    continue
                ua, ub = set(sa.unique()), set(sb.unique())
                overlap = len(ua & ub) / max(min(len(ua), len(ub)), 1)
                if overlap == 0:
                    continue
                a_unique, b_unique = sa.is_unique, sb.is_unique
                kind = ("one-to-one" if a_unique and b_unique else "one-to-many" if a_unique
                        else "many-to-one" if b_unique else "many-to-many")
                out.append({"left": a, "right": b, "column": col, "value_overlap": round(overlap, 3),
                            "relationship": f"{a} → {b}: {kind}"})
    return out


def join_preview(left: pd.DataFrame, right: pd.DataFrame, key: str, how: str):
    merged = left.merge(right, on=key, how=how, suffixes=("", "_r"))
    return merged, len(left), len(right), len(merged)


# ----------------------------------------------------------------------------- cleaning
@dataclass
class CleaningAction:
    key: str
    title: str
    count: int
    default: bool


def _text_cols(df):
    return [c for c in df.columns if is_text(df[c])]


def _placeholder_mask(s: pd.Series) -> pd.Series:
    return s.notna() & s.astype(str).str.strip().str.lower().isin(_PLACEHOLDERS)


def _numeric_text_cols(df):
    cols = []
    for c in _text_cols(df):
        if _ID_NAME_RE.search(c):
            continue
        nn = df[c].dropna().astype(str)
        if nn.empty:
            continue
        conv = pd.to_numeric(nn.str.replace(r"[,\s$₹€£%]", "", regex=True), errors="coerce")
        if conv.notna().mean() >= 0.95 and conv.notna().any():
            cols.append(c)
    return cols


def _date_text_cols(df):
    return [c for c in _text_cols(df) if df[c].dropna().astype(str).head(200).str.match(_DATE_RE).mean() >= 0.9
            and df[c].notna().any()]


def propose_cleaning(df: pd.DataFrame) -> list[CleaningAction]:
    text = _text_cols(df)
    padded = sum(int((df[c].dropna().astype(str) != df[c].dropna().astype(str).str.strip()).sum()) for c in text)
    placeholders = sum(int(_placeholder_mask(df[c]).sum()) for c in text)
    dups = int(df.duplicated().sum())
    num_missing = sum(int(df[c].isna().sum()) for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    cat_missing = sum(int(df[c].isna().sum()) for c in text)
    actions = [
        CleaningAction("strip", "Trim leading/trailing spaces in text cells", padded, True),
        CleaningAction("placeholders", "Turn placeholders (N/A, null, none, nan, ?, empty) into real missing values", placeholders, True),
        CleaningAction("numeric_text", f"Convert text columns that are really numbers: {', '.join(_numeric_text_cols(df)) or 'none'}", len(_numeric_text_cols(df)), True),
        CleaningAction("dates", f"Convert text columns that are really dates: {', '.join(_date_text_cols(df)) or 'none'}", len(_date_text_cols(df)), True),
        CleaningAction("dups", "Remove exact duplicate rows", dups, True),
        CleaningAction("fill_num", "Fill missing numeric values with the column median (imputation - changes statistics)", num_missing, False),
        CleaningAction("fill_cat", "Fill missing text values with 'Unknown' (keeps missingness visible)", cat_missing, False),
    ]
    return [a for a in actions if a.count > 0]


def apply_cleaning(df: pd.DataFrame, keys: list[str]):
    """Apply selected actions in a safe order. Returns (cleaned_df, audit_log)."""
    out, log = df.copy(), []
    if "strip" in keys:
        n = 0
        for c in _text_cols(out):
            s = out[c].dropna().astype(str)
            n += int((s != s.str.strip()).sum())
            out[c] = out[c].where(out[c].isna(), out[c].astype(str).str.strip())
        log.append(f"Trimmed spaces in {n} cells")
    if "placeholders" in keys:
        n = 0
        for c in _text_cols(out):
            m = _placeholder_mask(out[c])
            n += int(m.sum())
            out.loc[m, c] = np.nan
        log.append(f"Replaced {n} placeholder cells with missing values")
    if "numeric_text" in keys:
        for c in _numeric_text_cols(out):
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(r"[,\s$₹€£%]", "", regex=True), errors="coerce")
            log.append(f"Converted '{c}' to numeric")
    if "dates" in keys:
        for c in _date_text_cols(out):
            out[c] = pd.to_datetime(out[c], errors="coerce", format="mixed")
            log.append(f"Converted '{c}' to datetime")
    if "dups" in keys:
        n = int(out.duplicated().sum())
        out = out.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {n} duplicate rows")
    if "fill_num" in keys:
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]) and out[c].isna().any():
                n = int(out[c].isna().sum())
                out[c] = out[c].fillna(out[c].median())
                log.append(f"Filled {n} missing in '{c}' with median")
    if "fill_cat" in keys:
        for c in _text_cols(out):
            if out[c].isna().any():
                n = int(out[c].isna().sum())
                out[c] = out[c].fillna("Unknown")
                log.append(f"Filled {n} missing in '{c}' with 'Unknown'")
    return out, log


# ----------------------------------------------------------------------------- forecasting
FREQS = {"Daily": ("D", 7), "Weekly": ("W", 52), "Monthly": ("MS", 12)}


def build_series(df, date_col, target_col, freq, agg="sum", drop_last=True) -> pd.Series:
    d = pd.DataFrame({"ds": as_datetime(df, date_col),
                      "y": pd.to_numeric(df[target_col], errors="coerce")}).dropna()
    if d.empty:
        raise ValueError("No valid (date, value) rows to forecast from.")
    r = d.set_index("ds")["y"].sort_index().resample(freq)
    ser = r.sum() if agg == "sum" else r.mean().interpolate()
    if drop_last and freq != "D" and len(ser) > 2:
        ser = ser.iloc[:-1]  # last bucket is usually incomplete
    return ser


def _trend_seasonal(y: np.ndarray, h: int, m: int):
    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 data points.")
    t = np.arange(n)
    slope, icpt = np.polyfit(t, y, 1)
    detr = y - (slope * t + icpt)
    seas = np.zeros(1)
    if m > 1 and n >= 2 * m:
        seas = np.array([detr[i::m].mean() for i in range(m)])
        seas -= seas.mean()
    fitted = slope * t + icpt + seas[t % len(seas)]
    sigma = float(np.std(y - fitted))
    tf = np.arange(n, n + h)
    yhat = slope * tf + icpt + seas[tf % len(seas)]
    return yhat, yhat - 1.96 * sigma, yhat + 1.96 * sigma


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
        return True
    except Exception:
        return False


def _predict(ser: pd.Series, h: int, freq: str, m: int, model: str):
    if model == "prophet":
        from prophet import Prophet
        mdl = Prophet()
        mdl.fit(pd.DataFrame({"ds": ser.index, "y": ser.values}))
        f = mdl.predict(mdl.make_future_dataframe(h, freq=freq)).tail(h)
        return f["yhat"].to_numpy(), f["yhat_lower"].to_numpy(), f["yhat_upper"].to_numpy()
    return _trend_seasonal(ser.to_numpy(dtype=float), h, m)


def forecast(ser: pd.Series, horizon: int, freq: str, m: int, model: str = "builtin"):
    """Returns (forecast_df, backtest_dict_or_None). Backtest compares against a naive baseline."""
    yhat, lo, hi = _predict(ser, horizon, freq, m, model)
    idx = pd.date_range(ser.index[-1], periods=horizon + 1, freq=freq)[1:]
    fdf = pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lo, "yhat_upper": hi})

    n, k = len(ser), max(3, min(horizon, len(ser) // 4, 24))
    bt = None
    if n - k >= 6:
        train, test = ser.iloc[:-k], ser.iloc[-k:].to_numpy(dtype=float)
        pred, _, _ = _predict(train, k, freq, m, model)
        naive = np.repeat(float(train.iloc[-1]), k)
        mae, nmae = float(np.mean(abs(test - pred))), float(np.mean(abs(test - naive)))
        nz = test != 0
        bt = {"holdout_periods": k, "mae": mae, "naive_mae": nmae,
              "mape_pct": float(np.mean(abs(test[nz] - pred[nz]) / abs(test[nz])) * 100) if nz.any() else None,
              "improvement_vs_naive_pct": (1 - mae / nmae) * 100 if nmae > 0 else None}
    return fdf, bt

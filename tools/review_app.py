"""
Review dashboard — visually inspect the output of every pipeline stage
without manually grepping JSON/parquet files.

Run:
    uv add --optional review streamlit      # first time only
    uv run streamlit run tools/review_app.py

Reads directly from your data/ folders (read-only) -- rerun the pipeline,
then just refresh the browser tab.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# allow "from common.io_utils import ..." style imports and running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.io_utils import load_config, load_json  # noqa: E402

st.set_page_config(page_title="SBERT Legal Pipeline Review", layout="wide")


@st.cache_data
def get_config():
    return load_config(str(ROOT / "config.yaml"))


@st.cache_data
def list_judgment_ids(_cfg: dict) -> list[str]:
    seg_dir = ROOT / _cfg["paths"]["stage0_dir"]
    if not seg_dir.exists():
        return []
    return sorted(p.stem.replace("_segments", "") for p in seg_dir.glob("*_segments.json"))


@st.cache_data
def load_stage_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return load_json(p)


@st.cache_data
def load_parquet_cached(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


_jinja2_warned = {"shown": False}


def safe_styled_dataframe(df: pd.DataFrame, style_fn, **kwargs):
    """Applies row-highlighting via pandas .style, which requires jinja2 --
    an optional dependency that's easy to miss (it's not pulled in by
    streamlit or pandas by default). Falls back to a plain, unstyled
    dataframe instead of crashing the whole page if jinja2 isn't installed,
    and shows a one-time hint on how to fix it."""
    try:
        st.dataframe(df.style.apply(style_fn, axis=1), **kwargs)
    except (ImportError, AttributeError):
        if not _jinja2_warned["shown"]:
            st.info("Row highlighting needs `jinja2` (not installed) -- showing plain table instead. "
                     "Run `uv pip install jinja2` (or `uv add --optional review jinja2`) and refresh to enable it.")
            _jinja2_warned["shown"] = True
        st.dataframe(df, **kwargs)


cfg = get_config()
jids = list_judgment_ids(cfg)

st.sidebar.title("Pipeline Review")
if st.sidebar.button("🔄 Refresh (clear cache)"):
    st.cache_data.clear()
    st.rerun()

if not jids:
    st.warning(f"No segments found under `{cfg['paths']['stage0_dir']}`. "
               f"Run Stage 0 first: `uv run python src/stage0_merge_segments.py`")
    st.stop()

jid = st.sidebar.selectbox("Judgment ID", jids)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Stage 0: Segments", "Stage 1: Alignments", "Stage 2: Queries",
     "Stage 3: Training Table", "Stage 4: Splits", "Cross-Judgment Summary"]
)

# ---------------- Stage 0 ----------------
with tab0:
    st.subheader(f"Segments — judgment {jid}")
    segs = load_stage_json(str(ROOT / cfg["paths"]["stage0_dir"] / f"{jid}_segments.json"))
    if segs is None:
        st.info("Not generated yet.")
    else:
        df = pd.DataFrame(segs)
        c1, c2 = st.columns(2)
        c1.metric("Segments", len(df))
        c2.metric("Roles present", df["role"].nunique() if len(df) else 0)
        role_filter = st.multiselect("Filter by role", sorted(df["role"].unique()) if len(df) else [])
        view = df[df["role"].isin(role_filter)] if role_filter else df
        for _, row in view.iterrows():
            with st.expander(f"[{row['role']}] {row['segment_id']} — {row['text'][:80]}..."):
                st.write(row["text"])

# ---------------- Stage 1 ----------------
with tab1:
    st.subheader(f"Alignments — judgment {jid}")
    aligns = load_stage_json(str(ROOT / cfg["paths"]["stage1_dir"] / f"{jid}_alignments.json"))
    if aligns is None:
        st.info("Not generated yet.")
    elif len(aligns) == 0:
        st.warning("Zero alignments for this judgment.")
    else:
        df = pd.DataFrame(aligns)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total alignments", len(df))
        if "role_match" in df.columns:
            c2.metric("Same-role matches", int(df["role_match"].sum()))
            c3.metric("Cross-role matches", int((~df["role_match"]).sum()),
                      help="Matches the old same-role-restricted search would have missed entirely.")

        show_cols = [c for c in ["summary_role", "summary_field", "segment_id", "segment_role",
                                  "role_match", "cosine_score", "bm25_score", "fused_score",
                                  "similarity_score"] if c in df.columns]
        if "role_match" in df.columns:
            def _highlight_cross_role(row):
                return ["background-color: #fff3cd" if show_cols[i] == "role_match" and not row.get("role_match", True)
                        else "" for i in range(len(show_cols))]
            safe_styled_dataframe(df[show_cols], _highlight_cross_role, use_container_width=True, height=350)
        else:
            st.dataframe(df[show_cols], use_container_width=True, height=350)

        st.markdown("**Row detail**")
        idx = st.number_input("Row index", 0, len(df) - 1, 0)
        row = df.iloc[idx]
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**Summary sentence** _(role: {row.get('summary_role', row.get('role'))}, "
                        f"field: {row.get('summary_field', '?')})_")
            st.write(row["summary_sentence"])
        with colB:
            st.markdown(f"**Matched segment** _(role: {row.get('segment_role', '?')}, id: {row['segment_id']})_")
            st.write(row["segment_text"])

# ---------------- Stage 2 ----------------
with tab2:
    st.subheader(f"Generated queries — judgment {jid}")
    queries = load_stage_json(str(ROOT / cfg["paths"]["stage2_dir"] / f"{jid}_queries.json"))
    if queries is None:
        st.info("Not generated yet.")
    else:
        rows = []
        for rec in queries:
            for style, info in rec.get("queries", {}).items():
                rows.append({
                    "role": rec.get("summary_role", rec.get("role")),
                    "segment_id": rec["segment_id"],
                    "style": style,
                    "text": info["text"],
                    "word_count": info["word_count"],
                    "valid": info["valid"],
                    "used_fallback": info.get("used_fallback", False),
                    "fallback_reason": info.get("fallback_reason") or "",
                })
        qdf = pd.DataFrame(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total query fields", len(qdf))
        c2.metric("Valid", int(qdf["valid"].sum()) if len(qdf) else 0)
        c3.metric("Fell back to template", int(qdf["used_fallback"].sum()) if "used_fallback" in qdf.columns else 0)

        if "fallback_reason" in qdf.columns and qdf["used_fallback"].any():
            st.markdown("**Fallback reason breakdown** _(a field can have more than one reason, e.g. "
                        "both a duplicate AND a case-specific detail)_")
            reason_counts = (qdf[qdf["used_fallback"]]["fallback_reason"]
                              .str.split(",").explode().value_counts())
            st.bar_chart(reason_counts)

        def highlight(row):
            if not row["valid"]:
                return ["background-color: #f8d7da"] * len(row)
            if row.get("used_fallback"):
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        safe_styled_dataframe(qdf, highlight, use_container_width=True, height=450)

        dup_check = qdf[qdf["valid"]].groupby(["style", "text"]).size().reset_index(name="count")
        dups = dup_check[dup_check["count"] > 1]
        if len(dups):
            st.warning("Duplicate query text detected within this judgment:")
            st.dataframe(dups, use_container_width=True)

# ---------------- Stage 3 ----------------
with tab3:
    st.subheader("Training table")
    df3 = load_parquet_cached(str(ROOT / cfg["paths"]["stage3_dir"] / "training_table.parquet"))
    if df3 is None:
        st.info("Not generated yet.")
    else:
        this_jid = df3[df3["judgment_id"] == jid]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows (all judgments)", len(df3))
        c2.metric(f"Rows for {jid}", len(this_jid))
        c3.metric("Judgments in table", df3["judgment_id"].nunique())

        colA, colB = st.columns(2)
        with colA:
            st.bar_chart(df3["positive_source"].value_counts())
        with colB:
            st.bar_chart(df3["query_style"].value_counts())

        st.dataframe(this_jid, use_container_width=True, height=350)

# ---------------- Stage 4 ----------------
with tab4:
    st.subheader("Train / Val / Test splits")
    split_info = []
    for split_name in ["train", "val", "test"]:
        sdf = load_parquet_cached(str(ROOT / cfg["paths"]["stage3_dir"] / f"{split_name}.parquet"))
        if sdf is not None:
            split_info.append({
                "split": split_name,
                "rows": len(sdf),
                "judgments": sdf["judgment_id"].nunique(),
            })
    if not split_info:
        st.info("Not generated yet.")
    else:
        st.dataframe(pd.DataFrame(split_info), use_container_width=True)
        st.caption("Splitting is by judgment_id (never by row) to avoid leakage — "
                   "with only 1-2 judgments, everything may land in a single split; "
                   "this resolves once you run the full validation batch.")

# ---------------- Cross-judgment summary ----------------
with tab5:
    st.subheader("Cross-judgment summary (all judgments currently processed)")
    all_aligns = []
    all_queries = []
    for j in jids:
        a = load_stage_json(str(ROOT / cfg["paths"]["stage1_dir"] / f"{j}_alignments.json"))
        if a:
            all_aligns.extend(a)
        q = load_stage_json(str(ROOT / cfg["paths"]["stage2_dir"] / f"{j}_queries.json"))
        if q:
            for rec in q:
                for style, info in rec.get("queries", {}).items():
                    all_queries.append({"judgment_id": rec.get("judgment_id", j),
                                         "role": rec.get("summary_role", rec.get("role")),
                                         "style": style, "valid": info["valid"]})

    if all_aligns:
        adf = pd.DataFrame(all_aligns)
        st.markdown("**Alignment score by role**")
        score_col = "cosine_score" if "cosine_score" in adf.columns else "similarity_score"
        st.dataframe(adf.groupby("summary_role" if "summary_role" in adf.columns else "role")[score_col]
                     .agg(["count", "mean", "min", "max"]).round(3), use_container_width=True)

    if all_queries:
        qdf = pd.DataFrame(all_queries)
        st.markdown("**Query validity rate by role**")
        rate = qdf.groupby("role")["valid"].mean().round(3).rename("valid_rate")
        st.dataframe(rate, use_container_width=True)

    if not all_aligns and not all_queries:
        st.info("Run more judgments through the pipeline to populate this view.")

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu, spearmanr, t
from statsmodels.stats.multitest import multipletests

APP_TITLE = "Yachida 2019 CRC metabolomics + metagenomics"
DEFAULT_DATA = Path("data/yachida_simplified.xlsx")
CACHE_VERSION = "v2"
CACHE_DIR = Path("data/.cache")
ID_COL = "ID"
GROUP_COL = "Group"
STAGE_COL = "Stage"
LOCATION_COL = "Tumor location"
HEALTHY_LABEL = "Healthy"
PSEUDOCOUNT = 1e-9
STAGE_MP_LABEL = "Stage_MP"
STAGE_0_LABEL = "Stage_0"
STAGE_I_II_LABEL = "StageI_II"
STAGE_III_IV_LABEL = "Stage_III_IV"
RIGHT_SIDE_LABEL = "Right side"
LEFT_SIDE_LABEL = "Left side"
RECTUM_LABEL = "Rectum"
EXCLUDED_STAGE_LABELS = {"Stage_HS"}
STAGE_LABEL_RENAMES = {
    "MP": STAGE_MP_LABEL,
    "Stage_o": STAGE_0_LABEL,
    "Stage_I_II": STAGE_I_II_LABEL,
}
LOCATION_LABEL_RENAMES = {
    "Right colon": RIGHT_SIDE_LABEL,
    "Right side": RIGHT_SIDE_LABEL,
    "Left colon": LEFT_SIDE_LABEL,
    "Left side": LEFT_SIDE_LABEL,
    "Rectum": RECTUM_LABEL,
}
CANCER_GROUPS = {STAGE_0_LABEL, STAGE_I_II_LABEL, STAGE_III_IV_LABEL}
INVASIVE_CRC_GROUPS = {STAGE_I_II_LABEL, STAGE_III_IV_LABEL}
GROUP_ORDER = [HEALTHY_LABEL, STAGE_MP_LABEL, STAGE_0_LABEL, STAGE_I_II_LABEL, STAGE_III_IV_LABEL]
CRC_ENRICHMENT_ORDER = [HEALTHY_LABEL, STAGE_I_II_LABEL, STAGE_III_IV_LABEL]
LOCATION_ORDER = [RIGHT_SIDE_LABEL, LEFT_SIDE_LABEL, RECTUM_LABEL]
LOCATION_STAGE_COL = "Tumor location + Stage"

INTERESTING_SPECIES = [
    "Fusobacterium nucleatum",
    "Veillonella parvula",
    "Veillonella atypica",
    "Veillonella dispar",
    "Peptostreptococcus anaerobius",
    "Parvimonas micra",
    "Bacteroides fragilis",
    "Escherichia coli",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def normalize_stage_labels(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    for col in [GROUP_COL, STAGE_COL]:
        if col in meta.columns:
            meta[col] = meta[col].replace(STAGE_LABEL_RENAMES)
            meta = meta[~meta[col].isin(EXCLUDED_STAGE_LABELS)]
    return meta


def normalize_location_labels(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    if LOCATION_COL in meta.columns:
        meta[LOCATION_COL] = meta[LOCATION_COL].replace(LOCATION_LABEL_RENAMES)
    return meta


def single_location_meta(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[meta[LOCATION_COL].isin(LOCATION_ORDER)].copy()


def invasive_location_stage_meta(meta: pd.DataFrame) -> pd.DataFrame:
    healthy = meta[meta[GROUP_COL] == HEALTHY_LABEL]
    invasive = single_location_meta(meta[meta[STAGE_COL].isin(INVASIVE_CRC_GROUPS)])
    return (
        pd.concat([healthy, invasive], ignore_index=True)
        .drop_duplicates(subset=[ID_COL])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="Loading workbook...")
def load_data_from_path(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = Path(path)
    stat = source.stat()
    cache_key = f"{source.stem}_{CACHE_VERSION}_{stat.st_mtime_ns}_{stat.st_size}"
    cache_path = CACHE_DIR / cache_key
    meta_path = cache_path / "meta.pkl"
    metabolites_path = cache_path / "metabolites.pkl"
    species_path = cache_path / "species.pkl"

    if meta_path.exists() and metabolites_path.exists() and species_path.exists():
        try:
            return pd.read_pickle(meta_path), pd.read_pickle(metabolites_path), pd.read_pickle(species_path)
        except (OSError, ValueError, EOFError):
            pass

    meta, metabolites, species = load_data(path)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        meta.to_pickle(meta_path)
        metabolites.to_pickle(metabolites_path)
        species.to_pickle(species_path)
    except OSError:
        pass
    return meta, metabolites, species


@st.cache_data(show_spinner=False)
def load_data(file_or_path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = clean_columns(pd.read_excel(file_or_path, sheet_name="meta_data"))
    metabolites_raw = clean_columns(pd.read_excel(file_or_path, sheet_name="metabolites"))
    species_raw = clean_columns(pd.read_excel(file_or_path, sheet_name="species"))

    meta[ID_COL] = meta[ID_COL].astype(str)
    meta = normalize_stage_labels(meta)
    meta = normalize_location_labels(meta)
    metabolites_raw = metabolites_raw.rename(columns={metabolites_raw.columns[0]: "metabolites"})
    species_raw = species_raw.rename(columns={species_raw.columns[0]: "Species"})

    metabolites = metabolites_raw.set_index("metabolites")
    metabolites.columns = metabolites.columns.astype(str)
    metabolites = metabolites.apply(pd.to_numeric, errors="coerce").T
    metabolites.index = metabolites.index.astype(str)

    species = species_raw.set_index("Species")
    species.columns = species.columns.astype(str)
    species = species.apply(pd.to_numeric, errors="coerce").T
    species.index = species.index.astype(str)

    shared_ids = sorted(set(meta[ID_COL]) & set(metabolites.index))
    meta = meta[meta[ID_COL].isin(shared_ids)].copy()
    metabolites = metabolites.loc[shared_ids]

    shared_species_ids = sorted(set(meta[ID_COL]) & set(species.index))
    species = species.loc[shared_species_ids]

    return meta, metabolites, species


def available_locations(meta: pd.DataFrame) -> list[str]:
    return category_order(meta[LOCATION_COL].dropna().unique(), LOCATION_ORDER)


def filter_meta(
    meta: pd.DataFrame,
    groups: list[str] | None = None,
    locations: list[str] | None = None,
    stages: list[str] | None = None,
    genders: list[str] | None = None,
) -> pd.DataFrame:
    out = meta.copy()
    if groups:
        out = out[out[GROUP_COL].isin(groups)]
    if locations:
        out = out[out[LOCATION_COL].isin(locations)]
    if stages:
        out = out[out[STAGE_COL].isin(stages)]
    if genders:
        out = out[out["Gender"].isin(genders)]
    return out


def with_healthy_baseline(meta: pd.DataFrame, meta_sub: pd.DataFrame, groups: list[str], genders: list[str]) -> pd.DataFrame:
    if groups and HEALTHY_LABEL not in groups:
        return meta_sub
    healthy = filter_meta(meta, groups=[HEALTHY_LABEL], genders=genders)
    return (
        pd.concat([meta_sub, healthy], ignore_index=True)
        .drop_duplicates(subset=[ID_COL])
        .reset_index(drop=True)
    )


def ids_in_matrix(meta_sub: pd.DataFrame, matrix: pd.DataFrame) -> list[str]:
    return [sid for sid in meta_sub[ID_COL].astype(str).tolist() if sid in matrix.index]


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, len(d)
    rho, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return float(rho), float(p), int(len(d))


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    d = (np.mean(x) - np.mean(y)) / pooled
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return float(d * correction)


@st.cache_data(show_spinner=False)
def group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)[value_col]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    return out


def category_order(values: Iterable, preferred_order: Iterable[str] | None = None) -> list[str]:
    labels = [str(x) for x in values if pd.notna(x)]
    preferred = list(preferred_order or [])
    ordered = [label for label in preferred if label in labels]
    ordered.extend(sorted(label for label in set(labels) if label not in ordered))
    return ordered


def metadata_label(value, fallback: str) -> str:
    if pd.isna(value) or str(value) == "-":
        return fallback
    return str(value)


def add_location_stage_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    location = out[LOCATION_COL].apply(lambda value: metadata_label(value, "Unknown location"))
    stage = out[STAGE_COL].apply(lambda value: metadata_label(value, "Unknown stage"))
    out[LOCATION_STAGE_COL] = location + " | " + stage
    out.loc[out[GROUP_COL] == HEALTHY_LABEL, LOCATION_STAGE_COL] = HEALTHY_LABEL
    return out


def location_stage_order(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    if HEALTHY_LABEL in df[LOCATION_STAGE_COL].astype(str).values:
        labels.append(HEALTHY_LABEL)

    data = df[df[LOCATION_STAGE_COL] != HEALTHY_LABEL].copy()
    data["_location_label"] = data[LOCATION_COL].apply(lambda value: metadata_label(value, "Unknown location"))
    data["_stage_label"] = data[STAGE_COL].apply(lambda value: metadata_label(value, "Unknown stage"))
    locations = category_order(data["_location_label"].dropna().unique(), LOCATION_ORDER)
    stages = category_order(data["_stage_label"].dropna().unique(), [STAGE_I_II_LABEL, STAGE_III_IV_LABEL])
    for location in locations:
        for stage in stages:
            label = f"{location} | {stage}"
            if label in data[LOCATION_STAGE_COL].astype(str).values:
                labels.append(label)

    labels.extend(
        sorted(
            label
            for label in data[LOCATION_STAGE_COL].astype(str).unique()
            if label not in labels
        )
    )
    return labels


@st.cache_data(show_spinner=False)
def pairwise_tests(df: pd.DataFrame, group_col: str, value_col: str, reference: str | None = None) -> pd.DataFrame:
    groups = [g for g in category_order(df[group_col].dropna().unique(), GROUP_ORDER) if len(df.loc[df[group_col] == g, value_col].dropna()) >= 3]
    if reference and reference in groups:
        pairs = [(reference, g) for g in groups if g != reference]
    else:
        pairs = [(groups[i], groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    rows = []
    for a, b in pairs:
        x = df.loc[df[group_col] == a, value_col].dropna().values
        y = df.loc[df[group_col] == b, value_col].dropna().values
        if len(x) < 3 or len(y) < 3:
            continue
        try:
            _, p = mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append({
            "group_a": a,
            "group_b": b,
            "n_a": len(x),
            "n_b": len(y),
            "median_a": np.median(x),
            "median_b": np.median(y),
            "hedges_g_a_minus_b": hedges_g(x, y),
            "p_value": p,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def colored_boxplot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    hover_cols: Iterable[str],
    yaxis_title: str,
    categories: Iterable[str] | None = None,
) -> go.Figure:
    def transparent_color(color: str, alpha: float = 0.18) -> str:
        if color.startswith("#") and len(color) == 7:
            r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
            return f"rgba({r},{g},{b},{alpha})"
        if color.startswith("rgb("):
            return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")
        return color

    categories = list(categories or category_order(df[group_col].dropna().unique()))
    colors = px.colors.qualitative.Safe + px.colors.qualitative.Plotly
    fig = go.Figure()

    for idx, category in enumerate(categories):
        group_df = df[df[group_col].astype(str) == category].copy()
        if group_df.empty:
            continue
        color = colors[idx % len(colors)]
        customdata = group_df[list(hover_cols)].astype(str).values
        hovertemplate = (
            f"{group_col}={category}<br>"
            f"{yaxis_title}=%{{y}}<br>"
            + "<br>".join(f"{col}=%{{customdata[{i}]}}" for i, col in enumerate(hover_cols))
            + "<extra></extra>"
        )
        fig.add_trace(
            go.Box(
                x=[category] * len(group_df),
                y=group_df[value_col],
                name=category,
                marker=dict(color=color, opacity=0.72, size=6),
                line=dict(color=color),
                fillcolor=transparent_color(color),
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=yaxis_title,
        height=520,
        boxmode="overlay",
        showlegend=True,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=categories)
    return fig


@st.cache_data(show_spinner=False)
def correlate_target_with_matrix(target: pd.Series, matrix: pd.DataFrame, min_n: int, min_prevalence: float = 0.0) -> pd.DataFrame:
    target = target.astype(float)
    matrix = matrix.astype(float)

    if len(target) >= min_n and target.notna().all() and not matrix.isna().any().any():
        target_rank = target.rank(method="average")
        matrix_rank = matrix.rank(method="average")

        valid_cols = matrix.nunique(dropna=True) >= 2
        prevalence = (matrix > 0).mean(axis=0)
        valid_cols &= prevalence >= min_prevalence
        if target.nunique(dropna=True) < 2:
            valid_cols[:] = False

        matrix_rank = matrix_rank.loc[:, valid_cols]
        prevalence = prevalence.loc[valid_cols]
        if matrix_rank.empty:
            return pd.DataFrame()

        x = target_rank.to_numpy(dtype=float)
        y = matrix_rank.to_numpy(dtype=float)
        x_centered = x - x.mean()
        y_centered = y - y.mean(axis=0)
        denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2, axis=0))
        rho = np.divide(
            np.sum(y_centered * x_centered[:, None], axis=0),
            denom,
            out=np.full(y_centered.shape[1], np.nan),
            where=denom > 0,
        )

        n = len(target_rank)
        rho_clipped = np.clip(rho, -0.999999999999, 0.999999999999)
        t_stat = rho_clipped * np.sqrt((n - 2) / (1 - rho_clipped ** 2))
        p_values = 2 * t.sf(np.abs(t_stat), df=n - 2)

        out = pd.DataFrame({
            "feature": matrix_rank.columns,
            "n": n,
            "rho": rho,
            "p_value": p_values,
            "prevalence": prevalence.to_numpy(dtype=float),
        })
        out = out[np.isfinite(out["rho"]) & np.isfinite(out["p_value"])].copy()
        if not out.empty:
            out["q_value"] = multipletests(out["p_value"], method="fdr_bh")[1]
            out["abs_rho"] = out["rho"].abs()
            out = out.sort_values(["q_value", "abs_rho"], ascending=[True, False]).reset_index(drop=True)
        return out

    rows = []
    for col in matrix.columns:
        y = matrix[col]
        d = pd.concat([target, y], axis=1).dropna()
        n = len(d)
        if n < min_n or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
            continue
        if min_prevalence > 0 and (d.iloc[:, 1] > 0).mean() < min_prevalence:
            continue
        rho, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        if np.isfinite(rho) and np.isfinite(p):
            rows.append({"feature": col, "n": n, "rho": float(rho), "p_value": float(p), "prevalence": float((d.iloc[:, 1] > 0).mean())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"], method="fdr_bh")[1]
        out["abs_rho"] = out["rho"].abs()
        out = out.sort_values(["q_value", "abs_rho"], ascending=[True, False]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def crc_species_enrichment(meta: pd.DataFrame, species: pd.DataFrame, min_prevalence: float, min_n: int) -> pd.DataFrame:
    healthy_ids = ids_in_matrix(meta[meta[GROUP_COL] == HEALTHY_LABEL], species)
    crc_ids = ids_in_matrix(meta[meta[GROUP_COL].isin(INVASIVE_CRC_GROUPS)], species)
    rows = []
    for sp in species.columns:
        x = species.loc[crc_ids, sp].dropna().values
        y = species.loc[healthy_ids, sp].dropna().values
        if len(x) < min_n or len(y) < min_n:
            continue
        prev_crc = float((x > 0).mean())
        prev_h = float((y > 0).mean())
        if max(prev_crc, prev_h) < min_prevalence:
            continue
        try:
            _, p = mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append({
            "species": sp,
            "n_crc": len(x),
            "n_healthy": len(y),
            "prev_crc": prev_crc,
            "prev_healthy": prev_h,
            "mean_crc": float(np.mean(x)),
            "mean_healthy": float(np.mean(y)),
            "log2FC_crc_vs_healthy": float(np.log2((np.mean(x) + PSEUDOCOUNT) / (np.mean(y) + PSEUDOCOUNT))),
            "p_value": float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"], method="fdr_bh")[1]
        out = out.sort_values(["q_value", "log2FC_crc_vs_healthy"], ascending=[True, False]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def location_bubble_correlations(
    meta: pd.DataFrame,
    metabolites: pd.DataFrame,
    species: pd.DataFrame,
    metabolite: str,
    min_n: int,
) -> pd.DataFrame:
    subset_defs = {
        "Cancer": single_location_meta(meta[meta[GROUP_COL].isin(INVASIVE_CRC_GROUPS)]),
        RIGHT_SIDE_LABEL: meta[(meta[GROUP_COL].isin(INVASIVE_CRC_GROUPS)) & (meta[LOCATION_COL] == RIGHT_SIDE_LABEL)],
        LEFT_SIDE_LABEL: meta[(meta[GROUP_COL].isin(INVASIVE_CRC_GROUPS)) & (meta[LOCATION_COL] == LEFT_SIDE_LABEL)],
        RECTUM_LABEL: meta[(meta[GROUP_COL].isin(INVASIVE_CRC_GROUPS)) & (meta[LOCATION_COL] == RECTUM_LABEL)],
    }
    rows = []
    for subset_name, sub_meta in subset_defs.items():
        ids = [sid for sid in ids_in_matrix(sub_meta, metabolites) if sid in species.index]
        if len(ids) < min_n:
            continue
        res = correlate_target_with_matrix(metabolites.loc[ids, metabolite], species.loc[ids], min_n=min_n, min_prevalence=0.05)
        if not res.empty:
            res = res.rename(columns={"feature": "species"})
            res["subset"] = subset_name
            rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metric_header(meta: pd.DataFrame, metabolites: pd.DataFrame, species: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples with metabolomics", f"{metabolites.shape[0]:,}")
    c2.metric("Metabolites", f"{metabolites.shape[1]:,}")
    c3.metric("Samples with species", f"{species.shape[0]:,}")
    c4.metric("Species", f"{species.shape[1]:,}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload compatible Excel file", type=["xlsx"])
        if uploaded is None:
            meta, metabolites, species = load_data_from_path(str(DEFAULT_DATA))
        else:
            meta, metabolites, species = load_data(uploaded)

        st.header("Global filters")
        group_options = category_order(meta[GROUP_COL].dropna().unique(), GROUP_ORDER)
        groups = st.multiselect("Groups", group_options, default=group_options)
        loc_options = available_locations(meta)
        locations = st.multiselect("Tumor location", loc_options, default=[])
        stage_options = category_order(
            [str(x) for x in meta[STAGE_COL].dropna().unique().tolist() if str(x) != "-"],
            GROUP_ORDER,
        )
        
        stages = st.multiselect("Detailed stage", stage_options, default=[])
        genders = st.multiselect("Gender", sorted(meta["Gender"].dropna().unique()), default=[])
        min_n = st.slider("Minimum paired samples for correlations", 3, 50, 10)

    meta_sub = filter_meta(meta, groups=groups, locations=locations, stages=stages, genders=genders)
    metric_header(meta_sub, metabolites, species)
    st.caption(f"Current metadata filter keeps {len(meta_sub):,} samples.")

    tabs = st.tabs([
        "Metabolite Explorer",
        "Bacteria-Metabolite Correlations",
        "Location Bubble Plot",
        "Bacteria Co-occurrence",
        "CRC Enrichment",
    ])

    with tabs[0]:
        st.subheader("Metabolite abundance and metabolite-metabolite correlations")
        explorer_meta = with_healthy_baseline(meta, meta_sub, groups, genders)
        ids = ids_in_matrix(explorer_meta, metabolites)
        col1, col2, col3 = st.columns([2, 1, 1])
        metabolite = col1.selectbox("Metabolite", metabolites.columns, index=0)
        group_by = col2.selectbox("Group abundance by", [GROUP_COL, LOCATION_COL, STAGE_COL, LOCATION_STAGE_COL, "Gender"])
        use_log = col3.checkbox("Use log10(value + 1)", value=True)

        plot_df = explorer_meta[[ID_COL, GROUP_COL, LOCATION_COL, STAGE_COL, "Gender"]].copy()
        plot_df = plot_df[plot_df[ID_COL].isin(ids)]
        if group_by == LOCATION_STAGE_COL:
            plot_df = invasive_location_stage_meta(plot_df)
        elif group_by == LOCATION_COL:
            healthy_df = plot_df[plot_df[GROUP_COL] == HEALTHY_LABEL]
            plot_df = pd.concat([healthy_df, single_location_meta(plot_df)], ignore_index=True).drop_duplicates(subset=[ID_COL])
        plot_df = add_location_stage_group(plot_df)
        if group_by in {LOCATION_COL, STAGE_COL}:
            plot_df.loc[plot_df[GROUP_COL] == HEALTHY_LABEL, group_by] = HEALTHY_LABEL
        values = metabolites.loc[plot_df[ID_COL].astype(str), metabolite].values
        plot_df["abundance"] = values
        plot_df["plot_value"] = np.log10(plot_df["abundance"] + 1) if use_log else plot_df["abundance"]
        if group_by == GROUP_COL:
            plot_order = category_order(plot_df[group_by].unique(), GROUP_ORDER)
        elif group_by == STAGE_COL:
            plot_order = category_order(plot_df[group_by].unique(), GROUP_ORDER)
        elif group_by == LOCATION_COL:
            plot_order = category_order(plot_df[group_by].unique(), [HEALTHY_LABEL, *LOCATION_ORDER])
        elif group_by == LOCATION_STAGE_COL:
            plot_order = location_stage_order(plot_df)
        else:
            plot_order = category_order(plot_df[group_by].unique())

        fig = colored_boxplot(
            plot_df,
            group_col=group_by,
            value_col="plot_value",
            hover_cols=[ID_COL, GROUP_COL, LOCATION_COL, STAGE_COL],
            yaxis_title=("log10 abundance + 1" if use_log else "abundance"),
            categories=plot_order,
        )
        st.plotly_chart(fig, use_container_width=True)

        s1, s2 = st.columns(2)
        s1.write("Group summary")
        summary = group_summary(plot_df, group_by, "abundance")
        summary[group_by] = pd.Categorical(summary[group_by].astype(str), categories=plot_order, ordered=True)
        summary = summary.sort_values(group_by).reset_index(drop=True)
        s1.dataframe(summary, use_container_width=True)
        s2.write("Pairwise Mann-Whitney tests")
        ref = HEALTHY_LABEL if group_by == GROUP_COL and HEALTHY_LABEL in plot_df[group_by].unique() else None
        tests = pairwise_tests(plot_df, group_by, "abundance", reference=ref)
        s2.dataframe(tests, use_container_width=True)

        st.divider()
        st.markdown("### Correlations with other metabolites")
        st.caption("Spearman correlations between the selected metabolite and every other metabolite in the currently filtered samples.")
        corr_ids = ids_in_matrix(meta_sub, metabolites)
        corr_matrix = metabolites.loc[corr_ids].drop(columns=[metabolite], errors="ignore")
        target = metabolites.loc[corr_ids, metabolite]
        met_corr = correlate_target_with_matrix(target, corr_matrix, min_n=min_n)
        if met_corr.empty:
            st.warning("No metabolite correlations passed the current filters.")
        else:
            left, right = st.columns([1, 1])
            left.dataframe(met_corr.rename(columns={"feature": "metabolite"}), use_container_width=True, height=420)
            hit = right.selectbox("Scatter plot correlation hit", met_corr["feature"].tolist(), index=0)
            rho, p, n = safe_spearman(target, metabolites.loc[corr_ids, hit])
            scatter_df = pd.DataFrame({metabolite: target, hit: metabolites.loc[corr_ids, hit]}).dropna()
            fig2 = px.scatter(scatter_df, x=metabolite, y=hit, trendline="ols", hover_name=scatter_df.index)
            fig2.update_layout(title=f"Spearman rho={rho:.3f}, p={p:.2e}, n={n}", height=500)
            right.plotly_chart(fig2, use_container_width=True)
            st.download_button("Download metabolite correlation table", met_corr.to_csv(index=False).encode(), "metabolite_correlations.csv")

    with tabs[1]:
        st.subheader("Bacteria-metabolite correlations")
        ids = [sid for sid in ids_in_matrix(meta_sub, metabolites) if sid in species.index]
        corr_direction = st.radio(
            "Correlation target",
            ["Metabolite vs bacteria", "Bacteria vs metabolites"],
            horizontal=True,
        )
        min_prev = st.slider("Minimum species prevalence", 0.0, 1.0, 0.05, 0.01)
        if corr_direction == "Metabolite vs bacteria":
            metabolite_bac = st.selectbox("Metabolite", metabolites.columns, index=min(1, len(metabolites.columns)-1), key="bac_met")
            target = metabolites.loc[ids, metabolite_bac]
            matrix = species.loc[ids]
            feature_label = "species"
            result_filename = "bacteria_metabolite_correlations.csv"
            no_results_message = "No species correlations passed the current filters."
            bac_corr = correlate_target_with_matrix(target, matrix, min_n=min_n, min_prevalence=min_prev)
        else:
            species_bac = st.selectbox("Bacteria", species.columns, index=0, key="target_species")
            target = species.loc[ids, species_bac]
            matrix = metabolites.loc[ids]
            feature_label = "metabolite"
            result_filename = "bacteria_target_metabolite_correlations.csv"
            no_results_message = "No metabolite correlations passed the current filters."
            if (target > 0).mean() < min_prev:
                bac_corr = pd.DataFrame()
                no_results_message = "The selected bacteria does not pass the current prevalence threshold."
            else:
                bac_corr = correlate_target_with_matrix(target, matrix, min_n=min_n)
        if bac_corr.empty:
            st.warning(no_results_message)
        else:
            volcano = bac_corr.copy()
            volcano["minus_log10_q"] = -np.log10(volcano["q_value"].clip(lower=1e-300))
            volcano["direction"] = np.where(volcano["rho"] >= 0, "positive", "negative")
            fig = px.scatter(volcano, x="rho", y="minus_log10_q", color="direction", hover_name="feature", hover_data=["n", "p_value", "q_value", "prevalence"])
            fig.update_layout(xaxis_title="Spearman rho", yaxis_title="-log10(FDR)", height=620)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bac_corr.rename(columns={"feature": feature_label}), use_container_width=True, height=420)
            st.download_button("Download bacteria-metabolite correlations", bac_corr.to_csv(index=False).encode(), result_filename)

    with tabs[2]:
        st.subheader("Cancer location bubble plot")
        st.caption("Rows are top correlated species. Columns are all invasive CRC, right side, left side, and rectum. Multi-site tumor locations are ignored.")
        bubble_met = st.selectbox("Metabolite", metabolites.columns, key="bubble_met")
        top_n = st.slider("Top species", 5, 50, 25)
        bubble_df = location_bubble_correlations(meta, metabolites, species, bubble_met, min_n)
        if bubble_df.empty:
            st.warning("Not enough data to make the bubble plot.")
        else:
            top_species = (bubble_df.groupby("species")["rho"].max().sort_values(ascending=False).head(top_n).index.tolist())
            d = bubble_df[(bubble_df["species"].isin(top_species)) & (bubble_df["rho"] > 0)].copy()
            d["minus_log10_q"] = -np.log10(d["q_value"].clip(lower=1e-300))
            fig = px.scatter(d, x="subset", y="species", size="minus_log10_q", color="rho", hover_data=["p_value", "q_value", "n", "prevalence"], size_max=26)
            fig.update_layout(height=max(520, 22 * len(top_species)), xaxis_title="Tumor location", yaxis_title="Species")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bubble_df, use_container_width=True, height=380)

    with tabs[3]:
        st.subheader("Bacteria co-occurrence heatmap")
        ids = ids_in_matrix(meta_sub, species)
        defaults = [s for s in species.columns if any(t.lower() in s.lower() for t in INTERESTING_SPECIES)][:12]
        selected_species = st.multiselect("Species", species.columns, default=defaults)
        if len(selected_species) < 2:
            st.info("Select at least two species.")
        else:
            mat = species.loc[ids, selected_species].dropna(axis=1, how="all")
            corr = mat.corr(method="spearman")
            fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=max(520, 25 * len(selected_species)))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr, use_container_width=True)

    with tabs[4]:
        st.subheader("CRC enrichment explorer")
        col1, col2 = st.columns([2, 1])
        species_choice = col1.selectbox("Species", species.columns, key="species_enrich")
        enrich_group_by = col2.selectbox("Group abundance by", [GROUP_COL, LOCATION_STAGE_COL], key="crc_enrich_group_by")
        min_prev_enrich = st.slider("Minimum prevalence for enrichment table", 0.0, 1.0, 0.10, 0.01)
        enrich = crc_species_enrichment(meta, species, min_prev_enrich, min_n=3)
        sp_ids = ids_in_matrix(meta, species)
        sp_df = meta[meta[ID_COL].isin(sp_ids)][[ID_COL, GROUP_COL, LOCATION_COL, STAGE_COL]].copy()
        sp_df = sp_df[sp_df[GROUP_COL].isin(CRC_ENRICHMENT_ORDER)]
        if enrich_group_by == LOCATION_STAGE_COL:
            sp_df = invasive_location_stage_meta(sp_df)
        sp_df = add_location_stage_group(sp_df)
        sp_df["abundance"] = species.loc[sp_df[ID_COL].astype(str), species_choice].values
        sp_df["plot_value"] = np.log10(sp_df["abundance"] + 1)
        if enrich_group_by == GROUP_COL:
            enrich_order = category_order(sp_df[enrich_group_by].unique(), CRC_ENRICHMENT_ORDER)
        else:
            enrich_order = location_stage_order(sp_df)
        fig = colored_boxplot(
            sp_df,
            group_col=enrich_group_by,
            value_col="plot_value",
            hover_cols=[ID_COL, GROUP_COL, LOCATION_COL, STAGE_COL],
            yaxis_title="log10 species abundance + 1",
            categories=enrich_order,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(enrich, use_container_width=True, height=420)
        st.download_button("Download CRC enrichment table", enrich.to_csv(index=False).encode(), "crc_species_enrichment.csv")


if __name__ == "__main__":
    main()

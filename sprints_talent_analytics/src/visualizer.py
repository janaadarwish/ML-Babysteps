"""
visualizer.py
=============
Matplotlib, Seaborn, Plotly and PyGWalker views for the Sprints Talent
Pipeline project. All charts use the Okabe-Ito colourblind-safe palette and
avoid chart junk (no gridlines, no top/right spines).

    import sys; sys.path.append("../src")
    import data_processor as dp, visualizer as viz

    res = dp.run_pipeline()
    viz.static_overview(res["clean"], res["schema"])
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"

# Okabe-Ito: designed to be distinguishable under all common colour-vision deficiencies
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
DIVERGING = sns.diverging_palette(250, 30, l=60, as_cmap=True)  # blue <-> orange, CVD-safe

# --------------------------------------------------------------------------
# Global styling
# --------------------------------------------------------------------------
def apply_style() -> None:
    """Global Matplotlib/Seaborn style sheet: clean, no grid, CVD-safe cycle."""
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    sns.set_palette(PALETTE)


def _save(fig, name: str | None) -> None:
    if name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name)


def _plotly_layout(fig: go.Figure, title: str, height: int = 750) -> go.Figure:
    fig.update_layout(
        title=title, height=height, template="simple_white",
        colorway=PALETTE, hovermode="closest", legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------
# 1. Matplotlib: static publication figure (object-oriented API)
# --------------------------------------------------------------------------
def static_overview(df: pd.DataFrame, schema, save_as: str | None = "overview_static.png"):
    """Histogram of Skill Mastery Index + boxplot of SMI by track."""
    apply_style()
    track = schema.track or schema.cohort
    order = df.groupby(track)["skill_mastery_index"].median().sort_values(ascending=False).index

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})

    ax_hist.hist(df["skill_mastery_index"], bins=30, color=PALETTE[0], edgecolor="white")
    med = df["skill_mastery_index"].median()
    ax_hist.axvline(med, color=PALETTE[3], linewidth=2, linestyle="--")
    ax_hist.annotate(f"Median {med:.1f}", xy=(med, ax_hist.get_ylim()[1] * 0.92),
                     xytext=(6, 0), textcoords="offset points", color=PALETTE[3])
    ax_hist.set(title="Distribution of Skill Mastery Index",
                xlabel="Skill Mastery Index", ylabel="Learners")

    data = [df.loc[df[track] == t, "skill_mastery_index"] for t in order]
    bp = ax_box.boxplot(data, vert=True, patch_artist=True, labels=list(order),
                        medianprops={"color": "black", "linewidth": 1.5})
    for patch, colour in zip(bp["boxes"], PALETTE * 3):
        patch.set(facecolor=colour, alpha=0.8, edgecolor="black")
    ax_box.set(title=f"Skill Mastery Index by {track.replace('_', ' ')}",
               xlabel="", ylabel="Skill Mastery Index")
    ax_box.tick_params(axis="x", rotation=20)

    fig.suptitle("Sprints graduate skill profile", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# --------------------------------------------------------------------------
# 2. Seaborn: annotated correlation heatmap (early modules vs capstone)
# --------------------------------------------------------------------------
def correlation_heatmap(df: pd.DataFrame, schema, early_modules: int = 3,
                        save_as: str | None = "correlation_heatmap.png"):
    apply_style()
    cols = schema.modules[:early_modules] + ([schema.capstone] if schema.capstone else [])
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Pearson r", "shrink": 0.8}, ax=ax)
    ax.set_title("Early modules vs capstone score")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# --------------------------------------------------------------------------
# 3. Plotly: interactive multi-panel dashboard
# --------------------------------------------------------------------------
def plotly_dashboard(df: pd.DataFrame, long: pd.DataFrame, schema, save_html: bool = True) -> go.Figure:
    """
    2x2 dashboard: SMI histogram | SMI by track | median SMI by cohort |
    average score by program month (with range slider).
    """
    track = schema.track or schema.cohort
    cohort = schema.cohort or track
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Skill Mastery Index distribution", f"SMI by {track}",
                        f"Median SMI by {cohort}", "Average module score over the program"),
        vertical_spacing=0.17, horizontal_spacing=0.10,
    )

    fig.add_trace(go.Histogram(x=df["skill_mastery_index"], nbinsx=30, marker_color=PALETTE[0],
                               hovertemplate="SMI %{x}<br>Learners %{y}<extra></extra>",
                               showlegend=False), row=1, col=1)

    for i, (t, g) in enumerate(df.groupby(track)):
        fig.add_trace(go.Box(y=g["skill_mastery_index"], name=str(t), marker_color=PALETTE[i % len(PALETTE)],
                             hovertemplate="%{y:.1f}<extra>" + str(t) + "</extra>", showlegend=False),
                      row=1, col=2)

    by_cohort = df.groupby(cohort)["skill_mastery_index"].median().reset_index()
    fig.add_trace(go.Bar(x=by_cohort[cohort].astype(str), y=by_cohort["skill_mastery_index"],
                         marker_color=PALETTE[2], showlegend=False,
                         hovertemplate="%{x}<br>Median SMI %{y:.1f}<extra></extra>"), row=2, col=1)

    monthly = long.groupby("program_month")["score"].mean().reset_index()
    fig.add_trace(go.Scatter(x=monthly["program_month"], y=monthly["score"], mode="lines+markers",
                             line=dict(color=PALETTE[3], width=3), showlegend=False,
                             hovertemplate="Month %{x}<br>Avg score %{y:.1f}<extra></extra>"), row=2, col=2)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=2, col=2)

    _plotly_layout(fig, "Sprints talent pipeline dashboard", height=800)
    if save_html:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(FIG_DIR / "dashboard.html", include_plotlyjs="cdn")
    return fig


# --------------------------------------------------------------------------
# 4. Plotly Express: animated scatter over the 12-month timeline
# --------------------------------------------------------------------------
def animated_scatter(long: pd.DataFrame, schema, save_html: bool = True) -> go.Figure:
    group = schema.cohort or schema.track
    agg = (long.groupby([group, "program_month"])
               .agg(avg_score=("score", "mean"),
                    avg_engagement=(schema.engagement, "mean"),
                    learners=("score", "size"))
               .reset_index())
    fig = px.scatter(
        agg, x="avg_engagement", y="avg_score", size="learners", color=group,
        animation_frame="program_month", animation_group=group, size_max=45,
        color_discrete_sequence=PALETTE,
        range_x=[agg["avg_engagement"].min() * 0.95, agg["avg_engagement"].max() * 1.05],
        range_y=[agg["avg_score"].min() * 0.95, agg["avg_score"].max() * 1.05],
        labels={"avg_engagement": "Average engagement", "avg_score": "Average assessment score",
                "program_month": "Program month"},
        hover_data={"learners": True},
    )
    _plotly_layout(fig, "How cohorts evolve across the 12-month program", height=600)
    if save_html:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(FIG_DIR / "animated_scatter.html", include_plotlyjs="cdn")
    return fig


# --------------------------------------------------------------------------
# 5. PyGWalker: drag-and-drop explorer for non-technical partners
# --------------------------------------------------------------------------
def launch_pygwalker(df: pd.DataFrame):
    """Call from its own notebook cell; returns the PyGWalker widget."""
    try:
        import pygwalker as pyg
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install with: pip install pygwalker") from exc
    return pyg.walk(df, spec=str(ROOT / "assets" / "pygwalker_spec.json"))


if __name__ == "__main__":
    import data_processor as dp

    res = dp.run_pipeline()
    static_overview(res["clean"], res["schema"])
    correlation_heatmap(res["clean"], res["schema"])
    plotly_dashboard(res["clean"], res["long"], res["schema"])
    animated_scatter(res["long"], res["schema"])
    print(f"Figures saved to {FIG_DIR}")
"""
Uvidi o mobilnosti: RoG, dužina puta, emisije — profil po vozilu.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mobility_lab.io_sumo import apply_vehicle_name_cleanup, short_vehicle_id, simplify_vehicle_type
from mobility_lab.vehicle_features import aggregate_emissions_by_vehicle, attach_lengths_and_rog


def build_vehicle_profile(
    merged: pd.DataFrame,
    lengths: pd.Series | None,
    rog: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Jedan red po vozilu: tip, dužina, RoG, emisije, čekanje, izvedene metrike.

    Uključuje tertile RoG-a i "wide_dirty_score" (visok RoG × visok CO₂).
    """
    if merged is None or merged.empty:
        return pd.DataFrame()

    merged = apply_vehicle_name_cleanup(merged)
    prof = aggregate_emissions_by_vehicle(merged)
    prof = attach_lengths_and_rog(prof, lengths=lengths, rog=rog)

    if "rog_km" in prof.columns:
        valid = prof["rog_km"].notna()
        if valid.sum() >= 3:
            prof.loc[valid, "rog_tertile"] = pd.qcut(
                prof.loc[valid, "rog_km"],
                q=3,
                labels=["localno (nizak RoG)", "mešovito", "široko (visok RoG)"],
                duplicates="drop",
            )

    if "rog_km" in prof.columns and "co2_sum" in prof.columns:
        r = prof["rog_km"].astype(float)
        c = prof["co2_sum"].astype(float)
        r_n = (r - r.min()) / (r.max() - r.min() + 1e-9)
        c_n = (c - c.min()) / (c.max() - c.min() + 1e-9)
        prof["wide_dirty_score"] = r_n * c_n

    if "vehicle_id" in prof.columns:
        prof["vehicle_label"] = prof["vehicle_id"].map(short_vehicle_id)
    if "vehicle_type" in prof.columns:
        prof["vehicle_type"] = prof["vehicle_type"].map(simplify_vehicle_type)

    return (
        prof.sort_values("co2_sum", ascending=False)
        if "co2_sum" in prof.columns
        else prof
    )


def matplotlib_rog_vs_length(prof: pd.DataFrame) -> plt.Figure | None:
    """
    Scatter: radius of gyration (km) vs dužina puta (km), boja = tip vozila.
    """
    if prof.empty or "rog_km" not in prof.columns or "length_m" not in prof.columns:
        return None
    d = prof.dropna(subset=["rog_km", "length_m"])
    if d.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    types = d["vehicle_type"].astype(str)
    cats = types.value_counts().head(8).index.tolist()
    d = d.copy()
    d["plot_type"] = types.where(types.isin(cats), "ostalo")
    for t, g in d.groupby("plot_type"):
        ax.scatter(
            g["rog_km"],
            g["length_m"] / 1000.0,
            label=t,
            s=28,
            alpha=0.65,
        )
    ax.set_xlabel("Radius of gyration (km)")
    ax.set_ylabel("Dužina puta (km)")
    ax.set_title("RoG vs dužina trajektorije (boja = tip vozila)")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig


_TERTILE_ORDER = ["localno (nizak RoG)", "mešovito", "široko (visok RoG)"]
_TERTILE_SHORT = {
    "localno (nizak RoG)": "Lokalno\n(nizak RoG)",
    "mešovito": "Mešovito",
    "široko (visok RoG)": "Široko\n(visok RoG)",
}
_METRIC_LABELS = {
    "co2_sum": (
        "Prosečan ukupni CO₂ po vozilu",
        "Σ CO₂ (simulacija)",
    ),
    "waiting_sum": (
        "Prosečno ukupno čekanje po vozilu",
        "Σ čekanje (s)",
    ),
    "co2_per_km": (
        "Prosečan CO₂ po pređenom kilometru",
        "CO₂ / km",
    ),
}


def matplotlib_metrics_by_rog_tertile(prof: pd.DataFrame) -> plt.Figure | None:
    """
    Tri bar charta: prosečne emisije/čekanje po RoG tertilima.

    Vozila podeljena u tri grupe po radius of gyration (lokalno / mešovito / široko kretanje).
    """
    if prof.empty or "rog_tertile" not in prof.columns:
        return None
    g = prof.dropna(subset=["rog_tertile"])
    if g.empty:
        return None

    metrics = [c for c in ["co2_sum", "waiting_sum", "co2_per_km"] if c in g.columns]
    if not metrics:
        return None

    nrows = len(metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.8 * nrows + 1.2))
    if nrows == 1:
        axes = [axes]

    colors = ["#4daf4a", "#ff7f00", "#e41a1c"]

    for ax, col in zip(axes, metrics):
        title, ylabel = _METRIC_LABELS.get(col, (col, col))
        means = g.groupby("rog_tertile", observed=True)[col].mean()
        counts = g.groupby("rog_tertile", observed=True)[col].count()
        order_idx = [t for t in _TERTILE_ORDER if t in means.index]
        means = means.reindex(order_idx)
        counts = counts.reindex(order_idx)
        x = np.arange(len(means))
        bars = ax.bar(
            x,
            means.values,
            color=colors[: len(means)],
            edgecolor="#333333",
            linewidth=0.6,
            alpha=0.9,
            width=0.55,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_TERTILE_SHORT.get(t, str(t)) for t in means.index], fontsize=10
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ymax = max(means.max() * 1.15, 1e-9)
        ax.set_ylim(0, ymax)
        for bar, val, n in zip(bars, means.values, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{val:,.0f}\n(n={int(n)})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        "Kako se vozila ponašaju po prostornom opsegu (RoG tertili)\n"
        "Lokalno = mali poluprečnik kretanja · Široko = pokriva veći deo mreže u simulaciji",
        fontsize=12,
        y=1.01,
    )
    fig.subplots_adjust(top=0.88, bottom=0.06, hspace=0.55)
    return fig


def top_wide_and_dirty(prof: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top "n" vozila po "wide_dirty_score" (visok RoG i visok CO₂ u uzorku).

    Korisno za identifikaciju vozila koja imaju i širok prostorni opseg (RoG) i visoke emisije CO₂.
    """
    if prof.empty or "wide_dirty_score" not in prof.columns:
        return pd.DataFrame()
    cols = [
        c
        for c in [
            "vehicle_id",
            "vehicle_type",
            "rog_km",
            "length_m",
            "co2_sum",
            "co2_per_km",
            "waiting_sum",
            "wide_dirty_score",
        ]
        if c in prof.columns
    ]
    return prof.nlargest(n, "wide_dirty_score")[cols]


def insights_narrative(prof: pd.DataFrame) -> str:
    """
    Kratki tekstualni rezime (markdown) za mobilnost tab.
    """
    if prof.empty:
        return "Nema profila vozila — pokreni analizu (merge + skmob RoG)."

    lines = []
    n = len(prof)
    lines.append(f"Uzorak: **{n}** vozila sa spojenim FCD + emisijama.")

    if "rog_km" in prof.columns and prof["rog_km"].notna().any():
        r = prof["rog_km"].dropna()
        lines.append(
            f"**RoG** (koliko se vozilo prostire po gradu): medijana {r.median():.2f} km, "
            f"opseg {r.min():.2f}–{r.max():.2f} km."
        )

    if "vehicle_type" in prof.columns and "rog_km" in prof.columns:
        type_rog = (
            prof.groupby("vehicle_type")["rog_km"].median().sort_values(ascending=False)
        )
        if len(type_rog):
            top = type_rog.index[0]
            lines.append(
                f"Tip sa najvećim medijanskim RoG: **{top}** ({type_rog.iloc[0]:.2f} km) — "
                f"često autobusi / duže rute u SUMO scenariju."
            )

    if "wide_dirty_score" in prof.columns:
        t = top_wide_and_dirty(prof, 3)
        if not t.empty:
            ids = ", ".join(t["vehicle_id"].astype(str).tolist())
            lines.append(f"Vozila sa visokim RoG **i** visokim CO₂: {ids}.")

    return "\n\n".join(lines)

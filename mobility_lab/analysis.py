"""
Analiza mobilnosti: MovingPandas (dužine trajektorija), scikit-mobility (RoG, jump lengths),
agregacije i matplotlib grafici za Streamlit (CO₂, brzina, histogrami).
"""

from __future__ import annotations

import io

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from movingpandas import TrajectoryCollection
from skmob import TrajDataFrame
from skmob.measures.individual import jump_lengths, radius_of_gyration

from mobility_lab.io_sumo import apply_vehicle_name_cleanup, normalize_fcd_like

# Referentne tačke za Niš (sredina grada) — koristi se za aproksimaciju lat/lon iz planarnih x/y koordinata u FCD bez geografskih koordinata
LAT0, LON0 = 43.3209, 21.8958


def attach_lat_lon_planar(
    df: pd.DataFrame, lat0: float = LAT0, lon0: float = LON0
) -> pd.DataFrame:
    """
    Mapiraj lokalne planarne koordinate na WGS84 koordinate za skmob.
    """
    out = df.copy()
    scale = 111_000.0
    cos_lat = np.cos(np.radians(lat0))
    out["latitude"] = lat0 + (out["vehicle_y"] - out["vehicle_y"].median()) / scale
    out["longitude"] = lon0 + (out["vehicle_x"] - out["vehicle_x"].median()) / (
        scale * cos_lat
    )
    return out


def trajectory_lengths_movingpandas(fcd: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Dužine trajektorija u metrima. Koristi UTM 34N za geografske koordinate, u suprotnom tretira x/y kao planarne u 32634.
    """
    f, is_geo = normalize_fcd_like(fcd)
    if is_geo:
        gdf = gpd.GeoDataFrame(
            f,
            geometry=gpd.points_from_xy(
                f["vehicle_x"], f["vehicle_y"], crs="EPSG:4326"
            ),
        )
        gdf = gdf.to_crs("EPSG:32634")
    else:
        gdf = gpd.GeoDataFrame(
            f,
            geometry=gpd.points_from_xy(f["vehicle_x"], f["vehicle_y"]),
            crs="EPSG:32634",
        )
    tc = TrajectoryCollection(gdf, traj_id_col="vehicle_id", t="t")
    lengths = pd.Series({tr.id: tr.get_length() for tr in tc})
    note = f"MovingPandas: {len(tc)} trajektorija, CRS {'4326→32634' if is_geo else '32634 (planar x/y)'}"
    return lengths, note


def skmob_jump_and_rog(
    fcd: pd.DataFrame,
    max_vehicles: int,
    lat0: float = LAT0,
    lon0: float = LON0,
) -> tuple[pd.Series, pd.DataFrame | None, str]:
    """
    scikit-mobility: jump lengths i radius of gyration po vozilu.

    Ako ima više od "max_vehicles" jedinstvenih ID-jeva, nasumično uzima podskup (brzina).
    Vraća ``(jump_lengths_km, rog_df, napomena_za_UI)``.
    """
    f, is_geo = normalize_fcd_like(fcd)
    uids = f["vehicle_id"].drop_duplicates()
    if len(uids) > max_vehicles:
        uids = uids.sample(n=max_vehicles, random_state=42)
    sub = f[f["vehicle_id"].isin(uids)].copy()

    if is_geo:
        sub["latitude"] = sub["vehicle_y"]
        sub["longitude"] = sub["vehicle_x"]
    else:
        sub = attach_lat_lon_planar(sub, lat0=lat0, lon0=lon0)

    tdf = TrajDataFrame(
        sub,
        latitude="latitude",
        longitude="longitude",
        datetime="t",
        user_id="vehicle_id",
    )
    jl = jump_lengths(tdf, show_progress=False, merge=True)
    ser = pd.Series(jl, name="jump_length_km")

    rog_df: pd.DataFrame | None = None
    rog = radius_of_gyration(tdf, show_progress=False)
    if isinstance(rog, pd.DataFrame):
        rog_df = rog.rename(
            columns={"uid": "vehicle_id", "radius_of_gyration": "radius_of_gyration_km"}
        )

    mode = "WGS84 (geo FCD)" if is_geo else "planar → WGS84 (približno)"
    note = f"scikit-mobility ({mode}), vozila: {len(uids)}"
    return ser, rog_df, note


def aggregates_by_vehicle_type(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Agregira spojeni FCD+emission skup po "vehicle_type" (prosečna brzina, sume CO₂/NOx).

    Tipovi vozila su prethodno skraćeni na klase (passenger, bus, …).
    """
    merged = apply_vehicle_name_cleanup(merged)
    agg_map: dict = {"mean_speed": ("vehicle_speed", "mean")}
    if "vehicle_CO2" in merged.columns:
        agg_map["co2_sum"] = ("vehicle_CO2", "sum")
    if "vehicle_NOx" in merged.columns:
        agg_map["nox_sum"] = ("vehicle_NOx", "sum")
    by_type = merged.groupby("vehicle_type", dropna=False).agg(**agg_map)
    sort_col = "co2_sum" if "co2_sum" in by_type.columns else "mean_speed"
    return by_type.sort_values(sort_col, ascending=False)


def matplotlib_co2_timeseries(merged: pd.DataFrame) -> plt.Figure | None:
    """
    Grafik: zbir CO₂ po koraku simulacije (svi tipovi vozila zajedno).
    """
    if "vehicle_CO2" not in merged.columns:
        return None
    ts = merged.groupby("timestep_time")["vehicle_CO2"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax, title="Σ CO₂ po koraku simulacije (svi tipovi)")
    ax.set_xlabel("timestep_time (s)")
    ax.set_ylabel("Σ CO₂")
    fig.tight_layout()
    return fig


def matplotlib_co2_timeseries_by_type(merged: pd.DataFrame) -> plt.Figure | None:
    """
    Grafik: zbir CO₂ po vremenu — jedna linija po tipu vozila (passenger, bus, …).
    """
    if "vehicle_CO2" not in merged.columns or "vehicle_type" not in merged.columns:
        return None
    merged = apply_vehicle_name_cleanup(merged)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for vtype, grp in merged.groupby("vehicle_type", dropna=False):
        ts = grp.groupby("timestep_time")["vehicle_CO2"].sum()
        ts.plot(ax=ax, label=str(vtype), linewidth=1.2, alpha=0.9)
    ax.set_title("Σ CO₂ po koraku — po tipu vozila")
    ax.set_xlabel("timestep_time (s)")
    ax.set_ylabel("Σ CO₂")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig


def matplotlib_co2_total_by_type(merged: pd.DataFrame) -> plt.Figure | None:
    """
    Grafik: ukupni CO₂ u učitanom uzorku — stubci po tipu vozila.
    """
    if "vehicle_CO2" not in merged.columns or "vehicle_type" not in merged.columns:
        return None
    merged = apply_vehicle_name_cleanup(merged)
    by_type = (
        merged.groupby("vehicle_type", dropna=False)["vehicle_CO2"]
        .sum()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    by_type.plot(kind="bar", ax=ax, color="#2c7fb8", alpha=0.85)
    ax.set_title("Ukupni Σ CO₂ u učitanom uzorku — po tipu vozila")
    ax.set_ylabel("Σ CO₂")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def matplotlib_speed_vs_co2(merged: pd.DataFrame) -> plt.Figure | None:
    """
    Grafik: scatter brzina vs CO₂; boja tačke = tip vozila (do 6 tipova + ostalo).
    """
    if "vehicle_CO2" not in merged.columns or "vehicle_speed" not in merged.columns:
        return None
    merged = apply_vehicle_name_cleanup(merged)
    fig, ax = plt.subplots(figsize=(7, 5))
    if "vehicle_type" in merged.columns:
        types = merged["vehicle_type"].astype(str)
        top = types.value_counts().head(6).index.tolist()
        plot_df = merged.copy()
        plot_df["_t"] = types.where(types.isin(top), "ostalo")
        for t, g in plot_df.groupby("_t"):
            ax.scatter(
                g["vehicle_speed"],
                g["vehicle_CO2"],
                s=5,
                alpha=0.12,
                label=t,
            )
        ax.legend(loc="best", fontsize=7)
        ax.set_title("Brzina vs CO₂ (boja = tip vozila)")
    else:
        merged.plot.scatter(
            x="vehicle_speed",
            y="vehicle_CO2",
            s=4,
            alpha=0.15,
            ax=ax,
            title="Brzina vs CO₂",
        )
    ax.set_xlabel("vehicle_speed")
    ax.set_ylabel("vehicle_CO2")
    fig.tight_layout()
    return fig


def matplotlib_jump_hist(jump_km: pd.Series) -> plt.Figure:
    """
    Histogram jump lengths (km) iz scikit-mobility.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(jump_km.dropna(), bins=50, color="#2c7fb8", alpha=0.85)
    ax.set_title("Jump lengths (km) — scikit-mobility")
    ax.set_xlabel("Δr (km)")
    fig.tight_layout()
    return fig


def matplotlib_traj_length_hist(lengths: pd.Series) -> plt.Figure:
    """
    Histogram dužina trajektorija (m) iz MovingPandas.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    lengths.plot.hist(bins=40, ax=ax, title="Dužina trajektorije (m) — MovingPandas")
    ax.set_xlabel("length (m)")
    fig.tight_layout()
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serijalizuje DataFrame u CSV kao bytes za download dugme u Streamlit-ua.
    """
    buf = io.StringIO()
    df.to_csv(buf, sep=";", encoding="utf-8")
    return buf.getvalue().encode("utf-8")

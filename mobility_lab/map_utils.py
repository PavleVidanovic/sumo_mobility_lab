"""
Zajedničke pomoćne funkcije za mape: boje, H3, RoG težine toplote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPEED_CMAP_HEX = ["#cc0000", "#ff6600", "#ffcc00", "#66cc00", "#009900"]
HOTSPOT_CMAP_HEX = ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c"]


def fmt_legend_value(v: float) -> str:
    """Format broja za legendu (k, M) na mapi."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 10_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


def linear_colormap(colors: list[str], vmin: float, vmax: float):
    """Branca LinearColormap sa normalizovanim granicama."""
    import branca.colormap as cm

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)


def speed_to_hex_color(speed_kmh: float, vmin: float, vmax: float) -> str:
    """Mapira brzinu (km/h) na HEX boju: crveno = sporo, zeleno = brže."""
    if not np.isfinite(speed_kmh):
        return "#888888"
    cmap = linear_colormap(SPEED_CMAP_HEX, vmin, vmax)
    return cmap(float(np.clip(speed_kmh, vmin, vmax)))


def h3_latlng_to_cell(lat: float, lon: float, resolution: int) -> str:
    """H3 ćelija za tačku (podržava h3 v3 i v4 API)."""
    import h3

    try:
        return h3.latlng_to_cell(lat, lon, resolution)
    except AttributeError:
        return h3.geo_to_h3(lat, lon, resolution)


def h3_cell_center(cell: str) -> tuple[float, float]:
    """Centar H3 ćelije kao (lat, lon)."""
    import h3

    try:
        return h3.cell_to_latlng(cell)
    except AttributeError:
        return h3.h3_to_geo(cell)


def h3_cell_polygon(cell: str) -> list[tuple[float, float]]:
    """H3 ćelija kao lista (lat, lon) temena — za Folium Polygon."""
    import h3

    try:
        ring = h3.cell_to_boundary(cell)
    except AttributeError:
        ring = h3.h3_to_geo_boundary(cell, geo_json=False)
    return [(float(lat), float(lon)) for lat, lon in ring]


def attach_rog_weights(
    hdf: pd.DataFrame,
    rog_df: pd.DataFrame | None,
    *,
    rog_col: str = "radius_of_gyration_km",
) -> pd.DataFrame:
    """
    Dodaje kolonu ``_rog_n`` (normalizovan RoG po vozilu) za težinu heatmap tačaka.
    """
    out = hdf.copy()
    if (
        rog_df is not None
        and not rog_df.empty
        and rog_col in rog_df.columns
        and "vehicle_id" in out.columns
    ):
        r = rog_df[["vehicle_id", rog_col]].copy()
        r["vehicle_id"] = r["vehicle_id"].astype(str)
        out["vehicle_id"] = out["vehicle_id"].astype(str)
        out = out.merge(r, on="vehicle_id", how="left")
        has_rog = out[rog_col].notna()
        med = float(out.loc[has_rog, rog_col].median()) if has_rog.any() else 1.0
        med = med if med > 1e-6 else 1.0
        out["_rog_n"] = 1.0
        out.loc[has_rog, "_rog_n"] = (out.loc[has_rog, rog_col] / med).clip(0.25, 4.0)
    else:
        out["_rog_n"] = 1.0
    return out


def heatmap_weights(
    values: pd.Series,
    rog_n: pd.Series,
    *,
    rog_scale: float = 0.45,
    min_weight: float = 0.05,
) -> np.ndarray:
    """log1p(vrednost) × RoG faktor — težine za Folium HeatMap."""
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    rn = pd.to_numeric(rog_n, errors="coerce").fillna(1.0)
    w = np.log1p(v.values) * (1.0 + float(rog_scale) * (rn.values - 1.0))
    return np.clip(w, min_weight, None)


def heatmap_points(
    hdf: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    weights: np.ndarray,
) -> list[list[float]]:
    """Lista ``[lat, lon, weight]`` za Folium HeatMap."""
    lats = hdf[lat_col].astype(float).values
    lons = hdf[lon_col].astype(float).values
    pts: list[list[float]] = []
    for lat, lon, ww in zip(lats, lons, weights):
        if np.isfinite(lat) and np.isfinite(lon):
            pts.append([float(lat), float(lon), float(ww)])
    return pts

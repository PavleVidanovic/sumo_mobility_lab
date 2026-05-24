"""
Zajednička agregacija osobina po vozilu (emisije, dužina, RoG, ML izvedene kolone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mode_or_first(s: pd.Series) -> str:
    """Najčešća vrednost u seriji; ako nema moda, prva vrednost."""
    if s.empty:
        return "unknown"
    m = s.mode()
    return str(m.iloc[0]) if len(m) else str(s.iloc[0])


def aggregate_emissions_by_vehicle(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Agregira spojeni FCD+emission skup po ``vehicle_id`` (CO₂, čekanje, trajanje, broj tačaka).
    """
    agg: dict[str, tuple[str, str | callable]] = {
        "n_points": ("vehicle_id", "count"),
    }
    if "vehicle_type" in merged.columns:
        agg["vehicle_type"] = ("vehicle_type", mode_or_first)
    if "vehicle_speed" in merged.columns:
        agg["mean_speed"] = ("vehicle_speed", "mean")
    if "vehicle_CO2" in merged.columns:
        agg["co2_sum"] = ("vehicle_CO2", "sum")
        agg["co2_mean"] = ("vehicle_CO2", "mean")
    if "vehicle_waiting" in merged.columns:
        agg["waiting_sum"] = ("vehicle_waiting", "sum")
        agg["waiting_max"] = ("vehicle_waiting", "max")
        agg["waiting_mean"] = ("vehicle_waiting", "mean")
    if "timestep_time" in merged.columns:
        agg["duration_s"] = ("timestep_time", lambda s: float(s.max() - s.min()))

    out = (
        merged.assign(vehicle_id=merged["vehicle_id"].astype(str))
        .groupby("vehicle_id", dropna=False)
        .agg(**agg)
        .reset_index()
    )
    return out


def rog_series(rog: pd.DataFrame | None) -> pd.Series | None:
    """Vraća ``vehicle_id`` → RoG (km) mapiranje iz scikit-mobility tabele."""
    if rog is None or rog.empty:
        return None
    rcol = (
        "radius_of_gyration_km"
        if "radius_of_gyration_km" in rog.columns
        else next((c for c in rog.columns if "gyration" in c.lower()), None)
    )
    if not rcol:
        return None
    rid = "vehicle_id" if "vehicle_id" in rog.columns else rog.columns[0]
    return rog.set_index(rog[rid].astype(str))[rcol]


def attach_lengths_and_rog(
    df: pd.DataFrame,
    *,
    lengths: pd.Series | None = None,
    rog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dopunjava tabelu po vozilu dužinom trajektorije i RoG."""
    out = df.copy()
    if "vehicle_id" in out.columns:
        out["vehicle_id"] = out["vehicle_id"].astype(str)

    if lengths is not None and len(lengths):
        lm = lengths.copy()
        lm.index = lm.index.astype(str)
        mapped = out["vehicle_id"].map(lm)
        if "length_m" in out.columns:
            out["length_m"] = mapped.combine_first(out["length_m"])
        else:
            out["length_m"] = mapped

    rmap = rog_series(rog)
    if rmap is not None:
        out["rog_km"] = out["vehicle_id"].map(rmap)

    return add_co2_per_km(out)


def add_co2_per_km(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje ``co2_per_km`` ako postoje ``co2_sum`` i ``length_m``."""
    out = df.copy()
    if "co2_sum" in out.columns and "length_m" in out.columns:
        km = pd.to_numeric(out["length_m"], errors="coerce") / 1000.0
        out["co2_per_km"] = np.where(km > 1e-6, out["co2_sum"] / km, np.nan)
    return out


def merge_emission_into_features(
    features: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    lengths: pd.Series | None = None,
    rog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spaja MobiML agregat sa emisionim osobinama, dužinom i RoG."""
    out = features.copy()
    out["vehicle_id"] = out["vehicle_id"].astype(str)
    if not merged.empty:
        em = aggregate_emissions_by_vehicle(merged)
        out = out.merge(em, on="vehicle_id", how="left", suffixes=("", "_em"))
    return attach_lengths_and_rog(out, lengths=lengths, rog=rog)


def enrich_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje izvedene kolone za ML (displacement_km, co2_per_km, speed_max_over_median, …)."""
    out = df.copy()
    if all(c in out.columns for c in ("x_start", "y_start", "x_end", "y_end")):
        geo = out["x_start"].abs().max() < 180 and out["y_start"].abs().max() < 90
        scale_km = 111.32 if geo else 1.0 / 1000.0
        out["displacement_km"] = (
            np.hypot(out["x_end"] - out["x_start"], out["y_end"] - out["y_start"])
            * scale_km
        )
    if "speed_max" in out.columns and "speed_median" in out.columns:
        out["speed_max_over_median"] = out["speed_max"] / (
            out["speed_median"].abs() + 1e-3
        )
    if "co2_per_km" not in out.columns:
        out = add_co2_per_km(out)
    return out

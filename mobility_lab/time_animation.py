"""
Animacija kretanja vozila i gužvi kroz vreme (FCD + SUMO mreža).

Priprema se jednom (agregacija po koracima vremena), prikaz koristi Folium.
TimestampedGeoJson — play / vremenski klizač.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mobility_lab.folium_animation import (
    VEHICLE_ICON_CLASSES,
    SyncedTimeAnimation,
    vehicle_icon_data_uri,
)
from mobility_lab.folium_map import ensure_lat_lon_columns
from mobility_lab.io_sumo import simplify_vehicle_type
from mobility_lab.map_utils import speed_to_hex_color
from mobility_lab.sumo_network_speed import load_edge_shapes_latlon


def _mean_edge_speeds(snap: pd.DataFrame) -> pd.DataFrame:
    """Prosečna brzina po ivici u jednom kadru animacije."""
    edge_snap = snap.dropna(subset=["edge_id"])
    if edge_snap.empty:
        return edge_snap
    return (
        edge_snap.groupby("edge_id", as_index=False)["vehicle_speed"]
        .mean()
        .assign(speed_kmh=lambda x: x["vehicle_speed"] * 3.6)
    )


def lane_to_edge_id(lane: str | float | None) -> str | None:
    """
    SUMO lane id → edge id (poslednji '_<index>' je indeks trake).
    """
    if lane is None or (isinstance(lane, float) and np.isnan(lane)):
        return None
    lane = str(lane).strip()
    if not lane:
        return None
    if "_" in lane:
        base, idx = lane.rsplit("_", 1)
        if idx.isdigit():
            return base
    return lane


def _sim_time_iso(t_sec: float, origin: str = "2020-01-01") -> str:
    """
    Sekunde simulacije → ISO datetime string za Leaflet TimeDimension.
    """
    ts = pd.Timestamp(origin) + pd.Timedelta(seconds=float(t_sec))
    return ts.isoformat()


def _select_timesteps(
    times: np.ndarray, *, step_sec: float, max_frames: int
) -> np.ndarray:
    """
    Briše vremenske korake za animaciju: mreža po "step_sec", zatim ograničenje na "max_frames".

    Ako ima previše koraka, ravnomerno poduzorkuje ceo interval simulacije.
    """
    times = np.sort(np.unique(times.astype(float)))
    if len(times) == 0:
        return times
    step = max(float(step_sec), 1.0)
    grid = np.arange(times[0], times[-1] + step * 0.01, step)
    picked: list[float] = []
    for g in grid:
        idx = int(np.argmin(np.abs(times - g)))
        t = float(times[idx])
        if not picked or t > picked[-1] + 1e-6:
            picked.append(t)
    times = np.array(picked, dtype=float)
    if len(times) > max_frames:
        idx = np.linspace(0, len(times) - 1, max_frames, dtype=int)
        times = times[idx]
    return times


def _snap_for_frame(df: pd.DataFrame, t: float, step_sec: float) -> pd.DataFrame:
    """
    Redovi FCD za kadar t; tolerancija + najbliži korak ako isclose ne nađe ništa.
    """
    atol = max(float(step_sec) * 0.51, 0.5)
    snap = df[np.isclose(df["timestep_time"], t, rtol=0, atol=atol)]
    if not snap.empty:
        return snap
    delta = (df["timestep_time"] - t).abs()
    if delta.empty or float(delta.min()) > atol:
        return snap
    t_near = float(df.loc[delta.idxmin(), "timestep_time"])
    return df[np.isclose(df["timestep_time"], t_near, rtol=0, atol=1e-6)]


def prepare_time_animation(
    fcd: pd.DataFrame,
    *,
    is_geo: bool,
    lat0: float,
    lon0: float,
    net_path: str | Path | None,
    step_sec: float = 10.0,
    max_frames: int = 120,
    max_vehicles_per_frame: int = 450,
    show_vehicles: bool = True,
    show_street_speeds: bool = True,
    jam_speed_kmh: float = 15.0,
) -> dict[str, Any]:
    """
    Priprema GeoJSON za vremensku animaciju (vozila + opciono boje ulica po brzini).

    Vraća dictionary: "vehicle_geojson", "edge_geojson", "period", "meta" (broj kadrova, centar, itd.).
    Poziva se jednom u Streamlit-u pre prikaza mape.
    """
    need = {"vehicle_id", "timestep_time", "vehicle_x", "vehicle_y"}
    if not need.issubset(fcd.columns):
        raise ValueError(f"FCD mora imati kolone {need}.")

    df = fcd.copy()
    if "vehicle_speed" not in df.columns:
        df["vehicle_speed"] = np.nan
    if "vehicle_lane" not in df.columns:
        df["vehicle_lane"] = np.nan

    df = ensure_lat_lon_columns(df, is_geo, lat0, lon0)
    df["timestep_time"] = pd.to_numeric(df["timestep_time"], errors="coerce")
    df["vehicle_speed"] = pd.to_numeric(df["vehicle_speed"], errors="coerce")
    df = df.dropna(subset=["timestep_time", "latitude", "longitude"])
    df["edge_id"] = df["vehicle_lane"].map(lane_to_edge_id)

    times = _select_timesteps(
        df["timestep_time"].values, step_sec=step_sec, max_frames=max_frames
    )
    if len(times) == 0:
        raise ValueError("Nema koraka vremena u FCD podacima.")

    period_td = timedelta(seconds=max(float(step_sec), 1.0))
    period_str = f"PT{int(period_td.total_seconds())}S"
    frame_visible_sec = period_td.total_seconds()

    speed_kmh_all = df["vehicle_speed"].dropna() * 3.6
    if len(speed_kmh_all):
        vmin_g = float(np.percentile(speed_kmh_all, 5))
        vmax_g = float(np.percentile(speed_kmh_all, 95))
    else:
        vmin_g, vmax_g = 0.0, 30.0
    if vmax_g <= vmin_g:
        vmax_g = vmin_g + 1.0

    vehicle_features: list[dict] = []
    edge_features: list[dict] = []
    edges_seen: set[str] = set()

    for t in times:
        snap = _snap_for_frame(df, float(t), step_sec)
        if snap.empty:
            continue
        t_end = float(t) + frame_visible_sec
        t0_iso = _sim_time_iso(t)
        t1_iso = _sim_time_iso(t_end)

        if show_vehicles:
            snap_v = snap.groupby("vehicle_id", as_index=False).first()
            if len(snap_v) > max_vehicles_per_frame:
                snap_v = snap_v.sample(
                    n=max_vehicles_per_frame, random_state=int(t) % 10_000
                )
            for row in snap_v.itertuples(index=False):
                if not (np.isfinite(row.longitude) and np.isfinite(row.latitude)):
                    continue
                spd_kmh = (
                    float(row.vehicle_speed) * 3.6
                    if np.isfinite(row.vehicle_speed)
                    else np.nan
                )
                color = (
                    speed_to_hex_color(spd_kmh, vmin_g, vmax_g)
                    if np.isfinite(spd_kmh)
                    else "#3366cc"
                )
                vcls = simplify_vehicle_type(row.vehicle_id)
                if vcls not in VEHICLE_ICON_CLASSES:
                    vcls = "passenger"
                vehicle_features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(row.longitude), float(row.latitude)],
                        },
                        "properties": {
                            "times": [t0_iso, t1_iso],
                            "icon": "marker",
                            "iconstyle": {
                                "iconUrl": vehicle_icon_data_uri(color, vcls),
                                "iconSize": [18, 18],
                                "iconAnchor": [9, 9],
                                "popupAnchor": [0, -10],
                            },
                            "popup": (
                                f"vozilo {row.vehicle_id}<br>{spd_kmh:.0f} km/h"
                                if np.isfinite(spd_kmh)
                                else f"vozilo {row.vehicle_id}"
                            ),
                        },
                    }
                )

        if show_street_speeds and net_path and Path(net_path).is_file():
            edge_speed = _mean_edge_speeds(snap)
            for erow in edge_speed.itertuples(index=False):
                edges_seen.add(str(erow.edge_id))

    shapes: dict[str, list[tuple[float, float]]] = {}
    if show_street_speeds and net_path and edges_seen:
        shapes = load_edge_shapes_latlon(net_path, edge_ids=edges_seen)

    if show_street_speeds and shapes:
        for t in times:
            snap = _snap_for_frame(df, float(t), step_sec)
            if snap.empty:
                continue
            t_end = float(t) + frame_visible_sec
            t0_iso = _sim_time_iso(t)
            t1_iso = _sim_time_iso(t_end)
            edge_speed = _mean_edge_speeds(snap)
            if edge_speed.empty:
                continue
            for erow in edge_speed.itertuples(index=False):
                pts = shapes.get(str(erow.edge_id))
                if not pts or len(pts) < 2:
                    continue
                spd = float(erow.speed_kmh)
                color = speed_to_hex_color(spd, vmin_g, vmax_g)
                coords = [[float(lon), float(lat)] for lat, lon in pts]
                jam = "da" if spd < jam_speed_kmh else "ne"
                edge_features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {
                            "times": [t0_iso, t1_iso],
                            "style": {
                                "color": color,
                                "weight": 4,
                                "opacity": 0.88,
                            },
                            "popup": (
                                f"<b>{erow.edge_id}</b><br>"
                                f"Prosečna brzina: {spd:.1f} km/h<br>"
                                f"Gužva (&lt;{jam_speed_kmh:.0f} km/h): {jam}"
                            ),
                        },
                    }
                )

    center_lat = float(df["latitude"].median())
    center_lon = float(df["longitude"].median())

    vehicle_gj = {"type": "FeatureCollection", "features": vehicle_features}
    edge_gj = (
        {"type": "FeatureCollection", "features": edge_features}
        if edge_features
        else None
    )

    return {
        "vehicle_geojson": vehicle_gj,
        "edge_geojson": edge_gj,
        "period": period_str,
        "meta": {
            "n_frames": len(times),
            "time_min": float(times[0]),
            "time_max": float(times[-1]),
            "step_sec": float(step_sec),
            "n_vehicle_features": len(vehicle_features),
            "n_edge_features": len(edge_features),
            "center": [center_lat, center_lon],
            "speed_vmin_kmh": vmin_g,
            "speed_vmax_kmh": vmax_g,
            "jam_speed_kmh": jam_speed_kmh,
            "geo_version": 6,
        },
    }


_EMPTY_FC = {"type": "FeatureCollection", "features": []}


def folium_time_animation_map(
    prepared: dict[str, Any],
    *,
    zoom_start: int = 14,
    auto_play: bool = False,
    show_street_speeds: bool = True,
) -> Any:
    """
    Folium: sinhronizovana animacija ulica i vozila.
    """
    import folium
    from folium import plugins

    meta = prepared["meta"]
    center = meta["center"]
    period = prepared["period"]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    edge_gj = (
        (prepared.get("edge_geojson") or _EMPTY_FC) if show_street_speeds else _EMPTY_FC
    )
    veh_gj = prepared.get("vehicle_geojson") or _EMPTY_FC
    has_any = edge_gj.get("features") or veh_gj.get("features")
    if has_any:
        veh_duration = period if veh_gj.get("features") else ""
        step_s = float(meta.get("step_sec") or 10.0)
        transition_ms = int(np.clip(step_s * 120, 150, 900))
        SyncedTimeAnimation(
            edge_gj,
            veh_gj,
            period=period,
            vehicle_duration=veh_duration,
            auto_play=auto_play,
            transition_ms=transition_ms,
        ).add_to(m)

    plugins.Fullscreen(position="topright").add_to(m)
    return m


def prepared_json_size_kb(prepared: dict[str, Any]) -> float:
    """
    Procena veličine pripremljenog GeoJSON-a (KB).
    """
    n = len(json.dumps(prepared.get("vehicle_geojson", {})))
    if prepared.get("edge_geojson"):
        n += len(json.dumps(prepared["edge_geojson"]))
    return n / 1024.0

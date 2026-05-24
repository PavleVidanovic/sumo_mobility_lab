"""
Folium mape: trag vozila, toplote emisija/zastoja, sloj prosečne brzine ulica (edgeData).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mobility_lab.analysis import attach_lat_lon_planar
from mobility_lab.map_utils import (
    SPEED_CMAP_HEX,
    attach_rog_weights,
    fmt_legend_value,
    heatmap_points,
    heatmap_weights,
)


def trajectory_folium_map(
    fcd: pd.DataFrame,
    vehicle_id: str,
    *,
    timestep_col: str = "timestep_time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    zoom_start: int = 15,
):
    """
    Folium mapa: trag jednog vozila (polyline + start/end marker).

    Očekuje kolone "latitude/longitude" (geo FCD ili posle "ensure_lat_lon_columns").
    """
    import folium
    from folium import plugins

    sub = fcd[fcd["vehicle_id"] == vehicle_id].sort_values(timestep_col)
    if sub.empty:
        raise ValueError(f"Nema tačaka za vozilo {vehicle_id!r}")

    if lat_col not in sub.columns or lon_col not in sub.columns:
        raise ValueError(
            f"Nedostaju {lat_col!r} / {lon_col!r} — učitaj geo FCD ili prvo pokreni analizu."
        )

    coords = list(zip(sub[lat_col].astype(float), sub[lon_col].astype(float)))
    center = coords[len(coords) // 2]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.PolyLine(
        coords, color="#1a5276", weight=4, opacity=0.85, popup=vehicle_id
    ).add_to(m)
    folium.CircleMarker(
        location=coords[0], radius=6, color="green", fill=True, popup="start"
    ).add_to(m)
    folium.CircleMarker(
        location=coords[-1], radius=6, color="red", fill=True, popup="end"
    ).add_to(m)
    plugins.Fullscreen(position="topright").add_to(m)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


def ensure_lat_lon_columns(
    fcd: pd.DataFrame, is_geo: bool, lat0: float, lon0: float
) -> pd.DataFrame:
    """
    Dodaje "latitude" i "longitude" u FCD DataFrame.

    Geo FCD: x=lon, y=lat. Planarni: aproksimacija preko "attach_lat_lon_planar".
    """
    out = fcd.copy()
    if is_geo:
        out["longitude"] = out["vehicle_x"].astype(float)
        out["latitude"] = out["vehicle_y"].astype(float)
    else:
        out = attach_lat_lon_planar(out, lat0=lat0, lon0=lon0)
    return out


def heatmap_rog_weight_stats(
    merged: pd.DataFrame,
    rog_df: pd.DataFrame | None,
    *,
    is_geo: bool,
    lat0: float,
    lon0: float,
    max_heat_points: int = 28_000,
    rog_scale: float = 0.45,
) -> dict[str, float | int | bool]:
    """
    Dijagnostika: da li RoG menja težine toplote emisija (min/max množilac, broj tačaka).

    Koristi se za proveru uticaja radius of gyration na intenzitet heatmape.
    """
    out: dict[str, float | int | bool] = {
        "rog_table_rows": 0,
        "rog_applied": False,
        "n_sample": 0,
        "pct_rog_not_one": 0.0,
        "rog_n_min": 1.0,
        "rog_n_max": 1.0,
        "weight_ratio_max_over_min": 1.0,
    }
    if merged is None or merged.empty:
        return out
    hdf = merged_to_lat_lon(merged.copy(), is_geo, lat0, lon0)
    if "latitude" not in hdf.columns or "vehicle_CO2" not in hdf.columns:
        return out
    hdf = hdf.dropna(subset=["latitude", "longitude"]).copy()
    if len(hdf) > max_heat_points:
        hdf = hdf.sample(n=max_heat_points, random_state=42)
    out["n_sample"] = len(hdf)

    hdf = attach_rog_weights(hdf, rog_df)
    if rog_df is not None and not rog_df.empty:
        out["rog_table_rows"] = len(rog_df)
        rog_col = "radius_of_gyration_km"
        out["rog_applied"] = (
            rog_col in hdf.columns and bool(hdf[rog_col].notna().any())
        )

    rn = hdf["_rog_n"].astype(float)
    out["rog_n_min"] = float(rn.min())
    out["rog_n_max"] = float(rn.max())
    out["pct_rog_not_one"] = float((np.abs(rn - 1.0) > 0.05).mean() * 100.0)

    co2 = pd.to_numeric(hdf["vehicle_CO2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    w0 = np.log1p(co2.values)
    w1 = heatmap_weights(co2, rn, rog_scale=rog_scale, min_weight=1e-9)
    out["weight_ratio_max_over_min"] = float(np.nanmax(w1 / w0))
    return out


def _add_speed_network_legend(
    m,
    *,
    vmin_kmh: float,
    vmax_kmh: float,
    n_edges: int,
    bottom_px: int = 28,
    side: str = "right",
) -> None:
    """
    Legenda za sloj prosečne brzine na ulicama.
    """
    from branca.element import MacroElement
    from jinja2 import Template

    if not np.isfinite(vmin_kmh):
        vmin_kmh = 0.0
    if not np.isfinite(vmax_kmh) or vmax_kmh <= vmin_kmh:
        vmax_kmh = vmin_kmh + 1.0

    side_css = "right: 12px;" if side == "right" else "left: 12px;"
    gradient_css = (
        "linear-gradient(to right, "
        + ", ".join(SPEED_CMAP_HEX)
        + ")"
    )
    template = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: {bottom_px}px; {side_css} z-index: 9999;
         background: white; padding: 8px 10px; border: 1px solid #888; border-radius: 4px;
         font-family: Arial, sans-serif; font-size: 12px; line-height: 1.35; max-width: 220px;">
      <div style="font-weight: bold; margin-bottom: 4px;">Prosečna brzina (ulice)</div>
      <div style="height: 14px; width: 190px; margin: 4px 0;
           background: {gradient_css}; border: 1px solid #ccc;"></div>
      <div style="width: 190px; display: flex; justify-content: space-between; font-size: 11px;">
        <span>{vmin_kmh:.1f}</span>
        <span>{vmax_kmh:.1f}</span>
      </div>
      <div style="font-size: 11px; color: #333; margin-top: 4px;">km/h (iz edgeData.xml)</div>
      <div style="font-size: 11px; margin-top: 2px;">
        <span style="color:#cc0000;">■</span> sporo / gužva &nbsp;
        <span style="color:#009900;">■</span> brže
      </div>
      <div style="font-size: 10px; color: #555; margin-top: 3px;">{n_edges} segmenta sa saobraćajem</div>
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(template)
    macro.add_to(m)


def _add_heatmap_legend(
    m,
    *,
    title: str,
    vmin: float,
    vmax: float,
    unit: str,
    gradient_css: str,
    bottom_px: int = 28,
    side: str = "right",
) -> None:
    """
    HTML legenda: gradient + numerički raspon (odgovara bojama HeatMap-a).
    """
    from branca.element import MacroElement
    from jinja2 import Template

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    side_css = "right: 12px;" if side == "right" else "left: 12px;"
    template = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: {bottom_px}px; {side_css} z-index: 9999;
         background: white; padding: 8px 10px; border: 1px solid #888; border-radius: 4px;
         font-family: Arial, sans-serif; font-size: 12px; line-height: 1.35; max-width: 200px;">
      <div style="font-weight: bold; margin-bottom: 4px;">{title}</div>
      <div style="height: 14px; width: 170px; margin: 4px 0;
           background: {gradient_css}; border: 1px solid #ccc;"></div>
      <div style="width: 170px; display: flex; justify-content: space-between; font-size: 11px;">
        <span>{fmt_legend_value(vmin)}</span>
        <span>{fmt_legend_value(vmax)}</span>
      </div>
      <div style="font-size: 11px; color: #333; margin-top: 4px;">{unit}</div>
      <div style="font-size: 11px; margin-top: 2px;">
        <span style="color:#0066ff;">■</span> nisko &nbsp;
        <span style="color:#66cc00;">■</span> srednje &nbsp;
        <span style="color:#ff8800;">■</span> više &nbsp;
        <span style="color:#cc0000;">■</span> visoko
      </div>
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(template)
    macro.add_to(m)


def merged_to_lat_lon(
    merged: pd.DataFrame, is_geo: bool, lat0: float, lon0: float
) -> pd.DataFrame:
    """
    Dodaje latitude i longitude u "merged" DataFrame koji se koristi za toplote, ako već ne postoje.
    """
    if not {"vehicle_x", "vehicle_y"}.issubset(merged.columns):
        return merged
    return ensure_lat_lon_columns(merged.copy(), is_geo, lat0, lon0)


def mobility_map_trajectory_and_heatmap(
    f_map: pd.DataFrame,
    vehicle_id: str,
    merged: pd.DataFrame | None,
    rog_df: pd.DataFrame | None,
    *,
    is_geo: bool,
    lat0: float,
    lon0: float,
    timestep_col: str = "timestep_time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    zoom_start: int = 15,
    max_heat_points: int = 28_000,
    show_polyline: bool = True,
    show_emission_heat: bool = True,
    show_jam_heat: bool = False,
    show_speed_network: bool = False,
    edge_data_path: str | None = None,
    net_path: str | None = None,
    rog_scale_emission: float = 0.45,
):
    """
    Glavna Folium mapa u aplikaciji: trag izabranog vozila + opcione toplote i sloj brzine ulica.

    - Toplota emisija: intenzitet iz CO₂.
    - Toplota zastoja: iz "vehicle_waiting".
    - Prosečna brzina na ulici: "edgeData.xml" + SUMO mreža.
    """
    import folium
    from folium import plugins

    sub = f_map[f_map["vehicle_id"].astype(str) == str(vehicle_id)].sort_values(
        timestep_col
    )
    if sub.empty:
        raise ValueError(f"Nema tačaka za vozilo {vehicle_id!r}")

    if lat_col not in sub.columns or lon_col not in sub.columns:
        raise ValueError(
            f"Nedostaju {lat_col!r} / {lon_col!r} — učitaj geo FCD ili prilagodi Lat0/Lon0."
        )

    coords = list(zip(sub[lat_col].astype(float), sub[lon_col].astype(float)))
    center = coords[len(coords) // 2]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    speed_legend_bottom = 28
    if show_speed_network:
        if not edge_data_path or not net_path:
            raise ValueError(
                "Za sloj brzine na ulicama potrebni su edgeData.xml i osm.net.xml(.gz)."
            )
        from mobility_lab.sumo_network_speed import add_speed_network_layer

        stats = add_speed_network_layer(m, edge_data_path, net_path)
        # Legenda ispod toplota ako su uključene
        if show_emission_heat and show_jam_heat:
            speed_legend_bottom = 208
        elif show_emission_heat or show_jam_heat:
            speed_legend_bottom = 118
        _add_speed_network_legend(
            m,
            vmin_kmh=float(stats["vmin_kmh"]),
            vmax_kmh=float(stats["vmax_kmh"]),
            n_edges=int(stats["n_edges"]),
            bottom_px=speed_legend_bottom,
            side="left" if show_emission_heat and not show_jam_heat else "right",
        )

    if (
        merged is not None
        and not merged.empty
        and (show_emission_heat or show_jam_heat)
    ):
        hdf = merged_to_lat_lon(merged.copy(), is_geo, lat0, lon0)
        if lat_col in hdf.columns and lon_col in hdf.columns:
            hdf = hdf.dropna(subset=[lat_col, lon_col]).copy()
            if len(hdf) > max_heat_points:
                hdf = hdf.sample(n=max_heat_points, random_state=42)

            hdf = attach_rog_weights(hdf, rog_df)

            if show_emission_heat and "vehicle_CO2" in hdf.columns:
                co2 = (
                    pd.to_numeric(hdf["vehicle_CO2"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                )
                w = heatmap_weights(
                    co2, hdf["_rog_n"], rog_scale=rog_scale_emission
                )
                pts = heatmap_points(hdf, lat_col, lon_col, w)
                if pts:
                    fg_h = folium.FeatureGroup(
                        name="Toplota emisija", show=True, overlay=True, control=True
                    )
                    plugins.HeatMap(
                        pts,
                        min_opacity=0.15,
                        max_zoom=18,
                        radius=14,
                        blur=16,
                        gradient={0.2: "blue", 0.4: "lime", 0.65: "orange", 1.0: "red"},
                    ).add_to(fg_h)
                    fg_h.add_to(m)
                    co2_vals = co2[co2 > 0]
                    cmin = float(co2_vals.min()) if len(co2_vals) else 0.0
                    cmax = float(co2_vals.max()) if len(co2_vals) else 1.0
                    _add_heatmap_legend(
                        m,
                        title="Toplota emisija (CO₂)",
                        vmin=cmin,
                        vmax=cmax,
                        unit="Vrednosti CO₂ u uzorku tačaka",
                        gradient_css="linear-gradient(to right, #0066ff, #66ff00, #ffaa00, #cc0000)",
                        bottom_px=28,
                        side="right",
                    )

            if show_jam_heat and "vehicle_waiting" in hdf.columns:
                wait = (
                    pd.to_numeric(hdf["vehicle_waiting"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                )
                wj = heatmap_weights(
                    wait, hdf["_rog_n"], rog_scale=rog_scale_emission
                )
                pts_j = heatmap_points(hdf, lat_col, lon_col, wj)
                if pts_j:
                    fg_j = folium.FeatureGroup(
                        name="Toplota zastoja", show=True, overlay=True, control=True
                    )
                    plugins.HeatMap(
                        pts_j,
                        min_opacity=0.12,
                        max_zoom=18,
                        radius=12,
                        blur=14,
                        gradient={0.3: "cyan", 0.55: "magenta", 1.0: "darkred"},
                    ).add_to(fg_j)
                    fg_j.add_to(m)
                    wvals = wait[wait > 0]
                    wmin = float(wvals.min()) if len(wvals) else 0.0
                    wmax = float(wvals.max()) if len(wvals) else 1.0
                    _add_heatmap_legend(
                        m,
                        title="Toplota zastoja (čekanje)",
                        vmin=wmin,
                        vmax=wmax,
                        unit="Čekanje (s) u uzorku tačaka",
                        gradient_css="linear-gradient(to right, #00ffff, #ff00ff, #8b0000)",
                        bottom_px=28 if not show_emission_heat else 118,
                        side="left",
                    )

    if show_polyline:
        fg_t = folium.FeatureGroup(
            name=f"Trag: {vehicle_id}", show=True, overlay=True, control=True
        )
        folium.PolyLine(
            coords, color="#1a5276", weight=5, opacity=0.95, popup=str(vehicle_id)
        ).add_to(fg_t)
        folium.CircleMarker(
            location=coords[0], radius=6, color="green", fill=True, popup="start"
        ).add_to(fg_t)
        folium.CircleMarker(
            location=coords[-1], radius=6, color="red", fill=True, popup="end"
        ).add_to(fg_t)
        fg_t.add_to(m)

    plugins.Fullscreen(position="topright").add_to(m)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(m)
    return m

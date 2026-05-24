"""
SUMO mreža ulica obojena po prosečnoj brzini (edgeData + net.xml).

Koristi edgeData.xml (atribut speed u m/s) i oblike ivica iz osm.net.xml(.gz).
Koordinate mreže: SUMO file (x,y) → UTM (x - netOffset) → WGS84 (pyproj), kao geo FCD.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


@lru_cache(maxsize=4)
def _parse_location(net_path: str) -> dict:
    """
    Čita "<location>" iz SUMO mreže: "netOffset" i projekciju.

    Vraća rečnik sa pomakom mreže i "pyproj" transformatorom u WGS84 (EPSG:4326).
    Keširano — ista mreža se ne parsira više puta.
    """
    net_path = Path(net_path)
    opener = gzip.open if str(net_path).endswith(".gz") else open
    with opener(net_path, "rb") as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag == "location":
                off = [float(v) for v in elem.attrib["netOffset"].split(",")]
                proj = elem.attrib.get(
                    "projParameter",
                    "+proj=utm +zone=34 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
                )
                elem.clear()
                src = CRS.from_proj4(proj)
                to_wgs84 = Transformer.from_crs(
                    src, CRS.from_epsg(4326), always_xy=True
                )
                return {
                    "net_offset_x": off[0],
                    "net_offset_y": off[1],
                    "to_wgs84": to_wgs84,
                }
            elem.clear()
    raise ValueError(f"Nema <location> u mreži: {net_path}")


def net_xy_to_latlon(x: float, y: float, loc: dict) -> tuple[float, float]:
    """
    Pretvara SUMO koordinate ivice (x, y) u (latitude, longitude) za Folium.
    """
    px = x - loc["net_offset_x"]
    py = y - loc["net_offset_y"]
    lon, lat = loc["to_wgs84"].transform(px, py)
    return float(lat), float(lon)


def load_edge_speeds_from_edgedata(
    edge_data_path: str | Path,
    *,
    min_sampled_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Učitaj prosečnu brzinu po ivici iz SUMO edgeData.xml.
    Kolone: edge_id, speed_ms, speed_kmh, sampled_seconds.
    """
    root = ET.parse(edge_data_path).getroot()
    rows: list[dict] = []
    for interval in root.findall("interval"):
        for edge in interval.findall("edge"):
            sp = edge.get("speed")
            if sp is None:
                continue
            sampled = float(edge.get("sampledSeconds", 0) or 0)
            if sampled < min_sampled_seconds:
                continue
            speed_ms = float(sp)
            rows.append(
                {
                    "edge_id": edge.get("id"),
                    "speed_ms": speed_ms,
                    "speed_kmh": speed_ms * 3.6,
                    "sampled_seconds": sampled,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["edge_id", "speed_ms", "speed_kmh", "sampled_seconds"]
        )
    return pd.DataFrame(rows)


def _load_junction_xy(
    net_path: Path, loc: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """
    Učitava sve čvorove mreže: "junction_id" → "(lat, lon)".
    """
    junctions: dict[str, tuple[float, float]] = {}
    opener = gzip.open if str(net_path).endswith(".gz") else open
    with opener(net_path, "rb") as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag == "junction":
                jid = elem.get("id")
                x, y = elem.get("x"), elem.get("y")
                if jid and x is not None and y is not None:
                    junctions[jid] = net_xy_to_latlon(float(x), float(y), loc)
            elem.clear()
    return junctions


def load_edge_shapes_latlon(
    net_path: str | Path,
    edge_ids: set[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """
    Oblici ivica kao liste (lat, lon).

    Ako trake nemaju "shape" (npr. netconvert --geometry.remove), koristi se linija
    između "from" i "to" čvora; inače oblik prve trake.
    """
    net_path = Path(net_path)
    loc = _parse_location(str(net_path.resolve()))
    junctions = _load_junction_xy(net_path, loc)
    shapes: dict[str, list[tuple[float, float]]] = {}
    opener = gzip.open if str(net_path).endswith(".gz") else open
    with opener(net_path, "rb") as fh:
        for _event, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != "edge":
                elem.clear()
                continue
            eid = elem.get("id")
            if edge_ids is not None and eid not in edge_ids:
                elem.clear()
                continue

            pts: list[tuple[float, float]] = []
            lanes = elem.findall("lane")
            shape_str = lanes[0].get("shape") if lanes else None
            if shape_str:
                for pair in shape_str.split():
                    x_str, y_str = pair.split(",")
                    pts.append(net_xy_to_latlon(float(x_str), float(y_str), loc))
            else:
                j_from = elem.get("from")
                j_to = elem.get("to")
                if j_from and j_to and j_from in junctions and j_to in junctions:
                    pts = [junctions[j_from], junctions[j_to]]

            if len(pts) >= 2:
                shapes[eid] = pts
            elem.clear()
    return shapes


def speed_color_limits_kmh(
    speeds_kmh: pd.Series | np.ndarray,
    *,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
) -> tuple[float, float]:
    """
    Određuje "vmin"/"vmax" (km/h) za bojenje ulica pomoću percentila.

    Sprečava da cela mapa bude crvena kada su sve brzine slično niske.
    """
    arr = pd.to_numeric(speeds_kmh, errors="coerce").dropna().values
    if len(arr) == 0:
        return 0.0, 30.0
    vmin = float(np.percentile(arr, low_pct))
    vmax = float(np.percentile(arr, high_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


from mobility_lab.map_utils import speed_to_hex_color

def add_speed_network_layer(
    m,
    edge_data_path: str | Path,
    net_path: str | Path,
    *,
    min_sampled_seconds: float = 0.5,
    layer_name: str = "Prosečna brzina (ulice)",
    weight: int = 4,
    opacity: float = 0.85,
) -> dict[str, float]:
    """
    Dodaje Folium sloj: ulice obojene po prosečnoj brzini iz "edgeData.xml".

    Vraća "vmin_kmh", "vmax_kmh" i broj nacrtanih segmenata (za legendu).
    """
    import folium

    speeds = load_edge_speeds_from_edgedata(
        edge_data_path, min_sampled_seconds=min_sampled_seconds
    )
    if speeds.empty:
        raise ValueError(
            "edgeData.xml nema ivica sa brzinom (proveri putanju ili simulaciju)."
        )

    edge_ids = set(speeds["edge_id"].astype(str))
    shapes = load_edge_shapes_latlon(net_path, edge_ids=edge_ids)
    if not shapes:
        raise ValueError("Nije učitana nijedna geometrija ivica iz SUMO mreže.")

    merged = speeds[speeds["edge_id"].isin(shapes.keys())].copy()
    if merged.empty:
        raise ValueError(
            "Nema preklopa između edgeData i mreže (različiti scenario fajlovi?)."
        )

    vmin, vmax = speed_color_limits_kmh(merged["speed_kmh"])

    fg = folium.FeatureGroup(name=layer_name, show=True, overlay=True, control=True)
    n_drawn = 0
    for row in merged.itertuples(index=False):
        eid = row.edge_id
        pts = shapes.get(eid)
        if not pts:
            continue
        color = speed_to_hex_color(float(row.speed_kmh), vmin, vmax)
        popup = (
            f"<b>{eid}</b><br>"
            f"Prosečna brzina: {row.speed_kmh:.1f} km/h<br>"
            f"({row.speed_ms:.2f} m/s)"
        )
        folium.PolyLine(
            pts,
            color=color,
            weight=weight,
            opacity=opacity,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(fg)
        n_drawn += 1

    fg.add_to(m)
    return {
        "vmin_kmh": vmin,
        "vmax_kmh": vmax,
        "n_edges": n_drawn,
    }

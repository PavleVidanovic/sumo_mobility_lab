"""
Učitavanje i normalizacija SUMO CSV (FCD, emission), spajanje i skraćivanje imena vozila/tipova.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# SUMO vType / vehicle id: passenger_p0_b__passenger → passenger
_VEHICLE_CLASS_RE = re.compile(
    r"^(passenger|truck|bus|bicycle|motorcycle|pedestrian)(?:_\d+|[\w_]*)?(?:__\1)?$",
    re.I,
)


_VEHICLE_CLASSES = (
    "passenger",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "pedestrian",
)


def simplify_vehicle_type(value: str | float | None) -> str:
    """
    Skraćuje SUMO "type" ili "vehicle_id" na osnovnu klasu vozila.

    Primer: "passenger_p0_b__passenger" → "passenger".
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    low = s.lower()
    if low in _VEHICLE_CLASSES:
        return low
    if "__" in s:
        tail = s.split("__")[-1].strip().lower()
        if tail in _VEHICLE_CLASSES:
            return tail
    m = _VEHICLE_CLASS_RE.match(s)
    if m:
        return m.group(1).lower()
    for cls in _VEHICLE_CLASSES:
        if low.startswith(cls + "_") or low == cls:
            return cls
    head = low.split("_")[0]
    return head if head in _VEHICLE_CLASSES else head


def short_vehicle_id(vehicle_id: str | float | None) -> str:
    """
    Skraćuje pun SUMO ID vozila za prikaz u UI.

    Primer: "passenger_p0_b_42" → "passenger_42".
    """
    if vehicle_id is None or (isinstance(vehicle_id, float) and pd.isna(vehicle_id)):
        return "unknown"
    s = str(vehicle_id).strip()
    if not s:
        return "unknown"
    low = s.lower()
    for cls in _VEHICLE_CLASSES:
        if low.startswith(cls + "_"):
            parts = s.split("_")
            for part in reversed(parts):
                if part.isdigit():
                    return f"{cls}_{part}"
            return cls
    return s


def apply_vehicle_name_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuje imena u DataFrame-u: "vehicle_type" (klasa) i "vehicle_label" (kratak ID).

    Koristi se posle učitavanja emission CSV i posle spajanja sa FCD.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "vehicle_type" in out.columns:
        out["vehicle_type"] = out["vehicle_type"].map(simplify_vehicle_type)
    elif "vehicle_id" in out.columns:
        out["vehicle_type"] = out["vehicle_id"].map(simplify_vehicle_type)
    if "vehicle_id" in out.columns:
        out["vehicle_label"] = out["vehicle_id"].map(short_vehicle_id)
    return out


def sniff_separator(path: Path, n_lines: int = 5) -> str:
    """
    Pogađa separator CSV fajla (";" ili ",").

    SUMO "xml2csv" obično koristi ";".
    """
    text = path.read_text(encoding="utf-8", errors="replace")[:8000]
    lines = [ln for ln in text.splitlines() if ln.strip()][:n_lines]
    if not lines:
        return ";"
    head = "\n".join(lines)
    semi = head.count(";")
    comma = head.count(",")
    return ";" if semi >= comma else ","


def read_sumo_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Učitava SUMO CSV (FCD, emission, …) sa automatskim separatorom.

    Parametar "nrows" ograničava broj redova (brže učitavanje u Streamlit-u).
    """
    p = Path(path)
    sep = sniff_separator(p)
    return pd.read_csv(p, sep=sep, nrows=nrows, low_memory=False)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    """
    Vraća prvo ime kolone iz "candidates" koje postoji u "columns" (bez obzira na velika/mala slova).
    """
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_fcd_like(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Normalizuje FCD tabelu na kolone: "vehicle_id", "timestep_time", "vehicle_x", "vehicle_y".

    Vraća "(df, is_geo)" gde je "is_geo=True" ako su x/y geografske koordinate (WGS84, stepeni).
    Dodaje kolonu "t" (datetime) za MovingPandas / scikit-mobility.
    """
    out = df.copy()
    cols = list(out.columns)

    id_col = _first_existing(cols, ["vehicle_id", "id", "vehicle-id"])
    time_col = _first_existing(cols, ["timestep_time", "time", "timestep"])
    x_col = _first_existing(cols, ["vehicle_x", "x", "lon", "longitude", "pos_x"])
    y_col = _first_existing(cols, ["vehicle_y", "y", "lat", "latitude", "pos_y"])

    if not id_col or not time_col or not x_col or not y_col:
        missing = [
            n
            for n, v in [
                ("vehicle_id", id_col),
                ("time", time_col),
                ("x", x_col),
                ("y", y_col),
            ]
            if not v
        ]
        raise ValueError(f"Ne mogu mapirati kolone {missing}. Dostupne: {cols}")

    rename = {
        id_col: "vehicle_id",
        time_col: "timestep_time",
        x_col: "vehicle_x",
        y_col: "vehicle_y",
    }
    out = out.rename(columns=rename)
    out = out[
        out["vehicle_id"].notna() & (out["vehicle_id"].astype(str).str.len() > 0)
    ].copy()
    out["timestep_time"] = pd.to_numeric(out["timestep_time"], errors="coerce")
    out["vehicle_x"] = pd.to_numeric(out["vehicle_x"], errors="coerce")
    out["vehicle_y"] = pd.to_numeric(out["vehicle_y"], errors="coerce")
    out = out.dropna(subset=["timestep_time", "vehicle_x", "vehicle_y"])

    mx = out["vehicle_x"].abs().median()
    my = out["vehicle_y"].abs().median()
    # Heuristic: SUMO --fcd-output.geo uses x=lon, y=lat (degrees).
    is_geo = (
        mx < 180
        and my < 90
        and out["vehicle_x"].abs().max() < 180
        and out["vehicle_y"].abs().max() < 90
    )

    # SUMO timestep_time is simulation seconds; arbitrary calendar origin for TrajDataFrame.
    out["t"] = pd.to_datetime(out["timestep_time"], unit="s", origin="2020-01-01")
    return out, is_geo


def normalize_emission_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuje emission CSV: ID vozila, vreme, CO₂, brzina, tip vozila.

    Mapira SUMO nazive kolona (npr. "CO2" → "vehicle_CO2") i skraćuje "vehicle_type".
    """
    out = df.copy()
    cols = list(out.columns)

    id_col = _first_existing(cols, ["vehicle_id", "id"])
    time_col = _first_existing(cols, ["timestep_time", "time", "timestep"])

    if not id_col or not time_col:
        raise ValueError(f"Emission CSV: nedostaje id ili vreme. Kolone: {cols}")

    rename = {id_col: "vehicle_id", time_col: "timestep_time"}
    out = out.rename(columns=rename)
    out = out[
        out["vehicle_id"].notna() & (out["vehicle_id"].astype(str).str.len() > 0)
    ].copy()
    out["timestep_time"] = pd.to_numeric(out["timestep_time"], errors="coerce")
    out = out.dropna(subset=["timestep_time"])

    # SUMO emission attributes are often uppercase in XML
    for old, new in [
        ("CO2", "vehicle_CO2"),
        ("NOx", "vehicle_NOx"),
        ("PMx", "vehicle_PMx"),
        ("speed", "vehicle_speed"),
        ("type", "vehicle_type"),
    ]:
        c = _first_existing(list(out.columns), [old, old.lower(), new])
        if c and c != new and new not in out.columns:
            out = out.rename(columns={c: new})

    if "vehicle_speed" not in out.columns:
        sc = _first_existing(list(out.columns), ["speed", "Speed"])
        if sc:
            out = out.rename(columns={sc: "vehicle_speed"})
    if "vehicle_type" not in out.columns:
        tc = _first_existing(list(out.columns), ["type", "vType", "vehicletype"])
        if tc:
            out = out.rename(columns={tc: "vehicle_type"})

    return apply_vehicle_name_cleanup(out)


def merge_fcd_emission(fcd: pd.DataFrame, em: pd.DataFrame) -> pd.DataFrame:
    """
    Spaja FCD i emission po "(timestep_time, vehicle_id)" (inner join).

    U spojeni skup ulaze emisije, brzina i tip vozila iz emission grane.
    """
    f = normalize_fcd_like(fcd)[0]
    e = normalize_emission_like(em)
    key = ["timestep_time", "vehicle_id"]
    extra = [
        c
        for c in [
            "vehicle_CO2",
            "vehicle_NOx",
            "vehicle_PMx",
            "vehicle_speed",
            "vehicle_type",
            "vehicle_waiting",
        ]
        if c in e.columns
    ]
    merged = f.merge(e[key + extra], on=key, how="inner", suffixes=("", "_em"))
    return apply_vehicle_name_cleanup(merged)

"""
MobiML tab: agregacija trajektorija, H3 zone, klasifikacija tipa vozila, anomalije.
Klasifikacija i detekcija anomalija koriste scikit-learn na MobiML osobinama.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mobility_lab.geo_utils import vectorized_bearing_deg
from mobility_lab.map_utils import (
    HOTSPOT_CMAP_HEX,
    fmt_legend_value,
    h3_cell_center,
    h3_cell_polygon,
    h3_latlng_to_cell,
    linear_colormap,
)
from mobility_lab.vehicle_features import (
    enrich_trajectory_features,
    merge_emission_into_features,
    mode_or_first,
)

ROOT = Path(__file__).resolve().parents[1]
MOBIML_SRC = ROOT / "MobiML" / "src"
TRAJ_FEATURE_COLS = [
    "speed_max",
    "speed_median",
    "speed_mean",
    "length_m",
    "duration_s",
    "displacement_km",
    "speed_max_over_median",
]
_CLASSIFICATION_COLS = [
    "speed_max",
    "speed_median",
    "speed_mean",
    "speed_start",
    "speed_end",
    "length_m",
    "duration_s",
    "n_points",
    "displacement_km",
    "speed_max_over_median",
    "co2_mean",
    "co2_sum",
    "co2_per_km",
    "waiting_max",
    "waiting_mean",
    "rog_km",
]


def _ensure_mobiml_path() -> bool:
    """Dodaje "MobiML/src" u "sys.path" i proverava da li se "mobiml" može importovati."""
    src = str(MOBIML_SRC.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import mobiml  # noqa: F401
        return True
    except Exception:
        return False


def fcd_to_points_gdf(
    fcd_n: pd.DataFrame,
    merged: pd.DataFrame | None,
    is_geo: bool,
    lat0: float,
    lon0: float,
    max_points: int = 120_000,
) -> gpd.GeoDataFrame:
    """
    FCD tačke kao GeoDataFrame: brzina, smer, opciono CO₂/čekanje iz spojenog skupa.
    Ograničava broj tačaka na "max_points" radi performansi.
    """
    from mobility_lab.folium_map import ensure_lat_lon_columns
    df = ensure_lat_lon_columns(fcd_n.copy(), is_geo, lat0, lon0)
    if "latitude" in df.columns:
        df["y"] = df["latitude"].astype(float)
        df["x"] = df["longitude"].astype(float)
    elif "lat" in df.columns:
        df["y"] = df["lat"].astype(float)
        df["x"] = df["lon"].astype(float)
    else:
        df["x"] = df["vehicle_x"].astype(float)
        df["y"] = df["vehicle_y"].astype(float)
    if "vehicle_speed" not in df.columns:
        df["vehicle_speed"] = 0.0
    df["vehicle_speed"] = pd.to_numeric(df["vehicle_speed"], errors="coerce").fillna(0)
    if merged is not None and not merged.empty:
        m = ensure_lat_lon_columns(merged.copy(), is_geo, lat0, lon0)
        extra = [
            c
            for c in ["vehicle_CO2", "vehicle_waiting", "vehicle_type"]
            if c in m.columns
        ]
        lat_col = "latitude" if "latitude" in m.columns else "lat"
        lon_col = "longitude" if "longitude" in m.columns else "lon"
        if extra and lat_col in m.columns:
            msub = m[
                ["timestep_time", "vehicle_id", lat_col, lon_col] + extra
            ].drop_duplicates(subset=["timestep_time", "vehicle_id"])
            df = df.merge(
                msub,
                on=["timestep_time", "vehicle_id"],
                how="left",
            )
    df = df.sort_values(["vehicle_id", "timestep_time"])
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42).sort_values(
            ["vehicle_id", "timestep_time"]
        )
    px = df.groupby("vehicle_id")["x"].shift(1)
    py = df.groupby("vehicle_id")["y"].shift(1)
    df["direction"] = vectorized_bearing_deg(
        px.to_numpy(), py.to_numpy(), df["x"].to_numpy(), df["y"].to_numpy()
    )
    geom = gpd.points_from_xy(df["x"], df["y"])
    crs = "EPSG:4326" if is_geo else None
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=crs)
    gdf = gdf.rename(
        columns={
            "vehicle_id": "mover_id",
            "vehicle_speed": "speed",
            "t": "timestamp",
        }
    )
    gdf["traj_id"] = gdf["mover_id"].astype(str)
    return gdf


def _mobiml_aggregate(
    fcd_n: pd.DataFrame,
    merged: pd.DataFrame,
    is_geo: bool,
    lat0: float,
    lon0: float,
    h3_resolution: int,
    max_vehicles: int,
) -> tuple[pd.DataFrame, str]:
    """
    MobiML "TrajectoryCreator" + "TrajectoryAggregator": agregirane trajektorije po vozilu.
    Vraća DataFrame osobina i kratku napomenu (broj trajektorija, H3 rezolucija).
    """
    from mobiml.datasets.aisdk import SHIPTYPE
    from mobiml.datasets.utils import DIRECTION, MOVER_ID, SPEED, TIMESTAMP, TRAJ_ID
    from mobiml.transforms.traj_aggregator import TrajectoryAggregator
    from mobiml.transforms.traj_creator import TrajectoryCreator
    from mobility_lab.io_sumo import simplify_vehicle_type
    gdf = fcd_to_points_gdf(fcd_n, merged, is_geo, lat0, lon0)
    gdf = gdf.rename(
        columns={
            "mover_id": MOVER_ID,
            "traj_id": TRAJ_ID,
            "speed": SPEED,
            "direction": DIRECTION,
            "timestamp": TIMESTAMP,
        }
    )
    gdf["client"] = 0
    vids = gdf[MOVER_ID].astype(str).unique()
    if len(vids) > max_vehicles:
        keep = set(
            np.random.default_rng(42).choice(vids, size=max_vehicles, replace=False)
        )
        gdf = gdf[gdf[MOVER_ID].astype(str).isin(keep)]
    trajs = TrajectoryCreator(gdf, min_length=50).get_trajs(
        gap_duration=timedelta(minutes=10),
        generalization_tolerance=timedelta(seconds=30),
    )
    vessels = merged.groupby("vehicle_id")["vehicle_type"].agg(mode_or_first)
    vessels = vessels.map(simplify_vehicle_type).rename(SHIPTYPE)
    vessels.index = vessels.index.astype(str)
    agg = TrajectoryAggregator(trajs, vessels).aggregate_trajs(h3_resolution)
    agg = agg.reset_index(drop=True)
    agg["vehicle_id"] = agg[MOVER_ID].astype(str)
    agg["vehicle_type"] = agg[SHIPTYPE].astype(str)
    xs, ys = agg["x_start"].astype(float), agg["y_start"].astype(float)
    xe, ye = agg["x_end"].astype(float), agg["y_end"].astype(float)
    scale = 111_320.0 if is_geo else 1.0
    agg["length_m"] = np.hypot(xe - xs, ye - ys) * scale
    note = f"MobiML TrajectoryAggregator (h3={h3_resolution}), {len(agg)} trajektorija"
    return agg, note


def h3_planning_hotspots(
    merged: pd.DataFrame,
    is_geo: bool,
    lat0: float,
    lon0: float,
    resolution: int = 9,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    H3 heksagoni rangirani po CO₂ i čekanju — kandidati za „vruće zone“ u planiranju.
    "planner_score" = co2_sum + 0.5 × waiting_sum; vraća top "top_n" ćelija.
    """
    from mobility_lab.folium_map import ensure_lat_lon_columns
    m = ensure_lat_lon_columns(merged, is_geo, lat0, lon0)
    lat_col = "latitude" if "latitude" in m.columns else "lat"
    lon_col = "longitude" if "longitude" in m.columns else "lon"
    m = m.dropna(subset=[lat_col, lon_col])
    if len(m) > 80_000:
        m = m.sample(80_000, random_state=42)
    m = m.copy()
    m["h3_cell"] = [
        h3_latlng_to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(m[lat_col], m[lon_col])
    ]
    agg_spec: dict[str, tuple[str, str]] = {"point_count": (lat_col, "count")}
    if "vehicle_CO2" in m.columns:
        agg_spec["co2_sum"] = ("vehicle_CO2", "sum")
    if "vehicle_waiting" in m.columns:
        agg_spec["waiting_sum"] = ("vehicle_waiting", "sum")
    out = m.groupby("h3_cell", as_index=False).agg(**agg_spec)
    if out.empty:
        return out
    centers = out["h3_cell"].map(h3_cell_center)
    out["lat"] = centers.map(lambda t: t[0])
    out["lon"] = centers.map(lambda t: t[1])
    if "co2_sum" not in out.columns:
        out["co2_sum"] = 0.0
    if "waiting_sum" not in out.columns:
        out["waiting_sum"] = 0.0
    out["planner_score"] = out["co2_sum"] + 0.5 * out["waiting_sum"]
    return out.sort_values("planner_score", ascending=False).head(top_n)


def folium_h3_hotspots_map(hotspots: pd.DataFrame, *, zoom_start: int = 14):
    """
    Folium mapa: H3 heksagoni obojeni po "planner_score" (CO₂ + čekanje).
    Tamnija crvena = veći skor; nije point heatmap već poligoni ćelija.
    """
    import folium
    if hotspots is None or hotspots.empty:
        return folium.Map(location=[43.3209, 21.8958], zoom_start=zoom_start)
    hf = hotspots.copy()
    m = folium.Map(
        location=[float(hf["lat"].mean()), float(hf["lon"].mean())],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )
    vmax = float(hf["planner_score"].max()) or 1.0
    vmin = float(hf["planner_score"].min())
    mid = vmin + (vmax - vmin) / 2
    colormap = linear_colormap(HOTSPOT_CMAP_HEX, vmin, vmax)
    colormap.caption = "Više = više CO₂ i čekanja u zoni"
    colormap.tick_labels = [
        fmt_legend_value(vmin),
        fmt_legend_value(mid),
        fmt_legend_value(vmax),
    ]
    colormap.add_to(m)
    for rank, row in enumerate(hf.itertuples(index=False), start=1):
        score = float(row.planner_score)
        folium.Polygon(
            locations=h3_cell_polygon(row.h3_cell),
            color="#333333",
            weight=1,
            fill=True,
            fill_color=colormap(score),
            fill_opacity=0.55,
            popup=folium.Popup(
                f"<b>#{rank}</b> H3 {row.h3_cell}<br>"
                f"Planner score: {score:,.0f}<br>"
                f"CO₂ sum: {row.co2_sum:,.0f}<br>"
                f"Waiting sum: {row.waiting_sum:,.0f}<br>"
                f"Tačaka u zoni: {row.point_count:,}",
                max_width=280,
            ),
        ).add_to(m)
    return m


def _classification_feature_columns(df: pd.DataFrame) -> list[str]:
    """Lista kolona dostupnih u "df" za klasifikaciju (bez sirovih x/y koordinata)."""
    return [c for c in _CLASSIFICATION_COLS if c in df.columns]


def _build_classifier_pipelines() -> list[tuple[str, Pipeline]]:
    """RandomForest i logistička regresija u sklearn Pipeline (skaliranje + model)."""
    return [
        (
            "RandomForest",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=14,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            C=0.5,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]


def classify_vehicle_types(features: pd.DataFrame) -> dict[str, Any]:
    """
    Predviđa "vehicle_type" iz osobina trajektorije i emisija (RandomForest ili logistička regresija).
    Vraća metrike (accuracy, F1), confusion matrix i predikcije po vozilu.
    """
    if "vehicle_id" not in features.columns:
        features = features.reset_index(drop=True)
    df = enrich_trajectory_features(features)
    df = df[df["vehicle_type"].astype(str).str.len() > 0]
    df = df[~df["vehicle_type"].astype(str).str.lower().isin(["unknown", "nan", ""])]
    feat_cols = _classification_feature_columns(df)
    if len(feat_cols) < 4:
        return {"ok": False, "message": "Premalo kolona za klasifikaciju."}
    df = df.dropna(subset=["vehicle_type"])
    labels = sorted(df["vehicle_type"].astype(str).unique().tolist())
    min_class = int(df["vehicle_type"].value_counts().min())
    if len(labels) < 2 or len(df) < max(15, len(labels) * 4):
        return {
            "ok": False,
            "message": f"Premalo uzoraka za klasifikaciju ({len(df)} vozila, {len(labels)} tipova).",
        }
    X = df[feat_cols].astype(float).fillna(0.0)
    le = LabelEncoder()
    y = le.fit_transform(df["vehicle_type"].astype(str))
    stratify = y if min_class >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    cv_folds = min(5, min_class) if min_class >= 2 else 0
    pipelines = _build_classifier_pipelines()
    best_name, best_pipe, best_cv = pipelines[0][0], pipelines[0][1], -1.0
    for name, pipe in pipelines:
        if cv_folds >= 3:
            cv_scores = cross_val_score(
                pipe, X_train, y_train, cv=cv_folds, scoring="f1_macro", n_jobs=-1
            )
            score = float(cv_scores.mean())
        else:
            pipe.fit(X_train, y_train)
            score = float(
                f1_score(
                    y_train, pipe.predict(X_train), average="macro", zero_division=0
                )
            )
        if score > best_cv:
            best_cv, best_name, best_pipe = score, name, pipe
    best_pipe.fit(X_train, y_train)
    pred = best_pipe.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    per_class = f1_score(
        y_test, pred, average=None, labels=range(len(le.classes_)), zero_division=0
    )
    per_class_f1 = {
        str(le.classes_[i]): float(per_class[i]) for i in range(len(le.classes_))
    }
    present = sorted(set(y_test) | set(pred))
    cm = confusion_matrix(y_test, pred, labels=present)
    out_df = df[["vehicle_id", "vehicle_type"]].copy()
    out_df["predicted_type"] = le.inverse_transform(best_pipe.predict(X))
    return {
        "ok": True,
        "model": best_name,
        "cv_macro_f1": best_cv,
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "predictions": out_df.to_dict(orient="records"),
    }


def detect_anomalies(features: pd.DataFrame) -> dict[str, Any]:
    """
    Detekcija anomalija: IsolationForest na mobilnim i emisionim osobinama.
    Vraća do 40 vozila označenih kao anomalna (neobičan obrazac kretanja/emisija).
    """
    df = enrich_trajectory_features(features.dropna(subset=["vehicle_id"]).copy())
    extra = [
        "co2_mean",
        "co2_sum",
        "co2_per_km",
        "waiting_max",
        "waiting_mean",
        "rog_km",
        "n_points",
    ]
    cols = [c for c in TRAJ_FEATURE_COLS + extra if c in df.columns]
    if len(cols) < 3:
        cols = _classification_feature_columns(df)
    if len(df) < 8:
        return {"ok": False, "message": "Premalo vozila za detekciju anomalija."}
    if len(cols) < 3:
        return {"ok": False, "message": "Premalo kolona za detekciju anomalija."}
    X = df[cols].astype(float).fillna(0)
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination="auto", random_state=42)
    scores = iso.fit_predict(Xs)
    df = df.copy()
    df["anomaly"] = scores == -1
    df["anomaly_rank"] = iso.decision_function(Xs)
    flagged = df[df["anomaly"]].sort_values("anomaly_rank").head(40)
    return {
        "ok": True,
        "n_anomalies": int(df["anomaly"].sum()),
        "anomalies": flagged[
            [
                "vehicle_id",
                "vehicle_type",
                "anomaly_rank",
                "co2_mean",
                "waiting_max",
                "length_m",
            ]
        ].to_dict(orient="records"),
    }


def run_pipeline(
    fcd_n: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    is_geo: bool,
    lat0: float,
    lon0: float,
    max_vehicles: int = 150,
    h3_resolution: int = 9,
    lengths: pd.Series | None = None,
    rog: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Ceo pipeline MobiML taba: osobine, klasifikacija, anomalije, H3 hotspoti.
    Vraća dictionary pogodan za JSON (Streamlit prikaz i "save_results").
    """
    if not _ensure_mobiml_path():
        raise RuntimeError(
            "MobiML nije dostupan — proveri folder MobiML/src u korenu projekta."
        )
    try:
        agg_df, mobiml_note = _mobiml_aggregate(
            fcd_n, merged, is_geo, lat0, lon0, h3_resolution, max_vehicles
        )
    except Exception as exc:
        raise RuntimeError(f"MobiML agregacija nije uspela: {exc}") from exc
    features = merge_emission_into_features(
        agg_df, merged, lengths=lengths, rog=rog
    )
    classification = classify_vehicle_types(features)
    anomalies = detect_anomalies(features)
    hotspots = h3_planning_hotspots(
        merged, is_geo, lat0, lon0, resolution=h3_resolution
    )
    return {
        "mobiml_note": mobiml_note,
        "n_vehicles": int(len(features)),
        "feature_columns": list(features.columns),
        "features_preview": features.head(25).to_dict(orient="records"),
        "classification": classification,
        "anomalies": anomalies,
        "hotspots": hotspots.to_dict(orient="records") if not hotspots.empty else [],
    }


def save_results(payload: dict[str, Any], out_dir: Path) -> Path:
    """Upisuje rezultat pipeline-a u "mobiml_results.json"; vraća putanju fajla."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mobiml_results.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path

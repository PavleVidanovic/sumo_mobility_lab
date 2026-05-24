"""
Geometrijske pomoćne funkcije (smer kretanja iz koordinata).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vectorized_bearing_deg(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    """Azimut (0–360°) između parova tačaka — vektorizovano preko numpy."""
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(rlat2)
    y = np.cos(rlat1) * np.sin(rlat2) - np.sin(rlat1) * np.cos(rlat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    missing = pd.isna(lon1) | pd.isna(lat1) | pd.isna(lon2) | pd.isna(lat2)
    bearing = np.where(missing, 0.0, bearing)
    return bearing

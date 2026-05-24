#!/usr/bin/env python3
"""
Fazni saobraćaj (busy / light) po tipu vozila, 0–3600 s, --insertion-density u svakoj fazi (depart unutar [begin, end]).
Pokreće se iz build.bat posle netconvert.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIM_END = 3600
NET = "osm.net.xml.gz"
SUMO_HOME = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"))
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"

VIA = (
    "highway.motorway,highway.motorway_link,highway.trunk_link,"
    "highway.primary_link,highway.secondary_link,highway.tertiary_link"
)
TRIP_ATTR = 'departLane="best"'
FRINGE_ATTR = 'departSpeed="max"'

# insertion-density (busy / light) po klasi — light ≈ 25–30% busy
VEHICLE_SPECS: list[dict] = [
    {
        "name": "passenger",
        "out": "osm.passenger.rou.xml",
        "args": [
            "--fringe-factor", "15",
            "--vehicle-class", "passenger",
            "--vclass", "passenger",
            "--min-distance", "300",
            "--min-distance.fringe", "10",
            "--allow-fringe.min-length", "1000",
            "--lanes",
            "--seed", "45",
        ],
        "busy": 52.0,
        "light": 14.0,
    },
    {
        "name": "truck",
        "out": "osm.truck.rou.xml",
        "args": [
            "--fringe-factor", "10",
            "--vehicle-class", "truck",
            "--vclass", "truck",
            "--min-distance", "600",
            "--min-distance.fringe", "10",
            "--seed", "47",
        ],
        "busy": 10.0,
        "light": 3.0,
    },
    {
        "name": "bus",
        "out": "osm.bus.rou.xml",
        "args": [
            "--fringe-factor", "15",
            "--vehicle-class", "bus",
            "--vclass", "bus",
            "--min-distance", "600",
            "--min-distance.fringe", "10",
            "--seed", "43",
        ],
        "busy": 6.0,
        "light": 2.0,
    },
    {
        "name": "motorcycle",
        "out": "osm.motorcycle.rou.xml",
        "args": [
            "--fringe-factor", "4",
            "--vehicle-class", "motorcycle",
            "--vclass", "motorcycle",
            "--max-distance", "1200",
            "--seed", "44",
        ],
        "busy": 8.0,
        "light": 2.5,
    },
    {
        "name": "bicycle",
        "out": "osm.bicycle.rou.xml",
        "args": [
            "--fringe-factor", "3",
            "--vehicle-class", "bicycle",
            "--vclass", "bicycle",
            "--max-distance", "8000",
            "--seed", "42",
        ],
        "busy": 9.0,
        "light": 2.5,
    },
]

PEDESTRIAN_SPEC = {
    "name": "pedestrian",
    "out": "osm.pedestrian.rou.xml",
    "busy": 140.0,
    "light": 45.0,
    "seed": "46",
}


def generate_phases(
    duration: int = SIM_END,
    *,
    seed: int = 20260522,
    min_len: int = 280,
    max_len: int = 720,
) -> list[dict]:
    """Naizmenični busy/light segmenti, nasumične dužine (ne fiksno po 900 s)."""
    rng = random.Random(seed)
    t = 0.0
    phases: list[dict] = []
    busy = True
    while t < duration - 1:
        length = rng.randint(min_len, max_len)
        end = min(t + length, float(duration))
        phases.append(
            {
                "begin": round(t, 1),
                "end": round(end, 1),
                "mode": "busy" if busy else "light",
            }
        )
        t = end
        busy = not busy
    if phases and phases[-1]["end"] < duration:
        phases.append(
            {
                "begin": phases[-1]["end"],
                "end": float(duration),
                "mode": "busy" if not busy else "light",
            }
        )
    return phases


def merge_route_files(parts: list[Path], out_path: Path) -> None:
    """Spoji više .rou.xml u jedan (jedinstveni vType, svi trip/flow/person)."""
    routes = ET.Element("routes")
    vtypes: set[str] = set()
    for part in parts:
        if not part.is_file():
            continue
        tree = ET.parse(part)
        for child in tree.getroot():
            tag = child.tag
            if tag == "vType":
                vid = child.get("id")
                if vid and vid not in vtypes:
                    routes.append(child)
                    vtypes.add(vid)
            elif tag in ("trip", "flow", "vehicle", "person", "personFlow", "route"):
                routes.append(child)
    tree_out = ET.ElementTree(routes)
    ET.indent(tree_out, space="    ")
    tree_out.write(out_path, encoding="UTF-8", xml_declaration=True)


def run_random_trips_flow(
    spec: dict,
    phase: dict,
    phase_idx: int,
    *,
    persontrips: bool = False,
) -> Path:
    density = spec["busy"] if phase["mode"] == "busy" else spec["light"]
    prefix = f"{spec['name']}_"
    tmp = ROOT / f"_tmp_{spec['name']}_phase{phase_idx}.rou.xml"
    cmd = [
        sys.executable,
        str(RANDOM_TRIPS),
        "-n",
        str(ROOT / NET),
        "--insertion-density",
        str(density),
        "-b",
        str(phase["begin"]),
        "-e",
        str(phase["end"]),
        "-r",
        str(tmp),
        "--prefix",
        prefix,
        "--validate",
        "--remove-loops",
    ]
    if persontrips:
        cmd += [
            "--fringe-factor",
            "2",
            "--persontrips",
            "--vehicle-class",
            "pedestrian",
            "--trip-attributes",
            'modes="public"',
            "--additional-files",
            "osm_stops.add.xml,osm_pt.rou.xml",
            "--persontrip.walk-opposite-factor",
            "0.8",
            "--duarouter-weights.tls-penalty",
            "20",
            "--seed",
            PEDESTRIAN_SPEC["seed"],
        ]
    else:
        cmd += [
            "--trip-attributes",
            TRIP_ATTR,
            "--fringe-start-attributes",
            FRINGE_ATTR,
            "--via-edge-types",
            VIA,
            *spec.get("args", []),
        ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return tmp


def build_vehicle_class(spec: dict, phases: list[dict]) -> None:
    parts: list[Path] = []
    for i, phase in enumerate(phases):
        parts.append(run_random_trips_flow(spec, phase, i))
    merge_route_files(parts, ROOT / spec["out"])
    for p in parts:
        try:
            p.unlink()
        except OSError:
            pass
    print(f"Wrote {spec['out']}", flush=True)


def build_pedestrian(phases: list[dict]) -> None:
    spec = PEDESTRIAN_SPEC
    parts: list[Path] = []
    for i, phase in enumerate(phases):
        parts.append(run_random_trips_flow(spec, phase, i, persontrips=True))
    merge_route_files(parts, ROOT / spec["out"])
    for p in parts:
        try:
            p.unlink()
        except OSError:
            pass
    print(f"Wrote {spec['out']}", flush=True)


def main() -> None:
    if not RANDOM_TRIPS.is_file():
        sys.exit(f"randomTrips.py not found: {RANDOM_TRIPS} (set SUMO_HOME)")
    if not (ROOT / NET).is_file():
        sys.exit(f"Network missing: {ROOT / NET} — run netconvert first.")

    phases = generate_phases()
    (ROOT / "demand_phases.json").write_text(
        json.dumps(phases, indent=2),
        encoding="utf-8",
    )
    print(f"Phases ({len(phases)}):", phases, flush=True)

    for spec in VEHICLE_SPECS:
        build_vehicle_class(spec, phases)
    build_pedestrian(phases)
    print("Demand build done.", flush=True)


if __name__ == "__main__":
    main()

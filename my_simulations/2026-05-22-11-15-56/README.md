# Scenario `2026-05-22-11-15-56` (Niš, 1 h)

## Area

- Bbox (WGS84): `21.768897, 43.307693` → `21.975000, 43.375000`
- Extended **east** (Medijana / Stop Shop retail park direction) vs the 2026-04-12 scenario.

## Simulation

- Duration: **3600 s** (`osm.sumocfg` `<time>`)
- Demand: **flows** per phase (`build_demand.py`), new vehicles inserted throughout each segment
- Traffic pattern: alternating **busy / light** segments with random lengths (~280–720 s), **per vehicle class** (see `demand_phases.json` after build)

## Build (once)

```bat
build.bat
```

Requires `SUMO_HOME`, `netconvert`, and Python on PATH.

## Run

| Script | Purpose |
|--------|---------|
| `run.bat` | **sumo-gui** — visual run, outputs under `output/` |
| `run_batch.bat` | **sumo** headless — same outputs, for long runs |

Outputs (for Mobility Lab):

- `output/run_fcd.xml` (+ geo)
- `output/run_emission.xml`
- `output/tripinfos.xml`, `stopinfos.xml`, `stats.xml`
- `edgeData.xml` (from `output.add.xml`)

## Streamlit

Point **Korak 1** sumocfg to this folder’s `osm.sumocfg`, or copy `output/run_*.xml` to `sumo_output/` and convert with xml2csv.

Default paths in app still point at `2026-04-12-21-02-35` until you change them.

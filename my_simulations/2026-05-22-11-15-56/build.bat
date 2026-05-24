@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not defined SUMO_HOME set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"
set "PYTHON=python"

echo === SUMO scenario build: 2026-05-22-11-15-56 ===
echo Bbox: west=21.768897 south=43.307693 east=21.975 north=43.375 (extended east toward Stop Shop / Medijana)
echo Simulation: 3600 s, phased busy/light demand per vehicle class
echo.

if not exist output mkdir output

echo [1/5] Download OSM ...
"%PYTHON%" "%SUMO_HOME%\tools\osmGet.py" --bbox 21.768897,43.307693,21.975000,43.375000 -p osm_bbox -z
if errorlevel 1 goto :fail

echo [2/5] Build network (netconvert) ...
netconvert -c osm.netccfg
if errorlevel 1 goto :fail

echo [3/5] Public transport flows ...
"%PYTHON%" "%SUMO_HOME%\tools\ptlines2flows.py" -n osm.net.xml.gz -b 0 -e 3600 -p 600 --random-begin --seed 42 --ptstops osm_stops.add.xml --ptlines osm_ptlines.xml -o osm_pt.rou.xml --ignore-errors --vtype-prefix pt_ --min-stops 0 --extend-to-fringe --verbose
if errorlevel 1 goto :fail

echo [4/5] Phased road traffic (build_demand.py) ...
"%PYTHON%" build_demand.py
if errorlevel 1 goto :fail

echo [5/5] Done. Run: run.bat (sumo-gui) or run_batch.bat (sumo + outputs in output\)
goto :ok

:fail
echo BUILD FAILED.
exit /b 1

:ok
exit /b 0

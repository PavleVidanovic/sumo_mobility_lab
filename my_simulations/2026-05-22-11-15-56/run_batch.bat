@echo off
cd /d "%~dp0"
if not defined SUMO_HOME set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"
if not exist output mkdir output
echo Running headless SUMO (writes output\run_*.xml) ...
sumo -c osm.sumocfg --no-step-log
echo Done. Convert XML to CSV with xml2csv for Streamlit.

@echo off
cd /d "%~dp0"
if not defined SUMO_HOME set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"
if not exist output mkdir output
sumo-gui -c osm.sumocfg --start
